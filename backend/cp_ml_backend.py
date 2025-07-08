import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from collections import defaultdict
from datetime import datetime, timedelta

from ortools.sat.python import cp_model

# ✅ Tell Flask the correct template path: cp and model/templates
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_path, 'templates')
app = Flask(__name__, template_folder=template_dir)

# Default Weights
DEFAULT_DEADLINE_WEIGHT = 5.0
DURATION_WEIGHT = 1.0
SETUP_WEIGHT = 1.0

# ✅ Model paths (must be in same folder as this file)
reg_path = os.path.join(os.path.dirname(__file__), 'ml_modelnew.pkl')
clf_path = os.path.join(os.path.dirname(__file__), 'ml_classifier.pkl')

reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None
clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None

def run_scheduler(data):
    mode = data.get('mode', 'cp')
    deadline_weight = float(data.get('deadline_weight', DEFAULT_DEADLINE_WEIGHT))
    solver_time_limit = float(data.get('solver_time_limit', 0))

    machines = data['machines']
    product_tasks = data['product_tasks']
    setup_times = data['setup_times']
    orders_data = data['orders']
    start_dt = datetime.fromisoformat(data['start_datetime'])

    orders = []
    for od in orders_data:
        orders.append((od['quantities'], datetime.fromisoformat(od['deadline'])))

    tasks = []
    for oid, (order, dl) in enumerate(orders):
        for prod, qty in order.items():
            chain = product_tasks[prod]
            for q in range(qty):
                for idx, (m, dur) in enumerate(chain):
                    tasks.append({
                        'order': oid,
                        'product': prod,
                        'task_index': idx,
                        'machine': m,
                        'duration': dur,
                        'deadline': dl
                    })

    if mode == 'ml_only' and reg_model:
        return run_ml_scheduler(tasks, machines, start_dt, setup_times)

    machine_to_idx = {m: i for i, m in enumerate(machines)}
    horizon = int(sum(t['duration'] + setup_times.get(t['product'], {}).get(t['product'], 0) for t in tasks) * 60 + 1)

    model = cp_model.CpModel()
    start_vars, end_vars, intervals = [], [], []

    for i, t in enumerate(tasks):
        dur_min = int(t['duration'] * 60)
        s = model.NewIntVar(0, horizon, f's{i}')
        e = model.NewIntVar(0, horizon, f'e{i}')
        iv = model.NewIntervalVar(s, dur_min, e, f'iv{i}')
        start_vars.append(s)
        end_vars.append(e)
        intervals.append(iv)

    for oid, (order, dl) in enumerate(orders):
        for prod, qty in order.items():
            chain = product_tasks[prod]
            for q in range(qty):
                unit_indices = [i for i, t in enumerate(tasks) if t['order'] == oid and t['product'] == prod and t['task_index'] in range(len(chain))]
                for k in range(len(chain) - 1):
                    i1 = next(i for i in unit_indices if tasks[i]['task_index'] == k)
                    i2 = next(i for i in unit_indices if tasks[i]['task_index'] == k + 1)
                    model.Add(start_vars[i2] >= end_vars[i1])

    for m in machines:
        machine_intervals = [intervals[i] for i, t in enumerate(tasks) if t['machine'] == m]
        model.AddNoOverlap(machine_intervals)

    order_completion = []
    lateness_vars = []
    for oid, (_, dl) in enumerate(orders):
        ends = [end_vars[i] for i, t in enumerate(tasks) if t['order'] == oid and t['task_index'] == len(product_tasks[t['product']]) - 1]
        cmax = model.NewIntVar(0, horizon, f'order_end_{oid}')
        model.AddMaxEquality(cmax, ends)
        order_completion.append(cmax)
        dl_offset = int((dl - start_dt).total_seconds() / 60)
        late = model.NewIntVar(0, horizon, f'late_{oid}')
        model.Add(late >= cmax - dl_offset)
        model.Add(late >= 0)
        lateness_vars.append(late)

    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, order_completion)
    model.Minimize(makespan + sum(lateness_vars))

    solver = cp_model.CpSolver()
    if solver_time_limit > 0:
        solver.parameters.max_time_in_seconds = solver_time_limit
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    machine_schedule = {m: [] for m in machines}
    for i, t in enumerate(tasks):
        m = t['machine']
        s = solver.Value(start_vars[i])
        e = solver.Value(end_vars[i])
        st = start_dt + timedelta(minutes=s)
        en = start_dt + timedelta(minutes=e)
        machine_schedule[m].append({
            'order': t['order'] + 1,
            'product': t['product'],
            'task_index': t['task_index'] + 1,
            'start': st.isoformat(sep=' '),
            'end': en.isoformat(sep=' ')
        })

    orders_summary = []
    for oid, (_, dl) in enumerate(orders):
        fin = start_dt + timedelta(minutes=solver.Value(order_completion[oid]))
        late = (fin - dl).total_seconds() / 3600
        orders_summary.append({
            'order': oid + 1,
            'finished': fin.isoformat(sep=' '),
            'deadline': dl.isoformat(sep=' '),
            'status': 'Late' if late > 0 else 'On Time',
            'hours_overdue': round(late, 2) if late > 0 else 0
        })

    return {'machine_schedule': machine_schedule, 'orders_summary': orders_summary}

def run_ml_scheduler(tasks, machines, start_dt, setup_times):
    if not reg_model:
        return {"error": "ML model not loaded."}

    machine_schedule = {m: [] for m in machines}
    machine_available = {m: start_dt for m in machines}
    product_last_end = defaultdict(lambda: start_dt)
    last_product_on_machine = {m: None for m in machines}

    for task in sorted(tasks, key=lambda t: t['order']):
        m = task['machine']
        dur = task['duration']
        prev = last_product_on_machine[m]
        setup_hr = setup_times.get(prev, {}).get(task['product'], 0) if prev and prev != task['product'] else 0
        ready = max(machine_available[m], product_last_end[(task['order'], task['product'])])
        machine_wait = (ready - start_dt).total_seconds() / 3600
        urgency = (task['deadline'] - ready).total_seconds() / 3600

        features = np.array([[dur, setup_hr, machine_wait, urgency]])
        score = reg_model.predict(features)[0]
        start = ready + timedelta(hours=setup_hr)
        end = start + timedelta(hours=dur)

        machine_schedule[m].append({
            'order': task['order'] + 1,
            'product': task['product'],
            'task_index': task['task_index'] + 1,
            'start': start.isoformat(sep=' '),
            'end': end.isoformat(sep=' ')
        })
        machine_available[m] = end
        product_last_end[(task['order'], task['product'])] = end
        last_product_on_machine[m] = task['product']

    return {'machine_schedule': machine_schedule, 'orders_summary': []}

@app.route('/api/schedule', methods=['POST'])
def api_schedule():
    payload = request.get_json()
    result = run_scheduler(payload)
    return jsonify(result)

@app.route('/')
def home():
    return render_template('frontend.html')

if __name__ == '__main__':
    app.run(debug=True)
