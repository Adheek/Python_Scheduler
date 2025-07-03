import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from collections import defaultdict
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

# Flask setup
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_path, 'templates')
app = Flask(__name__, template_folder=template_dir)

# Load ML models (unused here)
reg_path = os.path.join(os.path.dirname(__file__), 'ml_modelnew.pkl')
clf_path = os.path.join(os.path.dirname(__file__), 'ml_classifier.pkl')
reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None
clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None

def run_scheduler(data):
    operations     = data['operations']
    machine_types  = data['machine_types']
    product_tasks  = data['product_tasks']
    setup_times    = data['setup_times']
    orders_input   = data['orders']
    start_dt       = datetime.fromisoformat(data['start_datetime'])

    machines = []
    operation_to_units = defaultdict(list)
    for mt in machine_types:
        mname = mt["name"]
        count = int(mt["count"])
        ops = mt.get("operations", operations)
        for i in range(count):
            unit = f"{mname}[{i+1}]"
            machines.append(unit)
            for op in ops:
                operation_to_units[op].append(unit)

    ops = []
    deadlines = []
    for oid, od in enumerate(orders_input):
        dl_dt = datetime.fromisoformat(od['deadline'])
        dl_off = int((dl_dt - start_dt).total_seconds() / 3600)
        deadlines.append((oid, dl_off, dl_dt))
        for prod, qty in od['quantities'].items():
            chain = product_tasks[prod]
            for unit in range(qty):
                for step, (operation, dur) in enumerate(chain):
                    ops.append({
                        'order': oid,
                        'uid': f"{oid}_{prod}_{unit}_{step}",
                        'operation': operation,
                        'duration': int(dur),
                        'product': f"{prod}({unit + 1})",
                        'product_type': prod,
                        'step': step
                    })

    total_work = sum(o['duration'] for o in ops)
    max_dead = max(dl for _, dl, _ in deadlines) if deadlines else 0
    horizon = total_work + max_dead + 1

    model = cp_model.CpModel()

    for o in ops:
        o['alt_intervals'] = []
        o['pres_vars'] = []
        o['main_start'] = model.NewIntVar(0, horizon, f"main_start_{o['uid']}")
        o['main_end'] = model.NewIntVar(0, horizon, f"main_end_{o['uid']}")
        for unit in operation_to_units[o['operation']]:
            pres = model.NewBoolVar(f"pres_{o['uid']}_{unit}")
            start = model.NewIntVar(0, horizon, f"s_{o['uid']}_{unit}")
            end = model.NewIntVar(0, horizon, f"e_{o['uid']}_{unit}")
            interval = model.NewOptionalIntervalVar(start, o['duration'], end, pres, f"iv_{o['uid']}_{unit}")
            o['alt_intervals'].append({'unit': unit, 'interval': interval, 'start': start, 'end': end, 'pres': pres})
            o['pres_vars'].append(pres)
            model.Add(o['main_start'] == start).OnlyEnforceIf(pres)
            model.Add(o['main_end'] == end).OnlyEnforceIf(pres)
        model.AddExactlyOne(o['pres_vars'])

    # Precedence
    chains = defaultdict(list)
    for o in ops:
        key = (o['order'], o['uid'].rsplit('_', 2)[0])
        chains[key].append(o)
    for chain_ops in chains.values():
        chain_ops.sort(key=lambda x: x['step'])
        for prev, nxt in zip(chain_ops, chain_ops[1:]):
            model.Add(nxt['main_start'] >= prev['main_end'])

    # Setup times
    for m in machines:
        tasks_on_m = []
        for o in ops:
            for alt in o['alt_intervals']:
                if alt['unit'] == m:
                    tasks_on_m.append((alt, o))

        for i in range(len(tasks_on_m)):
            for j in range(len(tasks_on_m)):
                if i == j:
                    continue
                a_alt, a_info = tasks_on_m[i]
                b_alt, b_info = tasks_on_m[j]
                from_prod = a_info['product_type']
                to_prod = b_info['product_type']
                if from_prod == to_prod:
                    continue
                setup = setup_times.get(from_prod, {}).get(to_prod, 0)
                if setup == 0:
                    continue
                before = model.NewBoolVar(f"{m}_before_{i}_{j}")
                model.Add(a_alt['end'] + setup <= b_alt['start']).OnlyEnforceIf(before)
                model.Add(b_alt['end'] <= a_alt['start']).OnlyEnforceIf(before.Not())
                model.AddBoolAnd([a_alt['pres'], b_alt['pres']]).OnlyEnforceIf(before)

    # NoOverlap
    for m in machines:
        m_intervals = [alt['interval'] for o in ops for alt in o['alt_intervals'] if alt['unit'] == m]
        if m_intervals:
            model.AddNoOverlap(m_intervals)

    # Lateness
    order_end = {}
    late_vars = []
    for oid, dl_off, _ in deadlines:
        ends = [o['main_end'] for o in ops if o['order'] == oid]
        oend = model.NewIntVar(0, horizon, f"order_end_{oid}")
        model.AddMaxEquality(oend, ends)
        order_end[oid] = oend
        late = model.NewIntVar(0, horizon, f"late_{oid}")
        model.Add(late >= oend - dl_off)
        model.Add(late >= 0)
        late_vars.append(late)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [o['main_end'] for o in ops])

    alpha = 1
    beta = 1
    model.Minimize(makespan + alpha * sum(late_vars) + beta * sum(late_vars))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {'error': 'No feasible schedule found.'}

    # Output formatting
    machine_schedule = {m: [] for m in machines}
    spans = []
    order_finish = defaultdict(lambda: start_dt)
    for o in ops:
        selected = None
        for alt in o['alt_intervals']:
            if solver.Value(alt['pres']):
                selected = alt
                break
        if not selected:
            continue
        st = solver.Value(selected['start'])
        en = solver.Value(selected['end'])
        dt_st = start_dt + timedelta(hours=st)
        dt_en = start_dt + timedelta(hours=en)
        machine_schedule[selected['unit']].append({
            'order': o['order'] + 1,
            'product': o['product'],
            'operation': o['operation'],
            'task_index': o['step'] + 1,
            'start': dt_st.isoformat(sep=' '),
            'end': dt_en.isoformat(sep=' ')
        })
        spans.append((selected['unit'], dt_st, dt_en))
        order_finish[o['order']] = max(order_finish[o['order']], dt_en)

    for m in machine_schedule:
        machine_schedule[m].sort(key=lambda x: x['start'])

    orders_summary = []
    for oid, _, dl_dt in deadlines:
        fin_dt = order_finish[oid]
        late_h = round((fin_dt - dl_dt).total_seconds() / 3600, 2)
        orders_summary.append({
            'order': oid + 1,
            'finished': fin_dt.isoformat(sep=' '),
            'deadline': dl_dt.isoformat(sep=' '),
            'status': 'On Time' if late_h <= 0 else 'Late',
            'hours_overdue': max(late_h, 0)
        })

    util = 0
    for m in machines:
        m_spans = [(s, e) for mm, s, e in spans if mm == m]
        if not m_spans:
            continue
        starts, ends = zip(*m_spans)
        span_h = (max(ends) - min(starts)).total_seconds() / 3600
        work_h = sum((e - s).total_seconds() / 3600 for s, e in m_spans)
        util += (work_h / span_h) if span_h > 0 else 0
    utilization_score = round(util * (10 / len(machines)), 2) if machines else 0

    return {
        'machine_schedule': machine_schedule,
        'orders_summary': orders_summary,
        'utilization_score': utilization_score,
        'optimality': {
            'status': solver.StatusName(status),
            'objective': solver.ObjectiveValue(),
            'lower_bound': solver.BestObjectiveBound(),
            'gap': round(100 * (solver.ObjectiveValue() - solver.BestObjectiveBound()) / solver.ObjectiveValue(), 2)
        }
    }

@app.route('/api/schedule', methods=['POST'])
def api_schedule():
    return jsonify(run_scheduler(request.get_json()))

@app.route('/')
def home():
    return render_template('frontend.html')

if __name__ == '__main__':
    app.run(debug=True)
