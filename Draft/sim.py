import os
import json
from datetime import datetime, timedelta, time
from flask import Flask, request, jsonify, render_template
from collections import defaultdict
import holidays
import simpy

# --- CONFIGURATION ---
COUNTRY = 'IN'

# --- Flask App Setup ---
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'temp'))

# --- Utility Functions ---
def parse_datetime(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M')

def parse_time(s):
    return datetime.strptime(s, '%H:%M').time()

def format_dt(dt):
    return dt.strftime('%Y-%m-%d %H:%M')

def in_shift(dt, shift):
    st, et = parse_time(shift['start']), parse_time(shift['end'])
    if st < et:
        return st <= dt.time() < et
    else:
        return dt.time() >= st or dt.time() < et  # overnight shift

def get_shift_for_machine(machine, shifts):
    return [s for s in shifts if machine in s['machines']]

def get_status(dt, machine, shifts, weekend_off, holiday_cal, planned, unplanned):
    if dt.strftime('%Y-%m-%d') in planned.get(machine, []):
        return 'Planned Maintenance'
    for rng in unplanned.get(machine, []):
        s, e = rng.split(' to ')
        if parse_datetime(s) <= dt < parse_datetime(e):
            return 'Unplanned Maintenance'
    if dt.date() in holiday_cal:
        return 'Idle (Holiday)'
    if weekend_off and dt.weekday() >= 5:
        return 'Idle (Weekend)'

    machine_shifts = get_shift_for_machine(machine, shifts)
    if not machine_shifts:
        return 'Idle (Holiday)' if dt.date() in holiday_cal else (
               'Idle (Weekend)' if weekend_off and dt.weekday() >= 5 else 'Working')
    
    for sh in machine_shifts:
        if in_shift(dt, sh):
            return 'Working'
    return 'Idle (Shift Ended)'

def find_next_boundary(dt, machine, shifts, holiday_cal, planned, unplanned, weekend_off):
    candidates = [dt + timedelta(days=1)]
    for sh in get_shift_for_machine(machine, shifts):
        for bound in (sh['start'], sh['end']):
            bd = datetime.combine(dt.date(), parse_time(bound))
            if parse_time(sh['end']) <= parse_time(sh['start']) and bound == sh['end']:
                bd += timedelta(days=1)
            if bd <= dt:
                bd += timedelta(days=1)
            candidates.append(bd)
    for d in planned.get(machine, []):
        day = datetime.strptime(d, '%Y-%m-%d')
        candidates.append(datetime.combine(day, time(0)))
    for rng in unplanned.get(machine, []):
        s, e = rng.split(' to ')
        candidates.append(parse_datetime(s))
        candidates.append(parse_datetime(e))
    for h in holiday_cal:
        candidates.append(datetime.combine(h, time(0)))
    for i in range(1, 4):
        nxt = dt + timedelta(days=i)
        if weekend_off and nxt.weekday() >= 5:
            candidates.append(datetime.combine(nxt.date(), time(0)))
    return min([c for c in candidates if c > dt])

# --- SimPy Simulation Core ---
def simpy_simulate(schedule, shifts, weekend_off, planned, unplanned):
    result = defaultdict(list)
    holiday_cal = holidays.country_holidays(COUNTRY)
    product_task_ends = {}

    for machine, tasks in schedule.items():
        env = simpy.Environment()
        cursor = [parse_datetime(tasks[0]['start'])]

        def machine_process():
            yield env.timeout(0)
            last_input_end = None
            last_sim_end = None
            machine_shifts = get_shift_for_machine(machine, shifts)

            for task in tasks:
                task_start = parse_datetime(task['start'])
                task_end = parse_datetime(task['end'])
                duration = (task_end - task_start).total_seconds() / 60
                product = task['product']
                task_index = task['task_index']

                if task_index > 1:
                    prev_key = (product, task_index - 1)
                    if prev_key in product_task_ends:
                        ready_time = product_task_ends[prev_key]
                        if cursor[0] < ready_time:
                            cursor[0] = ready_time

                if last_input_end:
                    input_gap = (task_start - last_input_end).total_seconds() / 60
                    sim_gap = (cursor[0] - last_sim_end).total_seconds() / 60 if last_sim_end else 0
                    if sim_gap < input_gap:
                        cursor[0] = last_sim_end + timedelta(minutes=input_gap)

                if cursor[0] < task_start:
                    while cursor[0] < task_start:
                        status = get_status(cursor[0], machine, shifts, weekend_off, holiday_cal, planned, unplanned)
                        next_boundary = find_next_boundary(cursor[0], machine, shifts, holiday_cal, planned, unplanned, weekend_off)
                        end = min(task_start, next_boundary)
                        result[machine].append({
                            'start': format_dt(cursor[0]),
                            'end': format_dt(end),
                            'product': product,
                            'task_index': task_index,
                            'status': status
                        })
                        cursor[0] = end

                if machine_shifts:
                    while True:
                        status = get_status(cursor[0], machine, shifts, weekend_off, holiday_cal, planned, unplanned)

                        if status in ['Idle (Holiday)', 'Idle (Weekend)']:
                            next_day = datetime.combine((cursor[0] + timedelta(days=1)).date(), time(0, 0))
                            result[machine].append({
                                'start': format_dt(cursor[0]),
                                'end': format_dt(next_day),
                                'product': product,
                                'task_index': task_index,
                                'status': status
                            })
                            cursor[0] = next_day
                            continue

                        elif status != 'Working':
                            next_boundary = find_next_boundary(cursor[0], machine, shifts, holiday_cal, planned, unplanned, weekend_off)
                            result[machine].append({
                                'start': format_dt(cursor[0]),
                                'end': format_dt(next_boundary),
                                'product': product,
                                'task_index': task_index,
                                'status': status
                            })
                            cursor[0] = next_boundary
                            continue

                        shift_ends = []
                        for sh in machine_shifts:
                            if in_shift(cursor[0], sh):
                                se = datetime.combine(cursor[0].date(), parse_time(sh['end']))
                                if parse_time(sh['end']) <= parse_time(sh['start']):
                                    se += timedelta(days=1)
                                shift_ends.append(se)

                        if not shift_ends:
                            cursor[0] += timedelta(minutes=1)
                            continue

                        current_shift_end = min(shift_ends)
                        available_minutes = (current_shift_end - cursor[0]).total_seconds() / 60

                        if available_minutes < duration:
                            cursor[0] = current_shift_end
                            continue
                        break

                else:
                    # ⏰ NEW: Check spill-over into holiday/weekend before starting
                    while True:
                        status = get_status(cursor[0], machine, shifts, weekend_off, holiday_cal, planned, unplanned)

                        task_end_estimate = cursor[0] + timedelta(minutes=duration)
                        will_spill = False
                        current = cursor[0]
                        while current < task_end_estimate:
                            st = get_status(current, machine, shifts, weekend_off, holiday_cal, planned, unplanned)
                            if st in ['Idle (Holiday)', 'Idle (Weekend)', 'Planned Maintenance', 'Unplanned Maintenance']:
                                will_spill = True
                                break
                            current += timedelta(minutes=15)

                        if will_spill:
                            next_day = datetime.combine((cursor[0] + timedelta(days=1)).date(), time(0))
                            result[machine].append({
                                'start': format_dt(cursor[0]),
                                'end': format_dt(next_day),
                                'product': product,
                                'task_index': task_index,
                                'status': status if status.startswith("Idle") else 'Idle (Blocked)'
                            })
                            cursor[0] = next_day
                            continue

                        if status in ['Idle (Holiday)', 'Idle (Weekend)']:
                            next_day = datetime.combine((cursor[0] + timedelta(days=1)).date(), time(0))
                            result[machine].append({
                                'start': format_dt(cursor[0]),
                                'end': format_dt(next_day),
                                'product': product,
                                'task_index': task_index,
                                'status': status
                            })
                            cursor[0] = next_day
                            continue

                        elif status != 'Working':
                            next_boundary = find_next_boundary(cursor[0], machine, shifts, holiday_cal, planned, unplanned, weekend_off)
                            result[machine].append({
                                'start': format_dt(cursor[0]),
                                'end': format_dt(next_boundary),
                                'product': product,
                                'task_index': task_index,
                                'status': status
                            })
                            cursor[0] = next_boundary
                            continue
                        break

                end_time = cursor[0] + timedelta(minutes=duration)
                result[machine].append({
                    'start': format_dt(cursor[0]),
                    'end': format_dt(end_time),
                    'product': product,
                    'task_index': task_index,
                    'status': 'Working'
                })
                product_task_ends[(product, task_index)] = end_time
                last_input_end = task_end
                last_sim_end = end_time
                cursor[0] = end_time

        env.process(machine_process())
        env.run()

    return result

# --- Flask API Endpoints ---
@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    payload = request.get_json()
    schedule = payload.get('schedule', {})
    shifts = payload.get('shifts', [])
    weekend_off = payload.get('weekends_off', True)
    planned = payload.get('planned', {})
    unplanned = payload.get('unplanned', {})

    all_machines = list(schedule.keys())
    if not shifts:
        shifts = [{'name': 'Default', 'start': '00:00', 'end': '23:59', 'machines': all_machines}]

    result = simpy_simulate(schedule, shifts, weekend_off, planned, unplanned)
    return jsonify(result)

@app.route('/')
def home():
    return render_template('front.html')

if __name__ == '__main__':
    app.run(debug=True)
