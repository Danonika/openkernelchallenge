"""Bounded public metrics, including partial runs and a compatible saved baseline."""
import json
import math
import re
import statistics
import zipfile
from notebook_reference import matching_reference
from training_run_limit import LIMIT_SECONDS, GPU_LIMIT_MIB

VERSION = 3


def number(value, positive=False):
    return value if type(value) in (int, float) and math.isfinite(value) and (
        value > 0 if positive else value >= 0) else None


def read_report(archive, name):
    try:
        entry = archive.getinfo(name)
        if entry.file_size > 2 * 1024**2:
            return {}
        value = json.loads(archive.read(entry))
        return value if isinstance(value, dict) else {}
    except (KeyError, ValueError, OSError, RuntimeError, RecursionError, zipfile.BadZipFile):
        return {}


def summarize_archive(path, reference=None, output_sha=None):
    empty = {'version': VERSION, 'scope': 'reported_public_metrics', 'available': False}
    reference = reference or {}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            selected = 'results/result.json'
            if selected not in names:
                candidates = [n for n in names if n.startswith('results/') and
                              n.endswith('/result.json') and not n.startswith('results/harness/')]
                selected = candidates[0] if len(candidates) == 1 else ''
            report = read_report(archive, selected)
            screening = read_report(archive, 'results/screening.json')
            telemetry = read_report(archive, 'results/run-metrics.json')
            host = read_report(archive, 'results/host-metrics.json')
            runner = read_report(archive, 'runner.json')
            failure = read_report(archive, 'results/failure.json') or runner.get('error') or {}
            deadline = read_report(archive, 'results/training-limit.json')
            evals = {}
            for side in ('base', 'solution'):
                candidates = [n for n in names if n.endswith('/eval_' + side + '.json') or
                              n.startswith('results/harness/') and n.endswith('-eval_' + side + '.log.json')]
                for name in candidates[:8]:
                    loss = number(read_report(archive, name).get('mean_loss'))
                    if loss is not None:
                        evals[side] = loss
            log_runs = []
            try:
                entry = archive.getinfo('execution.log')
                if entry.file_size <= 2 * 1024**2:
                    text = archive.read(entry).decode(errors='replace')
                    for name, status, wall, peak in re.findall(
                            r'\[run_task\]\s+((?:base|solution)_\d+):\s*(\w+),\s*([\d.]+)\s*s,\s*peak\s*(\d+)\s*MiB', text):
                        log_runs.append({'run': name, 'wall_s': float(wall), 'gpu_peak_mib': int(peak),
                                         'training_complete': status == 'ok', 'partial': status != 'ok'})
            except KeyError:
                pass
    except (OSError, zipfile.BadZipFile, ValueError, TypeError):
        return empty

    failure = failure if isinstance(failure, dict) else {}
    summary = report.get('summary')
    summary = summary if isinstance(summary, dict) else {}
    records = {}
    # Weak fallback first. Checkpoints replace host observations for the same run.
    for group in (host.get('runs'), log_runs, telemetry.get('runs')):
        for item in group[:64] if isinstance(group, list) else []:
            if isinstance(item, dict) and isinstance(item.get('run'), str):
                records[item['run']] = {**records.get(item['run'], {}), **item}
    if deadline:
        name = deadline.get('run_name', 'training')
        if isinstance(name, str):
            records[name] = {**records.get(name, {}), 'run': name,
                             'wall_s': deadline.get('elapsed_s'), 'partial': True,
                             'training_complete': False}
    if failure.get('phase') == 'training' and isinstance(failure.get('run'), str) and failure['run']:
        name = failure['run']
        item = records.setdefault(name, {'run': name, 'partial': True})
        if failure.get('unit') == 'seconds' and number(failure.get('measured')) is not None:
            item['wall_s'] = failure['measured']
        if failure.get('unit') == 'MiB' and number(failure.get('measured')) is not None:
            item['gpu_peak_mib'] = failure['measured']

    sides = {}
    for side in ('base', 'solution'):
        raw = summary.get(side)
        raw = raw if isinstance(raw, dict) else {}
        items = [r for name, r in records.items() if r.get('side', name.split('_')[0]) == side]
        partial = any(r.get('partial') is True for r in items)
        sides[side] = {
            'ok': raw.get('ok') is True or bool(items) and all(r.get('training_complete') is True for r in items),
            'wall_s': number(raw.get('wall_s'), positive=True),
            'gpu_peak_mib': number(raw.get('gpu_peak_mib')),
            'eval_loss': number(raw.get('eval_loss')),
            'train_loop_s': number(raw.get('train_loop_s')),
            'runs': number(raw.get('runs'), positive=True) or len(items) or None,
            'partial': partial,
        }
        result = sides[side]
        for key in ('wall_s', 'gpu_peak_mib', 'eval_loss', 'train_loop_s'):
            values = [number(r.get(key)) for r in items]
            values = [v for v in values if v is not None]
            if values and result[key] is None:
                result[key] = max(values) if key == 'gpu_peak_mib' else values[-1] if partial else statistics.median(values)
        if result['eval_loss'] is None:
            result['eval_loss'] = number(screening.get('baseline_loss' if side == 'base' else 'solution_loss'))
        if result['eval_loss'] is None:
            result['eval_loss'] = evals.get(side)
        result['eval_status'] = 'measured' if result['eval_loss'] is not None else 'not_evaluated'
    base, sol = sides['base'], sides['solution']
    if failure.get('code') == 'EVAL_LOSS_EXCEEDED':
        sol['eval_loss'] = number(failure.get('measured'))
        sol['eval_status'] = 'measured' if sol['eval_loss'] is not None else 'not_evaluated'
        if base['eval_loss'] is None:
            base['eval_loss'] = number(failure.get('baseline'))

    baseline_source = 'paired' if base['wall_s'] is not None else None
    historical = output_sha is not None and output_sha in reference.get('compatible_outputs', [])
    solution_records = [r for r in records.values() if r.get('side') == 'solution']
    compatible = historical or bool(solution_records) and all(
        r.get('reference_compatible') is True and matching_reference(reference,
            r.get('task_fingerprint'), image_id=runner.get('image_id', ''),
            eval_hash=r.get('eval_data_sha256')) for r in solution_records)
    if baseline_source is None and compatible and runner.get('image_id') == reference.get('image_id'):
        base.update(reference['metrics'], partial=False, eval_status='measured')
        baseline_source = 'saved'

    speedup = base['wall_s'] / sol['wall_s'] if base['wall_s'] and sol['wall_s'] else None
    memory_saved = 100 * (1 - sol['gpu_peak_mib'] / base['gpu_peak_mib']) if base['gpu_peak_mib'] and sol['gpu_peak_mib'] is not None else None
    loss_check, loss_limit, loss_delta = 'unknown', None, None
    if base['eval_loss'] is not None and sol['eval_loss'] is not None:
        loss_limit = base['eval_loss'] * 1.005
        loss_check = 'passed' if sol['eval_loss'] <= loss_limit else 'failed'
        if base['eval_loss'] > 0:
            loss_delta = 100 * (sol['eval_loss'] / base['eval_loss'] - 1)
        elif sol['eval_loss'] == 0:
            loss_delta = 0
    comparable = base['ok'] and sol['ok'] and not sol['partial'] and speedup is not None and memory_saved is not None
    time_check = 'unknown'
    measured_times = [s['wall_s'] for s in sides.values() if s['wall_s'] is not None]
    if any(t > LIMIT_SECONDS for t in measured_times):
        time_check = 'failed'
    elif sol['wall_s'] is not None and not sol['partial']:
        time_check = 'passed'
    memory_check = 'unknown'
    if any(s['gpu_peak_mib'] is not None and s['gpu_peak_mib'] > GPU_LIMIT_MIB for s in sides.values()) or (number(host.get('gpu_peak_mib')) or 0) > GPU_LIMIT_MIB:
        memory_check = 'failed'
    elif sol['gpu_peak_mib'] is not None and not sol['partial']:
        memory_check = 'passed'
    changed = report.get('packages_changed')
    environment = 'unchanged' if changed == [] else 'changed' if isinstance(changed, list) and changed else 'unknown'
    checks = {'loss': loss_check, 'time': time_check, 'memory': memory_check, 'environment': environment}
    if failure.get('code') == 'TIME_LIMIT_EXCEEDED' and failure.get('phase') == 'training':
        checks['time'] = 'failed'
    if failure.get('code') == 'MEMORY_LIMIT_EXCEEDED':
        checks['memory'] = 'failed'
    available = any(s[k] is not None for s in sides.values() for k in ('wall_s','gpu_peak_mib','eval_loss'))
    for name, value in (('speedup', speedup), ('memory_saved', memory_saved), ('loss_delta', loss_delta), ('loss_limit', loss_limit)):
        if value is not None and not math.isfinite(value):
            if name == 'speedup': speedup = None
            elif name == 'memory_saved': memory_saved = None
            elif name == 'loss_delta': loss_delta = None
            else: loss_limit, checks['loss'] = None, 'unknown'
            comparable = False
    display_side = 'base' if not any(sol[k] is not None for k in ('wall_s','gpu_peak_mib','eval_loss')) and any(base[k] is not None for k in ('wall_s','gpu_peak_mib','eval_loss')) else 'solution'
    display = dict(sides[display_side])
    display['side'] = display_side
    display['gpu_scope'] = 'training'
    if display['gpu_peak_mib'] is None:
        display['gpu_peak_mib'] = number(host.get('gpu_peak_mib'))
        display['gpu_scope'] = 'whole_notebook'
    available = available or display['gpu_peak_mib'] is not None
    return {**empty, 'available': available, 'source': selected if report else 'partial_evidence',
            **sides, 'display': display, 'comparable': comparable,
            'paired': comparable and baseline_source == 'paired', 'baseline_source': baseline_source,
            'baseline_job_id': reference.get('job_id') if baseline_source == 'saved' else None,
            'speedup': speedup, 'speedup_kind': 'upper_bound' if sol['partial'] else 'measured',
            'memory_saved_pct': memory_saved, 'loss_delta_pct': loss_delta,
            'loss_limit': loss_limit, 'checks': checks,
            'host_gpu_peak_mib': number(host.get('gpu_peak_mib')),
            'training_runs': [{'run': name[:100], 'side': 'solution' if r.get('side', name.split('_')[0]) == 'solution' else 'base',
                'wall_s': number(r.get('wall_s')), 'gpu_peak_mib': number(r.get('gpu_peak_mib')),
                'eval_loss': number(r.get('eval_loss')), 'partial': r.get('partial') is True}
                for name, r in records.items()]}


def comparison(metrics, state):
    if state in ('failed', 'timed_out', 'interrupted', 'cancelled'):
        return 'run_failed'
    if state != 'completed':
        return 'pending'
    if not metrics.get('available'):
        return 'no_metrics'
    checks = metrics.get('checks', {})
    if 'failed' in checks.values() or checks.get('environment') == 'changed':
        return 'checks_failed'
    if not metrics.get('comparable') or any(checks.get(k) != 'passed' for k in ('loss', 'time', 'memory')) or checks.get('environment') != 'unchanged':
        return 'incomplete'
    faster = metrics['speedup'] > 1
    less_memory = metrics['memory_saved_pct'] > 0
    if faster and metrics['memory_saved_pct'] >= 0:
        return 'beats_baseline'
    if faster:
        return 'faster_more_memory'
    if less_memory:
        return 'less_memory'
    return 'no_gain'

