"""Lossless Sheet projections of existing notebook API results; no recalculation.

project_tables(jobs, receipts=()) returns {tab_name: [headers, *rows]}.
Write all strings literally (Sheets RAW or userEnteredValue.stringValue), never
as formulas: notebook failure messages and other strings are untrusted data.
Numbers remain numbers, measured zero remains zero, and missing values are ''.
Complete job/receipt JSON is split into JSON-encoded string chunks in VM Details.
Reconstruct with decode_details_rows() to recover every original result field.
"""
from datetime import datetime, timezone
import hashlib
import json
import math


RESULT_HEADERS = [
    'Request ID', 'Candidate', 'Job ID', 'Status', 'Outcome', 'Training (s)',
    'Speedup (×)', 'Speedup kind', 'GPU peak (MiB)', 'Eval loss', 'Eval status',
    'Loss check', 'Time check', 'VRAM check', 'Environment', 'Train loop (s)',
    'Train loss', 'Train loss status', 'Whole job (s)', 'Memory saved (%)',
    'Eval loss delta (%)', 'Eval loss limit', 'Display side', 'GPU peak scope',
    'Partial training', 'Baseline source', 'Baseline job ID', 'Baseline training (s)',
    'Baseline GPU peak (MiB)', 'Baseline eval loss', 'Host GPU peak (MiB)',
    'Comparable', 'Paired', 'Flags', 'Error code', 'Error phase', 'Error run',
    'Error message', 'Error measured', 'Error limit', 'Error unit', 'Reason',
    'Created UTC', 'Started UTC', 'Finished UTC', 'Re-run of', 'Source SHA256',
    'Outputs SHA256', 'API request ID', 'Raw details SHA256',
]
RUN_HEADERS = ['Request ID', 'Candidate', 'Job ID', 'Job status', 'Run', 'Side',
    'Training (s)', 'GPU peak (MiB)', 'Eval loss', 'Partial', 'Error code',
    'Error message', 'Error measured', 'Error limit', 'Error unit']
DETAIL_HEADERS = ['Record key', 'Record type', 'Request ID', 'Job ID', 'Part',
    'Parts', 'JSON SHA256', 'JSON-encoded text chunk']
QUEUE_HEADERS = ['Request ID', 'Candidate', 'State', 'Job ID', 'Sheet row',
    'Error', 'Source SHA256', 'API request ID', 'Record key', 'Raw details SHA256',
    'Receipt type', 'Receipt name', 'Rows', 'Ready rows', 'Last poll UTC']
NUMBER_FORMATS = {'Training (s)': '0.00', 'Speedup (×)': '0.00',
    'GPU peak (MiB)': '0', 'Eval loss': '0.00000', 'Train loop (s)': '0.00',
    'Train loss': '0.00000', 'Whole job (s)': '0.00', 'Memory saved (%)': '0.00',
    'Eval loss delta (%)': '0.00', 'Eval loss limit': '0.00000',
    'Baseline training (s)': '0.00', 'Baseline GPU peak (MiB)': '0',
    'Baseline eval loss': '0.00000', 'Host GPU peak (MiB)': '0'}
OUTCOMES = {'beats_baseline': 'Beat baseline', 'faster_more_memory': 'Faster · more VRAM',
    'less_memory': 'Less VRAM · no speed gain', 'no_gain': 'No measured gain',
    'checks_failed': 'Checks failed', 'incomplete': 'Incomplete comparison',
    'no_metrics': 'No training report', 'run_failed': 'Run failed', 'pending': 'In progress'}
JSON_CHUNK_SIZE = 12000  # Escaping/surrogates still stay below Sheets' 50k cell limit.


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else ''


def boolean(value):
    return value if type(value) is bool else ''


def text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def utc(value):
    if number(value) == '':
        return ''
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z')
    except (OverflowError, OSError, ValueError):
        return ''


def request_id(value):
    value = text(value)
    return value[6:] if value.startswith('sheet-') else value


def checks_for(job):
    """Mirror the three public-check pills in web/notebooks.js exactly."""
    checks = mapping(mapping(job.get('metrics')).get('checks'))
    result = {key: checks.get(key) or 'unknown' for key in ('loss', 'time', 'memory')}
    failure = mapping(mapping(job.get('result')).get('error'))
    if failure.get('code') in ('EVAL_LOSS_EXCEEDED', 'INVALID_EVAL_RESULT'):
        result['loss'] = 'failed'
    if failure.get('code') == 'TIME_LIMIT_EXCEEDED' and failure.get('phase') == 'training':
        result['time'] = 'failed'
    if failure.get('code') == 'MEMORY_LIMIT_EXCEEDED' and failure.get('unit') == 'MiB':
        result['memory'] = 'failed'
    result['environment'] = checks.get('environment') or 'unknown'
    return result


def raw_details(value, key, kind, rid='', job_id=''):
    # The API already emits standard finite JSON; fail explicitly rather than
    # silently dropping unsupported data or truncating a large metrics report.
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    chunks = [raw[i:i + JSON_CHUNK_SIZE] for i in range(0, len(raw), JSON_CHUNK_SIZE)]
    rows = [[key, kind, rid, job_id, i + 1, len(chunks), sha,
             json.dumps(chunk, ensure_ascii=False)] for i, chunk in enumerate(chunks)]
    return sha, rows


def decode_details_rows(rows):
    """Decode data rows (without header) for one Record key, verifying the hash."""
    if not rows:
        raise ValueError('No JSON detail parts')
    ordered = sorted(rows, key=lambda row: row[4])
    first, total = ordered[0], ordered[0][5]
    if len(ordered) != total or [row[4] for row in ordered] != list(range(1, total + 1)):
        raise ValueError('Missing or duplicate JSON detail parts')
    if any(row[:4] != first[:4] or row[5:7] != first[5:7] for row in ordered):
        raise ValueError('Mixed JSON detail records')
    chunks = [json.loads(row[7]) for row in ordered]
    if any(not isinstance(chunk, str) for chunk in chunks):
        raise ValueError('Invalid JSON detail chunk')
    raw = ''.join(chunks)
    if hashlib.sha256(raw.encode()).hexdigest() != first[6]:
        raise ValueError('JSON detail integrity check failed')
    return json.loads(raw)


def project_job(job, raw_sha=''):
    """Return one Results row. Existing comparison/formulas are not changed."""
    m = mapping(job.get('metrics'))
    base, solution = mapping(m.get('base')), mapping(m.get('solution'))
    display = mapping(m.get('display')) if isinstance(m.get('display'), dict) else solution
    result, checks = mapping(job.get('result')), checks_for(job)
    failure = mapping(result.get('error'))
    train_loss = number(display.get('train_loss'))
    if train_loss == '':
        train_loss = number(display.get('training_loss'))
    flags = []
    for condition, flag in (
        (display.get('partial') is True or solution.get('partial') is True, 'partial_training'),
        (m.get('speedup_kind') == 'upper_bound', 'speedup_upper_bound'),
        (display.get('gpu_scope') == 'whole_notebook', 'whole_notebook_gpu'),
        (display.get('side') == 'base', 'baseline_display'),
        (m.get('baseline_source') == 'saved', 'saved_baseline'),
        (checks['environment'] == 'changed', 'environment_changed'),
        (number(display.get('eval_loss')) == '', 'not_evaluated'),
        (m.get('available') is False, 'no_metrics'),
        (job.get('state') in ('failed', 'timed_out', 'interrupted', 'cancelled'), 'run_failed'),
    ):
        if condition:
            flags.append(flag)
    values = {
        'Request ID': request_id(job.get('request_id')), 'Candidate': text(job.get('candidate')),
        'Job ID': text(job.get('id')), 'Status': text(job.get('state')),
        'Outcome': OUTCOMES.get(m.get('comparison') or 'no_metrics', 'No comparison'),
        'Training (s)': number(display.get('wall_s')), 'Speedup (×)': number(m.get('speedup')),
        'Speedup kind': text(m.get('speedup_kind')), 'GPU peak (MiB)': number(display.get('gpu_peak_mib')),
        'Eval loss': number(display.get('eval_loss')),
        'Eval status': 'measured' if number(display.get('eval_loss')) != '' else 'not_evaluated',
        'Loss check': text(checks['loss']), 'Time check': text(checks['time']),
        'VRAM check': text(checks['memory']), 'Environment': text(checks['environment']),
        'Train loop (s)': number(display.get('train_loop_s')), 'Train loss': train_loss,
        'Train loss status': 'measured' if train_loss != '' else 'not_reported',
        'Whole job (s)': number(result.get('elapsed_s')), 'Memory saved (%)': number(m.get('memory_saved_pct')),
        'Eval loss delta (%)': number(m.get('loss_delta_pct')), 'Eval loss limit': number(m.get('loss_limit')),
        'Display side': text(display.get('side') if m.get('display') is not None else 'solution' if solution else None),
        'GPU peak scope': text(display.get('gpu_scope')), 'Partial training': boolean(display.get('partial')),
        'Baseline source': text(m.get('baseline_source')), 'Baseline job ID': text(m.get('baseline_job_id')),
        'Baseline training (s)': number(base.get('wall_s')), 'Baseline GPU peak (MiB)': number(base.get('gpu_peak_mib')),
        'Baseline eval loss': number(base.get('eval_loss')), 'Host GPU peak (MiB)': number(m.get('host_gpu_peak_mib')),
        'Comparable': boolean(m.get('comparable')), 'Paired': boolean(m.get('paired')), 'Flags': ', '.join(flags),
        'Error code': text(failure.get('code')), 'Error phase': text(failure.get('phase')),
        'Error run': text(failure.get('run')), 'Error message': text(failure.get('message')),
        'Error measured': number(failure.get('measured')), 'Error limit': number(failure.get('limit')),
        'Error unit': text(failure.get('unit')), 'Reason': text(result.get('reason')),
        'Created UTC': utc(job.get('created')), 'Started UTC': utc(job.get('started')),
        'Finished UTC': utc(job.get('finished')), 'Re-run of': text(job.get('rerun_of')),
        'Source SHA256': text(job.get('source_sha')), 'Outputs SHA256': text(job.get('output_sha')),
        'API request ID': text(job.get('request_id')), 'Raw details SHA256': raw_sha,
    }
    return [values[header] for header in RESULT_HEADERS]


def project_runs(job):
    metrics, result = mapping(job.get('metrics')), mapping(job.get('result'))
    failure = mapping(result.get('error'))
    runs = metrics.get('training_runs')
    rows = []
    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, dict):
            continue
        error = failure if failure.get('run') and failure.get('run') == run.get('run') else {}
        rows.append([request_id(job.get('request_id')), text(job.get('candidate')), text(job.get('id')),
            text(job.get('state')), text(run.get('run')), text(run.get('side')), number(run.get('wall_s')),
            number(run.get('gpu_peak_mib')), number(run.get('eval_loss')), boolean(run.get('partial')),
            text(error.get('code')), text(error.get('message')), number(error.get('measured')),
            number(error.get('limit')), text(error.get('unit'))])
    return rows


def project_tables(jobs, receipts=()):
    """Build four complete table snapshots; caller owns authenticated publication."""
    tables = {'VM Results': [RESULT_HEADERS.copy()], 'VM Runs': [RUN_HEADERS.copy()],
              'VM Details': [DETAIL_HEADERS.copy()], 'VM Queue': [QUEUE_HEADERS.copy()]}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError('Each queue job must be an object')
        rid, jid = request_id(job.get('request_id')), text(job.get('id'))
        sha, details = raw_details(job, 'job:' + (jid or str(index)), 'job', rid, jid)
        tables['VM Results'].append(project_job(job, sha))
        tables['VM Runs'].extend(project_runs(job))
        tables['VM Details'].extend(details)
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise ValueError('Each queue receipt must be an object')
        job = mapping(receipt.get('job'))
        rid, jid = text(receipt.get('request_id')), text(receipt.get('job_id'))
        name = text(receipt.get('receipt_name'))
        kind = 'sheet_status' if name == 'sheet-status.json' else 'conflict' if name.endswith('.conflict.json') else \
            'rejection' if name.endswith('.rejected.json') or name.startswith('invalid-row-') else 'request'
        key = 'receipt:' + (name or (rid or 'row') + ':' + str(index))
        sha, details = raw_details(receipt, key, 'receipt', rid, jid)
        tables['VM Queue'].append([rid, text(receipt.get('candidate_id') or job.get('candidate')),
            text(receipt.get('state')), jid, number(receipt.get('row')), text(receipt.get('error')),
            text(receipt.get('source_sha256') or receipt.get('original_source_sha256')),
            text(receipt.get('api_request_id')), key, sha, kind, name,
            number(receipt.get('rows')), number(receipt.get('ready')), utc(receipt.get('checked_at'))])
        tables['VM Details'].extend(details)
    return tables


def project_receipts(receipts):
    """Project worker outbox records, including pre-submission and Sheet errors.

The worker supplies receipt_name and may supply candidate_id from its ledger.
No receipt is discarded. Repeated job IDs have one Results row; the last full
job snapshot wins, while every original receipt remains losslessly in Details.
"""
    receipts = list(receipts)
    jobs = {}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError('Each queue receipt must be an object')
        job = receipt.get('job')
        if isinstance(job, dict) and isinstance(job.get('id'), str):
            jobs[job['id']] = job
    ordered = sorted(jobs.values(), key=lambda item: number(item.get('created')) if number(item.get('created')) != '' else 0, reverse=True)
    return project_tables(ordered, receipts)
