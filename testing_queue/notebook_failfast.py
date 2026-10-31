"""Public-data screening and typed failures for the notebook execution adapter."""
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from notebook_reference import load_reference, matching_reference, harness_matches

FAILURE_PATH = Path('/output/delivery/failure.json')
CODES = {
    'TIME_LIMIT_EXCEEDED', 'MEMORY_LIMIT_EXCEEDED', 'EVAL_LOSS_EXCEEDED',
    'INVALID_EVAL_RESULT', 'BASELINE_REQUIRED', 'EXECUTION_ERROR',
    'STORAGE_LIMIT_EXCEEDED',
}


def error(code, message, phase, **details):
    return {'code': code, 'message': message[:500], 'phase': phase, **details}


def valid_error(value):
    if not isinstance(value, dict) or set(value) - {
        'code', 'message', 'phase', 'run', 'measured', 'limit', 'unit', 'baseline',
    }:
        return False
    if value.get('code') not in CODES:
        return False
    for key, bound in (('message', 500), ('phase', 80)):
        if not isinstance(value.get(key), str) or not 0 < len(value[key]) <= bound:
            return False
    for key in ('run', 'unit'):
        if key in value and (not isinstance(value[key], str) or len(value[key]) > 100):
            return False
    for key in ('measured', 'limit', 'baseline'):
        if key in value and (type(value[key]) not in (int, float) or
                             not math.isfinite(value[key]) or value[key] < 0):
            return False
    return True


def write_json(path, value, first=False):
    path = Path(path)
    raw = json.dumps(value, allow_nan=False).encode()
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)
        os.fchmod(handle.fileno(), 0o644)
    if first:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    else:
        os.replace(temporary, path)


def classify_exception(text, phase='notebook'):
    # An arbitrary SIGKILL/exit 137 is not evidence of OOM.
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    if re.search(r'No space left on device|Disk quota exceeded|\bENOSPC\b|\bEDQUOT\b', text, re.I):
        return error('STORAGE_LIMIT_EXCEEDED', 'Writable job storage is full or its quota was exceeded', phase)
    if re.search(r'CUDA (?:error: )?out of memory|torch\.OutOfMemoryError|CUDAOutOfMemoryError', text, re.I):
        return error('MEMORY_LIMIT_EXCEEDED', 'GPU memory exhausted (CUDA out of memory)', phase, unit='MiB')
    if re.search(r'\bMemoryError\b|DefaultCPUAllocator.*(?:allocate|memory)|std::bad_alloc|Cannot allocate memory', text, re.I):
        return error('MEMORY_LIMIT_EXCEEDED', 'Host RAM allocation failed', phase, unit='GiB')
    return error('EXECUTION_ERROR', text.strip()[-500:] or 'Execution failed; see logs', phase)


def container_failure(state, reported):
    # A cgroup OOM event may kill a kernel/child while the notebook runner is alive.
    # Prefer daemon OOM evidence over a consequent generic "kernel died" exception.
    if state.get('OOMKilled'):
        return error('MEMORY_LIMIT_EXCEEDED', 'Container killed for memory pressure',
                     'container', limit=30, unit='GiB')
    return reported


class ScreeningFailed(RuntimeError):
    pass


def compare_loss(measured, baseline):
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in (measured, baseline)):
        return error('INVALID_EVAL_RESULT', 'Evaluation loss must be finite and nonnegative', 'evaluation')
    limit = baseline * 1.005
    if measured > limit:
        return error('EVAL_LOSS_EXCEEDED',
                     f'Evaluation loss {measured:.8g} exceeds {limit:.8g} (baseline +0.5%)',
                     'evaluation', measured=measured, limit=limit, baseline=baseline, unit='loss')
    return None


def command_arg(cmd, flag):
    try:
        return str(cmd[cmd.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def training_identity(cmd):
    """Compare actual task settings/data, ignoring the per-run output directory."""
    import yaml
    config_path = command_arg(cmd, '--config')
    if not config_path:
        return None, None
    config = yaml.safe_load(Path(config_path).read_text())
    adapter = str(Path(config.pop('output')).resolve())
    train = Path(config['data']['train'])
    config['data']['train'] = hashlib.sha256(train.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return fingerprint, adapter


class ScreeningPolicy:
    def __init__(self, common, deadline_wrapper, failure_path=FAILURE_PATH):
        self.common = common
        self.failure_path = Path(failure_path)
        self.records = {}
        self.baselines = {}
        self.evals = {}
        self.telemetry = {}
        self.reference = load_reference()
        if not harness_matches(self.reference, getattr(common, 'REPO', '/nonexistent')):
            self.reference = {}
        self.original = common.run_logged
        self.training = deadline_wrapper(common, memory_limit=self.memory_limit,
                                         memory_exceeded=self.memory_exceeded,
                                         on_progress=self.progress)

    def progress(self, cmd, log_path, wall, peak, complete):
        name = Path(log_path).parent.name
        previous = self.telemetry.get(name, {})
        self.telemetry[name] = {**previous, 'run': name,
            'side': 'solution' if '--solution' in cmd else 'base',
            'wall_s': round(wall, 3), 'gpu_peak_mib': peak,
            'training_complete': complete, 'partial': not complete}
        self.save_metrics()

    def save_metrics(self):
        write_json(self.failure_path.parent / 'run-metrics.json', {
            'runs': list(self.telemetry.values())[-64:],
            'reference_job': self.reference.get('job_id')})

    def fail(self, detail):
        write_json(self.failure_path, detail, first=True)
        raise ScreeningFailed(detail['code'] + ': ' + detail['message'])

    def memory_limit(self, cmd):
        if '--solution' not in cmd:
            return None
        fingerprint, _ = training_identity(cmd)
        baseline = self.baselines.get(fingerprint)
        if baseline is None and matching_reference(self.reference, fingerprint):
            baseline = {'peak': self.reference['metrics']['gpu_peak_mib']}
        return baseline['peak'] * 1.1 if baseline and baseline['peak'] > 0 else None

    def memory_exceeded(self, log_path, peak, limit):
        self.fail(error('MEMORY_LIMIT_EXCEEDED',
                        f'{Path(log_path).parent.name}: GPU peak {peak} MiB exceeds {limit:g} MiB (baseline +10%)',
                        'training', run=Path(log_path).parent.name, measured=peak,
                        limit=limit, baseline=limit / 1.1, unit='MiB'))

    def preserve(self, log_path, cmd):
        # Small partial evidence is saved even when later notebook export cells never run.
        destination = self.failure_path.parent / 'harness'
        destination.mkdir(exist_ok=True)
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(log_path).parent.name + '-' + Path(log_path).name)
        try:
            with open(log_path, 'rb') as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - 2 * 1024**2))
                data = handle.read(2 * 1024**2)
            target = destination / name
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
            out = command_arg(cmd, '--out')
            if out and Path(out).is_file() and Path(out).stat().st_size < 65536:
                write_json(destination / (name + '.json'), json.loads(Path(out).read_text()))
        except (OSError, ValueError):
            pass

    def run_logged(self, cmd, log_path, timeout, env=None):
        from training_run_limit import TrainingRunLimitExceeded
        is_training = any(Path(str(arg)).name == '_train.py' for arg in cmd)
        is_eval = any(Path(str(arg)).name == 'eval_adapter.py' for arg in cmd)
        phase = 'training' if is_training else 'evaluation' if is_eval else 'warmup'
        if is_training:
            fingerprint, adapter = training_identity(cmd)
            self.telemetry[Path(log_path).parent.name] = {
                'task_fingerprint': fingerprint, 'adapter': adapter,
                'reference_compatible': matching_reference(self.reference, fingerprint)}
        try:
            try:
                result = self.training(cmd, log_path, timeout, env) if is_training else self.original(cmd, log_path, timeout, env)
            except TrainingRunLimitExceeded as exc:
                self.fail(error('TIME_LIMIT_EXCEEDED', str(exc), 'training',
                                run=Path(log_path).parent.name, measured=exc.elapsed_s,
                                limit=80, unit='seconds'))
            code, wall, peak, _ = result
            if code:
                if code == -9:
                    detail = error('TIME_LIMIT_EXCEEDED', f'{phase} subprocess deadline exceeded',
                                   phase, limit=timeout, measured=wall, unit='seconds')
                else:
                    with open(log_path, 'rb') as handle:
                        handle.seek(0, 2)
                        handle.seek(max(0, handle.tell() - 32768))
                        text = handle.read().decode(errors='replace')
                    detail = classify_exception(text, phase)
                self.fail(detail)
            if is_training:
                fingerprint, adapter = training_identity(cmd)
                if fingerprint:
                    record = {'fingerprint': fingerprint, 'peak': peak,
                              'side': 'solution' if '--solution' in cmd else 'base',
                              'run': Path(log_path).parent.name}
                    self.records[adapter] = record
                    if record['side'] == 'base':
                        # Freeze the first matching baseline for predictable early screening.
                        self.baselines.setdefault(fingerprint, record)
                    else:
                        limit = self.memory_limit(cmd)
                        if limit is not None and peak > limit:
                            self.memory_exceeded(log_path, peak, limit)
            if is_eval:
                out = command_arg(cmd, '--out')
                try:
                    loss = json.loads(Path(out).read_text())['mean_loss']
                except (OSError, ValueError, KeyError, TypeError):
                    self.fail(error('INVALID_EVAL_RESULT', 'Evaluation did not produce a valid mean_loss', phase))
                invalid = compare_loss(loss, loss)
                if invalid:
                    self.fail(invalid)
                adapter = command_arg(cmd, '--adapter')
                record = self.records.get(str(Path(adapter).resolve())) if adapter else None
                if not record:
                    self.fail(error('BASELINE_REQUIRED', 'No matching training evidence for this adapter', phase))
                eval_file = command_arg(cmd, '--eval-file')
                data_hash = hashlib.sha256(Path(eval_file).read_bytes()).hexdigest()
                key = (record['fingerprint'], data_hash)
                item = self.telemetry.setdefault(record.get('run', record['side'] + '_1'), {})
                item.update(side=record['side'], eval_loss=loss, eval_data_sha256=data_hash,
                            task_fingerprint=record['fingerprint'])
                self.save_metrics()
                if record['side'] == 'base':
                    self.evals.setdefault(key, loss)
                else:
                    baseline = self.evals.get(key)
                    if baseline is None and matching_reference(self.reference, record['fingerprint'], eval_hash=data_hash):
                        baseline = self.reference['metrics']['eval_loss']
                    if baseline is None:
                        self.fail(error('BASELINE_REQUIRED', 'Task or evaluation data differs from the saved baseline; run a matching base for this task', phase))
                    detail = compare_loss(loss, baseline)
                    write_json(self.failure_path.parent / 'screening.json', {
                        'scope': 'public_data_screen_only', 'baseline_loss': baseline,
                        'solution_loss': loss, 'eval_loss_limit': baseline * 1.005,
                        'eval_data_sha256': data_hash, 'task_fingerprint': record['fingerprint'],
                        'eval_loss_check': 'failed' if detail else 'passed',
                    })
                    if detail:
                        self.fail(detail)
            return result
        finally:
            self.preserve(log_path, cmd)

