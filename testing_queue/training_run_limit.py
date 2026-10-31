"""Deployment deadline for the challenge's measured training command.

The official harness files stay unchanged. Its run_logged function is wrapped
when imported by a notebook/kernel or a Python harness subprocess.
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
import uuid
from pathlib import Path

LIMIT_SECONDS = 80.0
FAILURE_PATH = Path('/output/delivery/training_limit.json')


class TrainingRunLimitExceeded(RuntimeError):
    pass


def record_failure(log_path, elapsed, deadline_reached, target=FAILURE_PATH):
    record = {'reason': 'training_run_exceeded_limit', 'limit_s': LIMIT_SECONDS,
              'run_name': Path(log_path).parent.name, 'elapsed_s': elapsed,
              'reported_wall_s': round(elapsed, 2), 'deadline_reached': deadline_reached}
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'w') as handle:
        json.dump(record, handle, allow_nan=False)
        os.fchmod(handle.fileno(), 0o644)
    os.replace(temporary, target)
    return record


def wrap_run_logged(common, *, clock=time.perf_counter, failure_path=FAILURE_PATH,
                    memory_limit=None, memory_exceeded=None, on_progress=None):
    original = common.run_logged

    def limited(cmd, log_path, timeout, env=None):
        # Match the measured training child only. Warm-up and evaluation retain
        # their original deadlines, and each base/solution repeat gets a new clock.
        if not any(Path(str(arg)).name == '_train.py' for arg in cmd):
            return original(cmd, log_path, timeout, env)
        gpu_limit = memory_limit(cmd) if memory_limit else None
        started = clock()
        deadline_reached = False
        elapsed, code = None, None
        with common.GpuPoller() as poller, open(log_path, 'w', encoding='utf-8') as log:
            proc = common.subprocess.Popen(cmd, cwd=common.REPO, env=env or common.ENV,
                stdout=log, stderr=common.subprocess.STDOUT, start_new_session=True)
            try:
                # Same start/stop boundary as the harness's reported wall time,
                # including GPU-poller/process startup, not just the training loop.
                remaining = max(0.0, LIMIT_SECONDS - (clock() - started))
                try:
                    if gpu_limit is None and on_progress is None:
                        code = proc.wait(timeout=min(timeout, remaining))
                    else:
                        deadline = started + min(LIMIT_SECONDS, timeout + (clock() - started))
                        while True:
                            if on_progress:
                                on_progress(cmd, log_path, clock() - started, poller.peak, False)
                            if gpu_limit is not None and poller.peak > gpu_limit:
                                memory_exceeded(log_path, poller.peak, gpu_limit)
                            left = deadline - clock()
                            if left <= 0:
                                raise common.subprocess.TimeoutExpired(cmd, timeout)
                            try:
                                code = proc.wait(timeout=min(0.1, left))
                                break
                            except common.subprocess.TimeoutExpired:
                                continue
                except common.subprocess.TimeoutExpired:
                    code = -9
                    deadline_reached = timeout >= remaining
                elapsed = clock() - started
            finally:
                if elapsed is None:
                    elapsed = clock() - started
                try:
                    if on_progress:
                        on_progress(cmd, log_path, elapsed, poller.peak,
                                    code == 0 and not deadline_reached and elapsed <= LIMIT_SECONDS)
                finally:
                    common.kill_group(proc)
            if code == -9:
                log.write('\nTIMEOUT: training run deadline reached\n')
        if deadline_reached or elapsed > LIMIT_SECONDS:
            record_failure(log_path, elapsed, deadline_reached, failure_path)
            failure = TrainingRunLimitExceeded(
                f'{Path(log_path).parent.name} exceeded {LIMIT_SECONDS:g} seconds '
                f'(measured {elapsed:.3f} s); FAIL')
            failure.elapsed_s = elapsed
            raise failure
        return code, round(elapsed, 2), poller.peak, poller.start

    limited._soup_training_limit = LIMIT_SECONDS
    return limited


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, 'create_module', None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        required = ('run_logged', 'GpuPoller', 'kill_group', 'subprocess', 'REPO', 'ENV')
        if all(hasattr(module, name) for name in required):
            from notebook_failfast import ScreeningPolicy
            module.run_logged = ScreeningPolicy(module, wrap_run_logged).run_logged


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('common', 'harness.common'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.origin and Path(spec.origin).parts[-2:] == ('harness', 'common.py') and hasattr(spec.loader, 'exec_module'):
            spec.loader = _Loader(spec.loader)
            return spec
        return None


def install():
    if not any(isinstance(item, _Finder) for item in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
