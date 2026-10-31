"""Prepared-only notebook support: reviewed local assets, unchanged official harness.

This file is not run by preparation and is not part of the queue solution ZIP.
GPU execution requires explicit future authorization and a prepared exact stack.
"""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

COMMIT = 'c7c2fcd15972ce8b898d8d3ff732ee192c817ceb'
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
PACKAGES = {'torch': '2.11.0+cu128', 'triton': '3.6.0', 'soup-cli': '0.77.0',
            'transformers': '5.16.1', 'peft': '0.20.0', 'trl': '0.29.1',
            'bitsandbytes': '0.50.2', 'accelerate': '1.14.0', 'datasets': '4.8.5'}
DATA = {'data/train.jsonl': '93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd',
        'data/eval_public.jsonl': 'b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e'}
LOCK_PATH = Path('/tmp/soup-gpu0.lock')
BLOCK_PATH = Path('/tmp/soup-gpu0.lock.blocked')
_DEDICATED_RUNNER = False
PREFIXES = ('SOUP_MEMORY_VS1_INSTALL ', 'SOUP_MEMORY_VS1_PROGRESS ', 'SOUP_MEMORY_VS1_RUNTIME ')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition: raise RuntimeError(message)


def public_cache_inventory(cache):
    root = Path(cache).resolve() / 'models--Qwen--Qwen2.5-1.5B-Instruct'
    require((root / 'refs/main').read_text().strip() == REVISION, 'Prepared model refs/main revision mismatch')
    snapshot = root / 'snapshots' / REVISION
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json'}
    index = snapshot / 'model.safetensors.index.json'
    if index.is_file():
        required.add(index.name)
        required.update(json.loads(index.read_text())['weight_map'].values())
    else:
        required.add('model.safetensors')
    require(all((snapshot / name).is_file() for name in required), 'Prepared model snapshot is incomplete; do not download inside run')
    hashes = {}
    for name in sorted(required):
        path = snapshot / name
        require(path.resolve().is_relative_to(root.resolve()), 'Model snapshot link escapes selected public model')
        hashes[name] = sha(path)
    return {'model': MODEL, 'revision': REVISION, 'required_file_hashes': hashes}


def stage(repo, output_root, sources, hashes, protected_hashes):
    repo = Path(repo).resolve()
    require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT,
            'Prepared official checkout must be at the exact starter commit')
    for name, expected in protected_hashes.items():
        require(sha(repo / name) == expected, 'Changed/missing official file: ' + name)
    for name, expected in DATA.items():
        require(sha(repo / name) == expected, 'Fixed public data hash mismatch: ' + name)
    root = Path(output_root).resolve() / ('memory_vs1_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    target = root / 'repo'; target.mkdir()
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', COMMIT])
    archive_path = root / 'official.tar'; archive_path.write_bytes(archive)
    with tarfile.open(archive_path) as tar:
        tar.extractall(target, filter='data')
    (target / 'data').mkdir(exist_ok=True)
    for name in DATA: shutil.copy2(repo / name, target / name)
    shutil.rmtree(target / 'solution')
    (target / 'solution').mkdir()
    for name, source in sources.items():
        require(name.startswith('solution/') and '..' not in Path(name).parts, 'Unsafe embedded source name')
        raw = source.encode()
        require(hashlib.sha256(raw).hexdigest() == hashes[name], 'Embedded candidate source mismatch: ' + name)
        dest = target / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
    for name, expected in protected_hashes.items():
        require(sha(target / name) == expected, 'Staged official file mismatch: ' + name)
    dump(root / 'source_manifest.json', hashes)
    dump(root / 'official_manifest.json', protected_hashes)
    return root, target


def process_table():
    """Linux process identity: PID, parent PID, kernel start ticks, state."""
    table = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            table[int(entry.name)] = {'ppid': int(fields[1]), 'birth': int(fields[19]), 'state': fields[0]}
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Unknown inaccessible processes are never adopted/signalled.
            continue
    return table


class OwnedProcesses:
    """Track only descendants of our child, preserving PID birth identity."""
    def __init__(self, process):
        self.process = process
        table = process_table()
        if process is None:
            self.owned = {pid: row['birth'] for pid, row in table.items() if row['ppid'] == os.getpid()}
        else:
            require(process.pid in table, 'Owned child identity unavailable; cannot safely supervise')
            self.owned = {process.pid: table[process.pid]['birth']}
        self.observe()

    def observe(self):
        table = process_table()
        # Existing owned children remain tracked when reparented or setsid().
        while True:
            live = {pid for pid, birth in self.owned.items()
                    if pid in table and table[pid]['birth'] == birth}
            new = {pid: row['birth'] for pid, row in table.items()
                   if (row['ppid'] in live or (_DEDICATED_RUNNER and row['ppid'] == os.getpid()))
                   and pid not in self.owned}
            if not new: break
            self.owned.update(new)
        return {pid: table[pid] for pid, birth in self.owned.items()
                if pid in table and table[pid]['birth'] == birth and table[pid]['state'] != 'Z'}

    def signal_owned(self, signo):
        for pid, row in self.observe().items():
            # pidfd is the actual signal target: PID reuse after validation
            # cannot redirect the signal to an unrelated process.
            try:
                descriptor = os.pidfd_open(pid, 0)
            except ProcessLookupError:
                continue
            try:
                current = process_table().get(pid)
                if current and current['birth'] == row['birth']:
                    signal.pidfd_send_signal(descriptor, signo, None, 0)
            except ProcessLookupError:
                pass
            finally:
                os.close(descriptor)

    def cleanup(self):
        handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            return self._cleanup()
        finally:
            for sig, handler in handlers.items(): signal.signal(sig, handler)

    def _cleanup(self):
        # Stop known owners before rescanning, so ordinary children cannot
        # keep forking while their parent is being killed. setsid is irrelevant.
        frozen = set()
        while True:
            self.signal_owned(signal.SIGSTOP)
            now = set(self.observe())
            if now <= frozen: break
            frozen.update(now)
        self.signal_owned(signal.SIGKILL)
        if self.process is not None:
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
        # Retain the lock until terminal, including an uninterruptible kernel.
        # An owner can investigate the durable block marker; no blind retry.
        while self.observe():
            self.signal_owned(signal.SIGKILL)
            time.sleep(.05)
        if _DEDICATED_RUNNER:
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if not pid: break
                except ChildProcessError: break
        return {'all_tracked_owned_processes_terminal': not bool(self.observe()),
                'owned_pid_birth_ticks': {str(pid): birth for pid, birth in self.owned.items()},
                'limit': 'Escaped descendants between observations cannot be universally excluded; any interruption blocks lane for reconciliation'}


def command(args, log, cwd, env):
    # Native harness owns normal timeout semantics. We additionally supervise
    # session-escaping descendants for notebook interrupt / wrapper failure.
    start = time.perf_counter()
    process = tracker = None
    try:
        with Path(log).open('x') as stream:
            process = subprocess.Popen(args, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            tracker = OwnedProcesses(process)
            while True:
                tracker.observe()
                try:
                    code = process.wait(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            remaining = tracker.observe()
            if remaining:
                cleanup = tracker.cleanup()
                dump(BLOCK_PATH, {'reason': 'owned descendants survived wrapper exit', 'cleanup': cleanup, 'log': str(log)})
                raise RuntimeError('Owned descendants survived command exit; GPU lane blocked for reconciliation')
    except BaseException as exc:
        dump(BLOCK_PATH, {'reason': type(exc).__name__, 'cleanup': 'IN_PROGRESS', 'log': str(log),
                          'manual_owner_reconciliation_required': True})
        cleanup = tracker.cleanup() if tracker is not None else {'identity_tracking_available': False}
        # Even successful tracked cleanup cannot prove a short-lived owner
        # never orphaned a descendant between observations. Do not auto-retry.
        dump(BLOCK_PATH, {'reason': type(exc).__name__, 'cleanup': cleanup, 'log': str(log),
                          'manual_owner_reconciliation_required': True})
        dump(str(log)+'.interrupted.json', {'status': 'INTERRUPTED_OR_UNCERTAIN', 'cleanup': cleanup,
                                            'score': None, 'admission': 'UNKNOWN'})
        raise
    return {'command': args, 'exit_code': code,
            'whole_command_wall_s': time.perf_counter() - start, 'log': str(log)}


def summarize_evidence(result_path, root):
    raw = json.loads(Path(result_path).read_text()) if Path(result_path).is_file() else {}
    errors, routes = [], []
    if raw.get('packages_changed'): errors.append('packages changed')
    runs = raw.get('runs', {})
    for side in ('base', 'solution'):
        entries = runs.get(side, [])
        if len(entries) != 2: errors.append(side + ': expected two fixed-task runs')
        for run in entries:
            metrics = run.get('metrics') or {}
            required = (run.get('exit_code') == 0, metrics.get('exit_code') == 0,
                        metrics.get('steps') == 64, run.get('global_step') == 64,
                        bool(run.get('adapter_tensors')), bool(run.get('adapter_config')),
                        isinstance(run.get('wall_s'), (float, int)) and run.get('wall_s', 0) > 0,
                        isinstance(run.get('gpu_peak_mib'), (float, int)) and run.get('gpu_peak_mib', 0) > 0)
            if not all(required): errors.append(side + '/' + run.get('run', '?') + ': incomplete native run')
            if side == 'solution':
                log = Path(root) / 'runs' / run['run'] / 'train.log'
                events = []
                if log.is_file():
                    for line in log.read_text(errors='replace').splitlines():
                        for prefix in PREFIXES:
                            if line.startswith(prefix):
                                try: events.append({'kind': prefix.strip(), 'value': json.loads(line[len(prefix):])})
                                except ValueError: pass
                routes.append({'run': run['run'], 'events': events,
                               'own_kernel_ms': metrics.get('own_kernel_ms'),
                               'timed_loop_s': metrics.get('timed_loop_s'),
                               'timed_steps': metrics.get('timed_steps')})
                finals = [e['value'] for e in events if e['kind'] == 'SOUP_MEMORY_VS1_RUNTIME']
                if not finals or finals[-1].get('encoded_operands', 0) <= 0 or finals[-1].get('restores', 0) <= 0:
                    errors.append(run['run'] + ': final nonvacuous codec activation absent')
        quality = raw.get('evals', {}).get(side, {})
        value = quality.get('mean_loss')
        if quality.get('exit_code') != 0 or not isinstance(value, (float, int)) or not math.isfinite(value):
            errors.append(side + ': finite successful plain public evaluation absent')
    # A crash can leave flushed train.log events without any result.json.
    recovered_logs = []
    for log in sorted((Path(root) / 'runs').glob('solution_*/train.log')):
        events = []
        for line in log.read_text(errors='replace').splitlines():
            for prefix in PREFIXES:
                if line.startswith(prefix):
                    try: events.append({'kind': prefix.strip(), 'value': json.loads(line[len(prefix):])})
                    except ValueError: pass
        recovered_logs.append({'run': log.parent.name, 'events': events,
                               'partial_unless_complete_result': True})
    report = {'candidate_id': 'soup-memory-vs1-20261030',
              'status': 'COMPLETE_COMPONENT_MEASUREMENT' if not errors else 'FAILED_OR_INCOMPLETE',
              'errors': errors, 'score': None, 'admission': 'UNKNOWN',
              'standalone_eligibility': 'Useful own Triton share must be independently measured; may be below 5%',
              'routes': routes, 'timeout_surviving_logs': recovered_logs, 'raw_summary_untrusted_if_incomplete': raw.get('summary'),
              'ratios': raw.get('summary', {}).get('ratios') if not errors else None}
    dump(Path(root) / 'review_summary.json', report)
    return report


def create_active_marker(token):
    # Exclusive create + fsync happens before any GPU-capable subprocess. A
    # SIGKILL therefore leaves an owner-bound reconciliation marker.
    with BLOCK_PATH.open('x') as handle:
        json.dump({'owner_token': token, 'phase': 'active', 'supervisor_pid': os.getpid(),
                   'supervisor_birth_ticks': process_table()[os.getpid()]['birth'],
                   'manual_owner_reconciliation_required_unless_cleanly_removed': True}, handle)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def clear_own_normal_completion(token):
    # Never clear another owner's marker or an interrupted/uncertain marker.
    try: marker = json.loads(BLOCK_PATH.read_text())
    except (OSError, ValueError): return False
    if marker.get('owner_token') != token or marker.get('phase') != 'active': return False
    BLOCK_PATH.unlink()
    return True


def run_prepared(**kwargs):
    require(kwargs.get('authorize_gpu') is True, 'Prepared only: explicit future target authorization required')
    require(_DEDICATED_RUNNER, 'GPU workflow must run in dedicated supervisor process, never the notebook kernel')
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd supervision is required')
    # One canonical cross-notebook GPU0 lease; never wait or run concurrently.
    require(not BLOCK_PATH.exists(), 'GPU0 lane has unresolved cleanup; owner reconciliation is required')
    with LOCK_PATH.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('GPU0 is leased by another Soup notebook/run') from exc
        token = uuid.uuid4().hex
        normal_verified_completion = False
        try:
            require(not BLOCK_PATH.exists(), 'GPU0 became blocked before lease acquisition')
            create_active_marker(token)
            outcome = _run_prepared_locked(**kwargs)
            normal_verified_completion = outcome[1].get('clean_process_completion') is True
            return outcome
        except (KeyboardInterrupt, InterruptedError) as exc:
            dump(BLOCK_PATH, {'owner_token': token, 'reason': type(exc).__name__, 'phase': 'supervisor interrupted',
                              'manual_owner_reconciliation_required': True})
            raise
        finally:
            # Also cover failures between supervised commands (e.g. inventory).
            remaining = OwnedProcesses(None)
            if remaining.observe():
                normal_verified_completion = False
                dump(BLOCK_PATH, {'owner_token': token, 'reason': 'supervisor exit with owned children', 'cleanup': 'IN_PROGRESS',
                                  'manual_owner_reconciliation_required': True})
                cleanup = remaining.cleanup()
                dump(BLOCK_PATH, {'owner_token': token, 'reason': 'supervisor exit with owned children', 'cleanup': cleanup,
                                  'manual_owner_reconciliation_required': True})
            if normal_verified_completion:
                clear_own_normal_completion(token)
            # Every failure, native timeout, interrupt or abrupt death retains
            # the marker; only clean owned completion removes its own token.
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_prepared_locked(*, authorize_gpu, repo, output_root, hf_hub_cache, sources, hashes,
                         protected_hashes, timeout_s=900):
    require(authorize_gpu is True, 'Prepared only: set AUTHORIZE_GPU=True only for an explicitly authorized target run')
    require(sys.version_info[:3] == (3, 13, 15), 'Exact Python3.13.15 required; no installation is performed')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == PACKAGES, 'Exact direct package pins required; no installation is performed')
    require(timeout_s == 900, 'This notebook freezes a 900-second native budget for both sides; new cohort requires review')
    require(len(os.sched_getaffinity(0)) == 4, 'Use exactly four assigned logical CPUs; do not assume CPU IDs')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_CACHE=str(Path(hf_hub_cache).resolve()), TOKENIZERS_PARALLELISM='false')
    # Inventory is a separate process; the notebook never creates its own CUDA context.
    inventory_code = '''import json,subprocess
active=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
assert not active.stdout.strip(), "GPU compute process present before inventory"
import torch
assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=="Tesla T4"
assert torch.cuda.get_device_capability(0)==(7,5) and torch.version.cuda=="12.8"
p=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
rows=[tuple(int(x.strip()) for x in line.split(",")) for line in p.stdout.splitlines()]
assert rows and sum(x[1] for x in rows)<1000, "All exposed GPUs must be idle"
print(json.dumps({"gpu":"Tesla T4","visible_count":1,"torch":torch.__version__,"torch_git":torch.version.git_version,"external_gpu_memory":rows}))'''
    inventory = json.loads(subprocess.check_output([sys.executable, '-c', inventory_code], env=env, text=True))
    cache = public_cache_inventory(hf_hub_cache)
    root, target = stage(repo, output_root, sources, hashes, protected_hashes)
    dump(root / 'environment.json', {'versions': versions, 'gpu': inventory, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                                     'python': sys.version, 'public_cache': cache, 'native_timeout_s': timeout_s})
    # Isolated preflight caches are never inherited by measured runs.
    preenv = dict(env, TRITON_CACHE_DIR=str(root/'preflight_triton'), TORCHINDUCTOR_CACHE_DIR=str(root/'preflight_inductor'))
    preflight = command([sys.executable, '-B', '-m', 'solution.verify_memory', '--out', str(root/'preflight.json')], root/'preflight.log', target, preenv)
    dump(root / 'preflight_command.json', preflight)
    require(preflight['exit_code'] == 0, 'Target preflight failed; retained evidence at ' + str(root))
    regressions = command([sys.executable, '-B', '-m', 'pytest', 'tests', '-q'], root/'regressions.log', target,
                          dict(env, SOUP_APPLY_SOLUTION='1'))
    dump(root / 'regression_command.json', regressions)
    require(regressions['exit_code'] == 0, 'Unchanged Soup regressions failed; do not measure')
    # Preflight/regression children have exited before this fresh idle check.
    # This short-lived inventory child also exits before measured Soup starts.
    post_preflight_inventory = json.loads(subprocess.check_output(
        [sys.executable, '-c', inventory_code], env=env, text=True))
    dump(root/'idle_after_preflight.json', post_preflight_inventory)
    measured = command([sys.executable, '-B', 'harness/run_task.py', '--repeats', '2', '--seed', '1',
                        '--timeout', str(timeout_s), '--out', str(root/'runs')], root/'harness.log', target, env)
    dump(root / 'harness_command.json', measured)
    # Read/report partial routes even when the wrapper or native task failed.
    report = summarize_evidence(root/'runs/result.json', root)
    if measured['exit_code'] != 0:
        report.update(status='FAILED_OR_INCOMPLETE', score=None, admission='UNKNOWN', ratios=None)
        report['errors'].append('official harness command failed')
        dump(root/'review_summary.json', report)
    for name, expected in protected_hashes.items():
        require(sha(target/name) == expected, 'Protected official file changed during execution: '+name)
    report['clean_process_completion'] = False
    if report['status'] == 'COMPLETE_COMPONENT_MEASUREMENT':
        checks = command([sys.executable, '-B', 'harness/check_rules.py', '--result', str(root/'runs/result.json')], root/'official_checks.log', target, env)
        dump(root/'checks_command.json', checks)
        report['clean_process_completion'] = checks['exit_code'] == 0
        # A component result is never converted to admission PASS or points.
    dump(root/'review_summary.json', report)
    print(json.dumps({'evidence': str(root), 'status': report['status'], 'score': None, 'admission': 'UNKNOWN'}, indent=2))
    return root, report


def main():
    global _DEDICATED_RUNNER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    require(config.get('authorize_gpu') is True, 'Explicit future GPU authority required')
    require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0,
            'Cannot establish dedicated Linux child subreaper')
    _DEDICATED_RUNNER = True
    def interrupted(signum, frame):
        raise InterruptedError('Dedicated runner interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_prepared(**config)


if __name__ == '__main__': main()
