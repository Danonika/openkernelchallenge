#!/usr/bin/env python3
"""Serial notebook worker: fixed Docker boundary, no credential or host-code mount."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import shlex
import stat
import subprocess
import time
import threading
import urllib.request
import zipfile
from pathlib import Path
from runner_adapter import host_gpu_lock, GPU_LOCK_PATH, output_bytes
from soup_queue import require, QueueError
from notebook_queue import MAX_OUTPUT
from notebook_failfast import error, valid_error, container_failure
from notebook_reference import load_reference

ROOT = Path(__file__).resolve().parent
LABEL = 'org.soup.notebook.worker'


def training_process_deadline(output, *, process_root=Path('/proc'), now=None, minimum=80):
    """Independent host watchdog if a subprocess removes the Python startup hook."""
    now = time.clock_gettime(time.CLOCK_BOOTTIME) if now is None else now
    ticks_per_second = os.sysconf('SC_CLK_TCK')
    for line in output.splitlines()[1:]:
        pieces = line.strip().split(None, 1)
        if len(pieces) != 2 or not pieces[0].isdigit():
            continue
        try:
            argv = shlex.split(pieces[1])
        except ValueError:
            continue
        if not any(Path(arg).name == '_train.py' for arg in argv):
            continue
        try:
            fields = (process_root / pieces[0] / 'stat').read_text().rsplit(') ', 1)[1].split()
            elapsed = now - int(fields[19]) / ticks_per_second
        except (FileNotFoundError, ProcessLookupError):
            continue
        if elapsed > minimum:
            name = 'training'
            if '--config' in argv and argv.index('--config') + 1 < len(argv):
                name = Path(argv[argv.index('--config') + 1]).parent.name
            return {'reason': 'training_run_exceeded_limit', 'limit_s': 80,
                    'elapsed_s': elapsed, 'run_name': name,
                    'enforcement': 'host_process_watchdog', 'deadline_reached': elapsed > 80}
    return None


class GpuSampler:
    """Host-owned samples survive a killed notebook/kernel; job-wide scope."""
    def __init__(self):
        self.peak, self.current, self.samples = None, None, 0
        self.process = None

    def start(self):
        try:
            self.process = subprocess.Popen(['/usr/bin/nvidia-smi', '-i', '0',
                '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '100'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self.read, daemon=True)
            self.thread.start()
        except OSError:
            pass

    def read(self):
        for line in self.process.stdout:
            try:
                value = int(line.strip())
            except ValueError:
                continue
            self.current = value
            self.peak = max(self.peak or 0, value)
            self.samples += 1

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.thread.join(timeout=2)
            self.process.stdout.close()


class Worker:
    def __init__(self, config, token):
        self.c = config
        self.token = token
        require(re.fullmatch(r'sha256:[0-9a-f]{64}', config['image_id']), 'Immutable local image ID required')
        require(config['queue_url'] == 'http://127.0.0.1:8765', 'Worker talks only to local API')
        self.root = Path(config['workspace_root'])
        self.root.mkdir(mode=0o700, exist_ok=True)

    def request(self, method, path, body=None, binary=False):
        data = body if binary else json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.c['queue_url'] + path, data=data, method=method,
            headers={'Authorization': 'Bearer ' + self.token,
                     'Content-Type': 'application/zip' if binary else 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read(MAX_OUTPUT + 1)
        require(len(raw) <= MAX_OUTPUT, 'API response exceeded bound')
        return raw if path.endswith('/input') else json.loads(raw)

    def docker(self, *args, check=True, timeout=60):
        result = subprocess.run(['/usr/bin/docker', *args], capture_output=True, timeout=timeout,
                                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        if check and result.returncode:
            raise RuntimeError('Docker operation failed: ' + args[0] + ': ' + result.stderr.decode(errors='replace')[-1500:])
        return result

    def inspect(self, name):
        result = self.docker('inspect', name, check=False)
        if result.returncode:
            # Prove daemon connectivity so daemon failure is never treated as cleanup.
            self.docker('info', '--format', '{{.ServerVersion}}')
            return None
        info = json.loads(result.stdout)[0]
        require(info['Config'].get('Labels', {}).get(LABEL) == 'v1', 'Foreign container; cleanup refused')
        return info

    def cleanup(self, name):
        try:
            if self.inspect(name) is not None:
                self.docker('rm', '--force', name)
            require(self.inspect(name) is None, 'Container still present')
        except Exception:
            marker = Path(str(GPU_LOCK_PATH) + '.blocked')
            marker.write_text(json.dumps({'container': name, 'reason': 'notebook_container_cleanup_unknown'}))
            raise

    def recover(self):
        # Called with the shared GPU flock held. No automatic rerun after an interruption.
        jobs = self.request('GET', '/v1/notebook-worker/running')['jobs']
        for job in jobs:
            name = 'soupnb-' + job['id']
            logs = self.docker('logs', '--tail=2000', name, check=False)
            self.cleanup(name)
            result = {
                'state': 'interrupted', 'exit_code': None, 'elapsed_s': max(0, time.time() - job['started']),
                'container_cleanup_confirmed': True, 'reason': 'Worker restarted; container removal verified; submit a new request to rerun',
                'image_id': self.c['image_id']}
            run = self.root / job['id']
            if not job['output_sha'] and run.is_dir():
                bundle = self.collect(run, (logs.stdout + logs.stderr)[-2 * 1024**2:], result)
                self.request('POST', '/v1/notebooks/' + job['id'] + '/outputs', bundle, binary=True)
            self.request('POST', '/v1/notebooks/' + job['id'] + '/finish', result)
            print(json.dumps({'recovered_interrupted': job['id']}), flush=True)

    def argv(self, name, run, job):
        args = ['create', '--pull=never', '--name', name, '--label', LABEL + '=v1',
            '--network=none', '--user=10001:10001', '--read-only', '--cap-drop=ALL',
            '--security-opt=no-new-privileges:true', '--ipc=private', '--pids-limit=512',
            '--cpuset-cpus=0,1,2,3', '--gpus=device=0', '--memory=30g', '--memory-swap=30g',
            '--memory-reservation=26g', '--oom-score-adj=500', '--shm-size=30g',
            '--log-driver=local', '--log-opt=max-size=10m', '--log-opt=max-file=1', '--log-opt=compress=false',
            '--workdir=/kaggle/working',
            '--entrypoint=/usr/local/bin/python']
        mounts = [(run / 'input', '/input', True), (run / 'launcher', '/runner', True),
            (run / 'output', '/output', False), (run / 'output' / 'work', '/kaggle/working', False),
            (run / 'output' / 'tmp', '/tmp', False),
            (Path(self.c['repo']), '/task', True),
            (Path(self.c['repo']) / 'data', '/kaggle/input/soup-data', True),
            (Path(self.c['model_cache']), '/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct', True)]
        for source, target, readonly in mounts:
            require(source.is_absolute() and ',' not in str(source), 'Invalid fixed mount')
            args += ['--mount', 'type=bind,src=' + str(source) + ',dst=' + target + (',readonly' if readonly else '')]
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'USER': 'soup-candidate', 'LOGNAME': 'soup-candidate',
            'HOME': '/output/home', 'TMPDIR': '/output/tmp', 'LANG': 'C.UTF-8',
            'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
            'PYTHONPATH': '/runner',
            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
            'HF_HOME': '/output/home/redacted', 'HF_HUB_CACHE': '/hf_cache', 'HF_HUB_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'CUDA_VISIBLE_DEVICES': '0', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'SOUP_NO_AUDIT_LOG': '1',
            'SOUP_REPO': '/kaggle/working/repo', 'SOUP_RESULTS': '/kaggle/working/results',
            'TRITON_CACHE_DIR': '/output/tmp/triton', 'TORCHINDUCTOR_CACHE_DIR': '/output/tmp/inductor',
            'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'}
        for key, value in env.items():
            args += ['--env', key + '=' + value]
        return args + [self.c['image_id'], '-B', '/runner/notebook_runner.py', str(job['cell_timeout'])]

    def collect(self, run, log, result):
        buffer = io.BytesIO()
        used, count = len(log), 2
        warnings = []
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('execution.log', log)
            # Host result is authoritative only for process lifecycle, never submitted claims.
            archive.writestr('runner.json', json.dumps(result, allow_nan=False))
            host_metrics = run / 'host-metrics.json'
            if host_metrics.exists():
                archive.writestr('results/host-metrics.json', host_metrics.read_bytes())
                count += 1
            host_limit = run / 'host-training-limit.json'
            if host_limit.exists():
                archive.writestr('results/training-limit.json', host_limit.read_bytes())
                count += 1
            targets = [(('delivery',), False), (('work', 'results'), True)]
            for components, results in targets:
                # Open each component relative to a directory descriptor; candidate-written
                # symlink ancestors must never redirect the host collector outside output.
                basefd = os.open(run / 'output', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    for component in components:
                        nextfd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=basefd)
                        os.close(basefd)
                        basefd = nextfd
                except OSError:
                    os.close(basefd)
                    continue
                scanned = 0
                try:
                  for parent, dirs, files, dirfd in os.fwalk('.', dir_fd=basefd, follow_symlinks=False):
                    scanned += len(dirs) + len(files)
                    if scanned > 1024:
                        warnings.append('Output traversal stopped after 1024 entries')
                        break
                    dirs[:] = [d for d in dirs if not stat.S_ISLNK(os.stat(d, dir_fd=dirfd, follow_symlinks=False).st_mode)]
                    for filename in files:
                        if not results and parent == '.' and filename not in ('executed.ipynb', 'training_limit.json', 'failure.json', 'screening.json', 'run-metrics.json'):
                            continue
                        if not results and parent != '.' and Path(parent).parts[0] != 'harness':
                            continue
                        if not results and filename == 'training_limit.json' and host_limit.exists():
                            continue
                        try:
                            fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd)
                            with os.fdopen(fd, 'rb') as handle:
                                metadata = os.fstat(handle.fileno())
                                require(stat.S_ISREG(metadata.st_mode), 'Not a regular file')
                                require(count < 127 and used + metadata.st_size < MAX_OUTPUT - 65536, 'Output budget exceeded')
                                data = handle.read(min(metadata.st_size + 1, MAX_OUTPUT))
                                require(len(data) == metadata.st_size, 'Output changed during collection')
                            relative = str(Path(parent) / filename)
                            require('\\' not in relative, 'Invalid filename')
                            if relative in ('collection-warnings.json', 'host-metrics.json') or results and (relative in ('training-limit.json', 'failure.json', 'screening.json', 'run-metrics.json') or relative.startswith('harness/')):
                                continue
                            archive_name = 'executed.ipynb' if not results and relative == 'executed.ipynb' else 'results/training-limit.json' if not results and relative == 'training_limit.json' else 'results/' + relative
                            archive.writestr(archive_name, data)
                            used += len(data)
                            count += 1
                        except (OSError, QueueError) as exc:
                            if len(warnings) < 20:
                                warnings.append(type(exc).__name__ + ': ' + str(exc)[:200])
                finally:
                    os.close(basefd)
            if warnings:
                archive.writestr('results/collection-warnings.json', json.dumps(warnings))
        raw = buffer.getvalue()
        require(len(raw) <= MAX_OUTPUT, 'Compressed output exceeds budget')
        return raw

    def training_limit_failure(self, run):
        raw = output_bytes(run / 'output', 'delivery/training_limit.json', limit=4096, optional=True)
        if raw is None:
            return None
        record = json.loads(raw)
        require(record.get('reason') == 'training_run_exceeded_limit' and record.get('limit_s') == 80, 'Invalid training deadline evidence')
        elapsed = record.get('elapsed_s')
        require(type(elapsed) in (int, float) and 0 <= elapsed < 1000000, 'Invalid training wall time')
        require(elapsed > 80 or record.get('deadline_reached') is True, 'Training limit was not exceeded')
        name = record.get('run_name')
        require(isinstance(name, str) and len(name) <= 100, 'Invalid run name')
        return f'{name} exceeded 80 seconds (measured {elapsed:.3f} s); FAIL'

    def screening_failure(self, run):
        raw = output_bytes(run / 'output', 'delivery/failure.json', limit=4096, optional=True)
        if raw is None:
            return None
        record = json.loads(raw)
        require(valid_error(record), 'Invalid screening failure evidence')
        return record

    def execute(self, job):
        jid = job['id']
        require(re.fullmatch(r'n_[0-9a-f]{32}', jid), 'Invalid server job ID')
        name = 'soupnb-' + jid
        run = self.root / jid
        run.mkdir(mode=0o700)
        for subdir in ('input', 'launcher', 'output', 'output/work', 'output/home', 'output/tmp', 'output/delivery'):
            path = run / subdir
            path.mkdir()
            path.chmod(0o1777 if subdir == 'output/tmp' else 0o777 if subdir.startswith('output') else 0o755)
        raw = self.request('GET', '/v1/notebooks/' + jid + '/input')
        require(hashlib.sha256(raw).hexdigest() == job['execution_sha'], 'Notebook hash mismatch')
        (run / 'input' / 'input.ipynb').write_bytes(raw)
        (run / 'input' / 'input.ipynb').chmod(0o644)
        shutil.copyfile(ROOT / 'notebook_runner.py', run / 'launcher' / 'notebook_runner.py')
        (run / 'launcher' / 'notebook_runner.py').chmod(0o644)
        for source, target in (('training_run_limit.py', 'training_run_limit.py'), ('notebook_failfast.py', 'notebook_failfast.py'), ('notebook_reference.py', 'notebook_reference.py'), ('notebook_sitecustomize.py', 'sitecustomize.py')):
            shutil.copyfile(ROOT / source, run / 'launcher' / target)
            (run / 'launcher' / target).chmod(0o644)
        reference = load_reference()
        if reference.get('image_id') == self.c['image_id']:
            (run / 'launcher' / 'public-baseline.json').write_text(json.dumps(reference))
            (run / 'launcher' / 'public-baseline.json').chmod(0o644)
        start = time.monotonic()
        state, code, reason, log = 'failed', None, 'Container did not start', b''
        failure = None
        info = None
        sampler = GpuSampler()
        host_runs = {}
        try:
            self.docker(*self.argv(name, run, job))
            sampler.start()
            self.docker('start', name)
            while True:
                info = self.inspect(name)
                require(info is not None, 'Running container disappeared')
                failure = container_failure(info['State'], self.screening_failure(run))
                if failure:
                    state = 'timed_out' if failure['code'] == 'TIME_LIMIT_EXCEEDED' and failure['phase'] == 'cell' else 'failed'
                    reason = failure['message']
                    if not info['State']['Running']:
                        code = info['State']['ExitCode']
                    break
                limit_failure = self.training_limit_failure(run)
                if limit_failure:
                    state, reason = 'failed', limit_failure
                    failure = error('TIME_LIMIT_EXCEEDED', reason, 'training', limit=80, unit='seconds')
                    if not info['State']['Running']:
                        code = info['State']['ExitCode']
                    break
                if not info['State']['Running']:
                    code = info['State']['ExitCode']
                    state = 'completed' if code == 0 else 'timed_out' if code == 124 else 'failed'
                    reason = 'Container exited' if not info['State'].get('OOMKilled') else 'Container killed for memory pressure'
                    if info['State'].get('OOMKilled'):
                        state = 'failed'
                        failure = error('MEMORY_LIMIT_EXCEEDED', reason, 'container', limit=30, unit='GiB')
                    break
                top = self.docker('top', name, '-eo', 'pid,args', check=False)
                if top.returncode == 0:
                    observed = training_process_deadline(top.stdout.decode(errors='replace'), minimum=0)
                    if observed:
                        previous = host_runs.get(observed['run_name'], {})
                        host_runs[observed['run_name']] = {
                            'run': observed['run_name'], 'wall_s': round(observed['elapsed_s'], 3),
                            'gpu_peak_mib': max(previous.get('gpu_peak_mib') or 0, sampler.current or 0),
                            'partial': True, 'scope': 'host_observed_training'}
                    exceeded = training_process_deadline(top.stdout.decode(errors='replace'))
                    if exceeded:
                        (run / 'host-training-limit.json').write_text(json.dumps(exceeded, allow_nan=False))
                        state, reason = 'failed', f"{exceeded['run_name']} exceeded 80 seconds (measured {exceeded['elapsed_s']:.3f} s); FAIL"
                        failure = error('TIME_LIMIT_EXCEEDED', reason, 'training', run=exceeded['run_name'],
                                        measured=exceeded['elapsed_s'], limit=80, unit='seconds')
                        break
                heartbeat = self.request('POST', '/v1/notebooks/' + jid + '/heartbeat', {})
                if heartbeat['cancel_requested']:
                    state, reason = 'cancelled', 'Cancellation requested'
                    break
                if time.monotonic() - start >= job['timeout']:
                    state, reason = 'timed_out', 'Whole-notebook deadline exceeded'
                    failure = error('TIME_LIMIT_EXCEEDED', reason, 'notebook', limit=job['timeout'], unit='seconds')
                    break
                if shutil.disk_usage(run).free < 512 * 1024**2:
                    state, reason = 'failed', 'Workspace storage limit reached'
                    failure = error('STORAGE_LIMIT_EXCEEDED', reason, 'container')
                    break
                time.sleep(2)
            if info and info['State']['Running']:
                self.docker('kill', name)
                self.docker('wait', name, timeout=30)
                info = self.inspect(name)
                code = info['State']['ExitCode']
            logs = self.docker('logs', '--tail=2000', name, check=False)
            log = (logs.stdout + logs.stderr)[-2 * 1024**2:]
        except Exception as exc:
            state, reason = 'failed', type(exc).__name__ + ': ' + str(exc)[:350]
        finally:
            try:
                self.cleanup(name)
            finally:
                sampler.stop()
                (run / 'host-metrics.json').write_text(json.dumps({
                    'gpu_peak_mib': sampler.peak, 'gpu_scope': 'whole_notebook',
                    'sample_interval_ms': 100, 'samples': sampler.samples,
                    'runs': list(host_runs.values())}, allow_nan=False))
        if failure is None and state in ('failed', 'timed_out'):
            failure = error('TIME_LIMIT_EXCEEDED' if state == 'timed_out' else 'EXECUTION_ERROR', reason, 'container')
        result = {'state': state, 'exit_code': code, 'elapsed_s': round(time.monotonic() - start, 3),
            'container_cleanup_confirmed': True, 'reason': reason, 'image_id': self.c['image_id'],
            'error': failure}
        bundle = self.collect(run, log, {**result, 'source_sha256': job['source_sha'],
            'execution_sha256': job['execution_sha'], 'admission': 'UNKNOWN', 'score': None,
            'scope': 'notebook_process_execution', 'container': info})
        # Write a supervisor-owned receipt for recovery/audit, never execute notebook outputs.
        (run / 'outputs.zip').write_bytes(bundle)
        (run / 'receipt.json').write_text(json.dumps(result, indent=2))
        self.request('POST', '/v1/notebooks/' + jid + '/outputs', bundle, binary=True)
        self.request('POST', '/v1/notebooks/' + jid + '/finish', result)
        print(json.dumps({'job': jid, **result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    worker = Worker(json.loads(Path(args.config).read_text()), os.environ['SOUP_QUEUE_WORKER_TOKEN'])
    # Prevent a second daemon from reconciling the first one's live job.
    with (worker.root / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                with host_gpu_lock():
                    worker.recover()
                    if shutil.disk_usage(worker.root).free < 2 * 1024**3:
                        raise QueueError('Workspace low on space; worker paused until storage is freed')
                    job = worker.request('POST', '/v1/notebook-worker/claim', {})['job']
                    if job:
                        worker.execute(job)
            except Exception as exc:
                print(json.dumps({'worker_error': type(exc).__name__, 'message': str(exc)[:500]}), flush=True)
                if args.once:
                    raise
                time.sleep(10)
            if args.once:
                return
            time.sleep(2)


if __name__ == '__main__':
    main()
