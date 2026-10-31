"""Durable, automatic notebook lane. Only the separately authenticated worker executes.

This is execution reporting, never benchmark admission or a scientific score.
The original reviewed solution-bundle queue remains a separate lane.
"""
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from soup_queue import QueueError, require, fields
from notebook_failfast import valid_error
from notebook_metrics import VERSION as METRICS_VERSION, summarize_archive, comparison
from notebook_reference import load_reference

MAX_NOTEBOOK = 8 * 1024**2
MAX_OUTPUT = 64 * 1024**2
JOB_PATTERN = r'n_[0-9a-f]{32}'
TERMINAL = {'completed', 'failed', 'timed_out', 'interrupted', 'cancelled'}


def clean_notebook(raw):
    require(0 < len(raw) <= MAX_NOTEBOOK, 'Notebook size must be 1 byte to 8 MiB')
    nb = json.loads(raw)
    require(isinstance(nb, dict) and nb.get('nbformat') == 4, 'Notebook v4 required')
    require(isinstance(nb.get('cells'), list) and 0 < len(nb['cells']) <= 512, 'Expected 1..512 cells')
    cells = []
    for cell in nb['cells']:
        require(isinstance(cell, dict) and cell.get('cell_type') in ('code', 'markdown', 'raw'), 'Invalid notebook cell')
        source = cell.get('source', '')
        if isinstance(source, list):
            require(all(isinstance(x, str) for x in source), 'Cell source must contain text')
            source = ''.join(source)
        require(isinstance(source, str), 'Cell source must be text')
        # Never inherit saved outputs, skip/raises-exception tags, kernelspecs or widgets.
        item = {'cell_type': cell['cell_type'], 'source': source, 'metadata': {}, 'id': uuid.uuid4().hex[:12]}
        if cell['cell_type'] == 'code':
            item.update(outputs=[], execution_count=None)
        cells.append(item)
    require(any(x['cell_type'] == 'code' for x in cells), 'At least one code cell required')
    return json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}}, 'cells': cells}, ensure_ascii=False).encode()


def validate_outputs(raw):
    require(0 < len(raw) <= MAX_OUTPUT, 'Output bundle exceeds 64 MiB')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            require(0 < len(entries) <= 128, 'Output bundle must have 1..128 files')
            names = set()
            total = 0
            for entry in entries:
                name = entry.filename
                path = PurePosixPath(name)
                require(name not in names and '\\' not in name and not path.is_absolute() and all(p not in ('..', '.') for p in path.parts), 'Unsafe output filename')
                require(name in ('executed.ipynb', 'execution.log', 'runner.json') or name.startswith('results/'), 'Unexpected output filename')
                require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16), 'Output entries must be regular files')
                total += entry.file_size
                require(total <= MAX_OUTPUT and not entry.flag_bits & 1, 'Expanded output too large or encrypted')
                names.add(name)
            require({'execution.log', 'runner.json'} <= names, 'Missing execution evidence')
            require(archive.testzip() is None, 'Corrupt output ZIP')
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise QueueError('Invalid output ZIP') from exc


class NotebookQueue:
    def __init__(self, root, config):
        self.root = Path(root) / 'notebooks'
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.config = config
        self.reference = load_reference()
        self.reference_sha = hashlib.sha256(json.dumps(self.reference, sort_keys=True).encode()).hexdigest()
        with self.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, candidate TEXT NOT NULL,
                source_sha TEXT NOT NULL, execution_sha TEXT NOT NULL, state TEXT NOT NULL,
                created REAL NOT NULL, started REAL, finished REAL, heartbeat REAL,
                timeout INTEGER NOT NULL, cell_timeout INTEGER NOT NULL,
                submitted_by TEXT NOT NULL, worker TEXT, result TEXT, output_sha TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0)''')
            if 'rerun_of' not in {row[1] for row in db.execute('PRAGMA table_info(jobs)')}:
                db.execute('ALTER TABLE jobs ADD COLUMN rerun_of TEXT')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / 'notebooks.sqlite3', timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def describe(self):
        return {**self.config, 'mode': 'automatic_authenticated_notebooks', 'review_required': False,
                'max_notebook_bytes': MAX_NOTEBOOK, 'max_output_bytes': MAX_OUTPUT,
                'default_timeout_s': 3600, 'max_timeout_s': 14400,
                'network': 'none', 'concurrency': 1, 'admission': 'UNKNOWN', 'score': None}

    def _job(self, row):
        result = dict(row)
        result['result'] = json.loads(result['result']) if result['result'] else None
        result['cancel_requested'] = bool(result['cancel_requested'])
        result['outputs_url'] = '/v1/notebooks/' + result['id'] + '/outputs' if result['output_sha'] else None
        metrics = {'version': METRICS_VERSION, 'available': False, 'scope': 'reported_public_metrics'}
        if result['output_sha']:
            directory = self.root / result['id']
            cache = directory / 'metrics-summary.json'
            try:
                saved = json.loads(cache.read_text())
                if saved.get('version') != METRICS_VERSION or saved.get('output_sha') != result['output_sha'] or saved.get('reference_sha') != self.reference_sha:
                    raise ValueError('Metrics cache changed')
                metrics = saved
            except (OSError, ValueError):
                metrics = {**summarize_archive(directory / 'outputs.zip', self.reference, result['output_sha']),
                           'output_sha': result['output_sha'], 'reference_sha': self.reference_sha}
                temporary = directory / ('metrics-' + uuid.uuid4().hex + '.tmp')
                temporary.write_text(json.dumps(metrics, allow_nan=False))
                os.replace(temporary, cache)
        result['metrics'] = {**metrics, 'comparison': comparison(metrics, result['state'])}
        return result

    def status(self, jid=None):
        with self.db() as db:
            if jid is None:
                return [self._job(row) for row in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 100')]
            row = db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
            require(row is not None, 'Unknown notebook job')
            return self._job(row)

    def submit(self, body, subject, rerun_of=None):
        fields(body, {'candidate_id', 'request_id', 'notebook_base64', 'contains_secrets'}, {'timeout_s', 'cell_timeout_s'})
        require(body['contains_secrets'] is False, 'Remove secrets before submitting a notebook')
        for key in ('candidate_id', 'request_id'):
            require(isinstance(body[key], str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', body[key]), key + ' must be a short identifier')
        timeout = body.get('timeout_s', 3600)
        cell_timeout = body.get('cell_timeout_s', min(1800, timeout))
        require(type(timeout) is int and 10 <= timeout <= 14400, 'timeout_s must be 10..14400')
        require(type(cell_timeout) is int and 1 <= cell_timeout <= timeout, 'Invalid cell timeout')
        encoded = body['notebook_base64']
        require(isinstance(encoded, str) and len(encoded) <= 4 * ((MAX_NOTEBOOK + 2) // 3), 'Notebook too large')
        raw = base64.b64decode(encoded, validate=True)
        cleaned = clean_notebook(raw)
        sha = hashlib.sha256(raw).hexdigest()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT * FROM jobs WHERE request_id=?', (body['request_id'],)).fetchone()
            if existing:
                require(existing['source_sha'] == sha and existing['candidate'] == body['candidate_id'] and existing['timeout'] == timeout and existing['cell_timeout'] == cell_timeout, 'request_id already used with different content/options')
                require(existing['rerun_of'] == rerun_of, 'request_id already used for another submission')
                return self._job(existing)
            require(db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] < 32, 'Queue capacity reached')
            require(shutil.disk_usage(self.root).free > 2 * 1024**3, 'Queue storage low')
            jid = 'n_' + uuid.uuid4().hex
            directory = self.root / jid
            directory.mkdir(mode=0o700)
            (directory / 'submitted.ipynb').write_bytes(raw)
            (directory / 'input.ipynb').write_bytes(cleaned)
            db.execute('''INSERT INTO jobs
                (id,request_id,candidate,source_sha,execution_sha,state,created,timeout,cell_timeout,submitted_by)
                VALUES (?,?,?,?,?,'queued',?,?,?,?)''',
                (jid, body['request_id'], body['candidate_id'], sha, hashlib.sha256(cleaned).hexdigest(), time.time(), timeout, cell_timeout, subject))
            if rerun_of:
                db.execute('UPDATE jobs SET rerun_of=? WHERE id=?', (rerun_of, jid))
        return self.status(jid)

    def rerun(self, jid, body, subject):
        fields(body, {'request_id'})
        original = self.status(jid)
        require(original['state'] in TERMINAL, 'Wait for the original job to finish before rerunning')
        raw = (self.root / jid / 'submitted.ipynb').read_bytes()
        require(hashlib.sha256(raw).hexdigest() == original['source_sha'], 'Saved notebook hash mismatch')
        return self.submit({'candidate_id': original['candidate'], 'request_id': body['request_id'],
            'notebook_base64': base64.b64encode(raw).decode(), 'contains_secrets': False,
            'timeout_s': original['timeout'], 'cell_timeout_s': original['cell_timeout']},
            subject, rerun_of=jid)

    def claim(self, worker):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE state='running'").fetchone():
                return None  # No expired-lease retry: recovery first proves container shutdown.
            row = db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET state='running',started=?,heartbeat=?,worker=? WHERE id=?", (time.time(), time.time(), worker, row['id']))
        return self.status(row['id'])

    def owned(self, db, jid, worker):
        row = db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
        require(row is not None and row['state'] == 'running' and row['worker'] == worker, 'Worker does not own this running job')
        return row

    def heartbeat(self, jid, worker):
        with self.db() as db:
            row = self.owned(db, jid, worker)
            db.execute('UPDATE jobs SET heartbeat=? WHERE id=?', (time.time(), jid))
        return {'cancel_requested': bool(row['cancel_requested'])}

    def cancel(self, jid):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM jobs WHERE id=?', (jid,)).fetchone()
            require(row is not None, 'Unknown notebook job')
            if row['state'] == 'queued':
                db.execute("UPDATE jobs SET state='cancelled',finished=?,cancel_requested=1 WHERE id=?", (time.time(), jid))
            elif row['state'] == 'running':
                db.execute('UPDATE jobs SET cancel_requested=1 WHERE id=?', (jid,))
        return self.status(jid)

    def upload(self, jid, worker, raw):
        validate_outputs(raw)
        sha = hashlib.sha256(raw).hexdigest()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.owned(db, jid, worker)
            require(row['output_sha'] in (None, sha), 'Output already frozen with different content')
            target = self.root / jid / 'outputs.zip'
            temp = target.with_suffix('.tmp')
            temp.write_bytes(raw)
            os.replace(temp, target)
            db.execute('UPDATE jobs SET output_sha=? WHERE id=?', (sha, jid))
        return {'sha256': sha}

    def finish(self, jid, worker, result):
        fields(result, {'state', 'exit_code', 'elapsed_s', 'container_cleanup_confirmed', 'reason', 'image_id'}, {'error'})
        if result.get('error') is not None:
            require(valid_error(result['error']), 'Invalid structured failure')
            require(result['state'] != 'completed', 'A failed screen cannot be completed')
        require(result['state'] in TERMINAL, 'Invalid terminal state')
        require(result['container_cleanup_confirmed'] is True, 'Container shutdown must be verified')
        require(type(result['elapsed_s']) in (int, float) and 0 <= result['elapsed_s'] < 1000000, 'Invalid elapsed time')
        require(result['exit_code'] is None or type(result['exit_code']) is int, 'Invalid exit code')
        require(isinstance(result['reason'], str) and len(result['reason']) <= 500, 'Invalid reason')
        require(result['image_id'] == self.config['image_id'], 'Worker image differs from configured image')
        if result['state'] == 'completed':
            require(result['exit_code'] == 0, 'Successful execution requires exit zero')
        result = {**result, 'admission': 'UNKNOWN', 'score': None, 'scope': 'notebook_process_execution',
                  'notebook_outputs_are_untrusted': True}
        with self.db() as db:
            row = self.owned(db, jid, worker)
            require(row['output_sha'] is not None or result['state'] == 'interrupted', 'Upload evidence before finishing')
            db.execute('UPDATE jobs SET state=?,finished=?,result=? WHERE id=?', (result['state'], time.time(), json.dumps(result, allow_nan=False), jid))
        return self.status(jid)


def handle_get(handler, path):
    q = getattr(handler.server, 'notebooks', None)
    if q is None or not path.startswith('/v1/notebook'):
        return False
    match = re.fullmatch(r'/v1/notebooks/(' + JOB_PATTERN + r')(?:/(input|outputs))?', path)
    scope = 'work' if path == '/v1/notebook-worker/running' or match and match[2] == 'input' else 'read'
    if not handler._auth(scope):
        return True
    if path == '/v1/notebook-worker/running':
        with q.db() as db:
            jobs = [q._job(row) for row in db.execute("SELECT * FROM jobs WHERE state='running'")]
        handler._send(200, {'jobs': jobs})
    elif path == '/v1/notebook-runtime':
        handler._send(200, q.describe())
    elif path == '/v1/notebooks':
        handler._send(200, {'jobs': q.status()})
    elif match:
        job = q.status(match[1])
        if match[2]:
            name = 'input.ipynb' if match[2] == 'input' else 'outputs.zip'
            if name == 'outputs.zip' and not job['output_sha']:
                handler._send(404, {'error': 'Outputs not available yet'})
            else:
                handler._send(200, (q.root / job['id'] / name).read_bytes(), 'application/octet-stream', {
                    'Content-Disposition': 'attachment; filename="' + job['id'] + '-' + name + '"',
                    'X-Artifact-SHA256': job['execution_sha'] if name == 'input.ipynb' else job['output_sha']})
        else:
            handler._send(200, job)
    else:
        handler._send(404, {'error': 'Unknown notebook route'})
    return True


def handle_post(handler, path):
    q = getattr(handler.server, 'notebooks', None)
    if q is None or not path.startswith('/v1/notebook'):
        return False
    match = re.fullmatch(r'/v1/notebooks/(' + JOB_PATTERN + r')/(heartbeat|outputs|finish|cancel|rerun)', path)
    scope = 'submit' if path == '/v1/notebooks' or match and match[2] in ('cancel', 'rerun') else 'work'
    actor = handler._auth(scope)
    if not actor:
        return True
    if match and match[2] == 'outputs':
        require(handler.headers.get('Transfer-Encoding') is None, 'Chunked output uploads unsupported')
        require(handler.headers.get('Content-Type') == 'application/zip', 'ZIP content type required')
        length = int(handler.headers.get('Content-Length', '0'))
        require(0 < length <= MAX_OUTPUT, 'Output bundle exceeds fixed limit')
        raw = handler.rfile.read(length)
        require(len(raw) == length, 'Truncated output upload')
        handler._send(200, q.upload(match[1], actor[0], raw))
        return True
    body = handler._json()
    if body is None:
        return True
    if path == '/v1/notebooks':
        handler._send(201, q.submit(body, actor[0]))
    elif path == '/v1/notebook-worker/claim':
        fields(body, set())
        handler._send(200, {'job': q.claim(actor[0])})
    elif match:
        if match[2] == 'rerun':
            handler._send(201, q.rerun(match[1], body, actor[0]))
            return True
        elif match[2] == 'finish':
            result = q.finish(match[1], actor[0], body)
        else:
            fields(body, set())
            result = q.cancel(match[1]) if match[2] == 'cancel' else q.heartbeat(match[1], actor[0])
        handler._send(200, result)
    else:
        handler._send(404, {'error': 'Unknown notebook route'})
    return True
