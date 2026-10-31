#!/usr/bin/env python3
"""Read ready Google Sheet rows and submit immutable notebooks to the local queue.

Reads the To Run tab and mirrors job results into dedicated VM result tabs.
Private receipts are retained in state/outbox for retry and recovery. Notebook
code is never executed by this service; the existing GPU worker executes it.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SHEET_URL = 'https://example.invalid/redacted'
HEADERS = ['Request ID', 'Candidate ID', 'Notebook URL', 'Notebook JSON', 'Ready', 'Timeout (s)', 'Cell timeout (s)', 'Notes']
MAX_CSV, MAX_NOTEBOOK = 16 * 1024**2, 8 * 1024**2
ID_PATTERN = r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}'
TERMINAL = {'completed', 'failed', 'timed_out', 'interrupted', 'cancelled', 'rejected'}


class InvalidInput(ValueError):
    pass


class RemoteError(Exception):
    def __init__(self, message, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def require(condition, message):
    if not condition:
        raise InvalidInput(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reserved_id(value):
    return value in ('sheet-status', 'publisher-status') or value.startswith('invalid-row-') or value.endswith(('.conflict', '.rejected'))


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def public_url(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise InvalidInput('Invalid public HTTPS URL') from exc
    require(parsed.scheme == 'https' and parsed.hostname and parsed.username is None and
            parsed.password is None and port in (None, 443) and not parsed.fragment,
            'A public HTTPS URL without credentials is required')
    return parsed


class DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = public_url(newurl)
        require(target.hostname == 'drive.usercontent.google.com' or
                target.hostname.endswith('.googleusercontent.com'), 'Download redirect host is not allowed')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RemoteError('Local API redirect refused', False)


def download(url, limit):
    # No credentials, cookies, or inherited proxy credentials are sent externally.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), DownloadRedirect())
    try:
        with opener.open(url, timeout=20) as response:
            raw = response.read(limit + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RemoteError('Public download unavailable') from exc
    require(0 < len(raw) <= limit, 'Download is empty or exceeds its size limit')
    return raw


def sheet_url(value):
    url = public_url(value)
    query = urllib.parse.parse_qs(url.query, keep_blank_values=True)
    require(url.hostname == 'docs.google.com' and re.fullmatch(r'/spreadsheets/d/[A-Za-z0-9_-]+/export', url.path)
            and query.get('format') == ['csv'] and len(query.get('gid', [])) == 1
            and query['gid'][0].isdigit()
            and set(query) in ({'format', 'gid'}, {'format', 'gid', 'range'})
            and ('range' not in query or query['range'] == ['A1:H1000']),
            'Expected the configured Google Sheet CSV export URL with input range A1:H1000')
    # Result columns may contain large JSON reports. Always export inputs only.
    return urllib.parse.urlunsplit((url.scheme, url.netloc, url.path,
        urllib.parse.urlencode({'format': 'csv', 'gid': query['gid'][0], 'range': 'A1:H1000'}), ''))


def notebook_url(value):
    require(len(value) <= 2048, 'Notebook URL is too long')
    url = public_url(value)
    if url.hostname == 'drive.google.com':
        query = urllib.parse.parse_qs(url.query)
        match = re.fullmatch(r'/file/d/([A-Za-z0-9_-]+)/view', url.path)
        file_id = match[1] if match else None
        if url.path in ('/uc', '/open'):
            ids = query.get('id', [])
            file_id = ids[0] if len(ids) == 1 and re.fullmatch(r'[A-Za-z0-9_-]+', ids[0]) else None
        require(file_id is not None, 'Expected a public Google Drive file link')
        parameters = {'id': file_id, 'export': 'download', 'confirm': 't'}
        if 'resourcekey' in query:
            keys = query['resourcekey']
            require(len(keys) == 1 and re.fullmatch(r'[A-Za-z0-9_-]{1,256}', keys[0]), 'Invalid public Drive resource key')
            parameters['resourcekey'] = keys[0]
        return 'https://drive.usercontent.google.com/download?' + urllib.parse.urlencode(parameters)
    require(url.hostname in ('github.com', 'raw.githubusercontent.com') and not url.query, 'Notebook URL must be public Google Drive or GitHub raw/blob')
    parts = urllib.parse.unquote(url.path).split('/')[1:]
    require(all(p and p not in ('.', '..') and not re.search(r'[\\\x00-\x20]', p) for p in parts), 'Invalid GitHub notebook path')
    if url.hostname == 'github.com':
        require(len(parts) >= 5 and parts[2] == 'blob', 'Expected a GitHub blob notebook link')
        parts.pop(2)
    require(len(parts) >= 4 and parts[-1].endswith('.ipynb'), 'Notebook URL must identify an ipynb file')
    return 'https://raw.githubusercontent.com/' + '/'.join(urllib.parse.quote(p, safe='-._~') for p in parts)


def parse_sheet(raw):
    require(len(raw) <= MAX_CSV, 'Sheet exceeds 16 MiB')
    csv.field_size_limit(MAX_CSV)
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) and
            set(HEADERS) <= set(reader.fieldnames), 'Sheet headers do not match the queue template')
    rows = []
    for number, row in enumerate(reader, 2):
        require(len(rows) < 1000 and None not in row, 'Sheet exceeds 1000 rows or contains malformed CSV')
        rows.append((number, row))
    return rows


def specification(row):
    value = {key: (row.get(key) or '').strip() for key in HEADERS}
    require(re.fullmatch(ID_PATTERN, value['Request ID']), 'Request ID must be 1..80 letters, digits, dots, underscores or hyphens, starting with a letter/digit')
    require(not reserved_id(value['Request ID']), 'Request ID is reserved for queue status; choose another ID')
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value['Candidate ID']), 'Candidate ID must be a short identifier')
    require(bool(value['Notebook URL']) != bool(value['Notebook JSON']), 'Provide exactly one Notebook URL or Notebook JSON')
    timeout_text, cell_text = value['Timeout (s)'], value['Cell timeout (s)']
    require((not timeout_text or re.fullmatch(r'[0-9]{1,5}', timeout_text)) and
            (not cell_text or re.fullmatch(r'[0-9]{1,5}', cell_text)), 'Timeouts must be whole seconds')
    timeout = int(timeout_text) if timeout_text else 3600
    cell_timeout = int(cell_text) if cell_text else min(1800, timeout)
    require(10 <= timeout <= 14400 and 1 <= cell_timeout <= timeout, 'Invalid notebook or cell timeout')
    require(len(value['Notebook JSON']) <= 50000, 'Inline notebook exceeds the 50000-character cell limit')
    return {'request_id': value['Request ID'], 'candidate_id': value['Candidate ID'],
            'url': notebook_url(value['Notebook URL']) if value['Notebook URL'] else None,
            'inline': value['Notebook JSON'] or None, 'timeout_s': timeout, 'cell_timeout_s': cell_timeout}


def validate_notebook(raw):
    require(0 < len(raw) <= MAX_NOTEBOOK, 'Notebook exceeds 8 MiB or is empty')
    try:
        notebook = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise InvalidInput('Source is not notebook JSON; check public sharing/download access') from exc
    require(isinstance(notebook, dict) and notebook.get('nbformat') == 4, 'Notebook v4 is required')
    cells = notebook.get('cells')
    require(isinstance(cells, list) and 0 < len(cells) <= 512, 'Expected 1..512 notebook cells')
    for cell in cells:
        require(isinstance(cell, dict) and cell.get('cell_type') in ('code', 'markdown', 'raw'), 'Invalid notebook cell')
        source = cell.get('source', '')
        require(isinstance(source, str) or isinstance(source, list) and all(isinstance(x, str) for x in source), 'Notebook cell source must be text')
    require(any(cell['cell_type'] == 'code' for cell in cells), 'Notebook needs a code cell')
    return raw


def read_token(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    require(len(lines) == 1, 'Expected one submitter credential assignment')
    parts = lines[0].split('=', 1)
    require(len(parts) == 2 and re.fullmatch(r'[A-Z][A-Z0-9_]*', parts[0]) and re.fullmatch(r'[A-Za-z0-9_-]{32,256}', parts[1]), 'Invalid submitter credential file')
    return parts[1]


class SoupAPI:
    def __init__(self, token):
        self.token = token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method, path, body=None):
        require(path == '/v1/notebooks' or re.fullmatch(r'/v1/notebooks/n_[0-9a-f]{32}', path), 'Invalid local API route')
        req = urllib.request.Request('http://127.0.0.1:8765' + path, method=method,
            data=encoded(body) if body is not None else None,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(req, timeout=30) as response:
                raw = response.read(MAX_CSV + 1)
            if len(raw) > MAX_CSV:
                raise RemoteError('Local API response too large')
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode(errors='replace')
            retry = exc.code >= 500 or exc.code == 429 or 'Queue capacity reached' in detail or 'Queue storage low' in detail
            raise RemoteError('Local API rejected request (HTTP ' + str(exc.code) + ')', retry) from exc
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise RemoteError('Local API temporarily unavailable') from exc


class Worker:
    def __init__(self, state_dir, api, url=SHEET_URL, fetch=download, publisher=None):
        self.root, self.api, self.url, self.fetch = Path(state_dir), api, sheet_url(url), fetch
        self.publisher = publisher
        for path in (self.root, self.root / 'inputs', self.root / 'outbox'):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'ledger.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, spec TEXT NOT NULL, source_sha TEXT NOT NULL, state TEXT NOT NULL, job_id TEXT, job_json TEXT, error TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS rejections (fingerprint TEXT PRIMARY KEY, row_number INTEGER, request_id TEXT, error TEXT)')
        self.db.commit()
        os.chmod(self.root / 'ledger.sqlite3', 0o600)

    def close(self):
        self.db.close()

    def write(self, name, value):
        target = self.root / 'outbox' / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_bytes(encoded(value))
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)

    def reject(self, number, row, message):
        fingerprint = digest(encoded(row))
        rid = (row.get('Request ID') or '').strip()
        safe_id = rid if re.fullmatch(ID_PATTERN, rid) and not reserved_id(rid) else None
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO rejections VALUES (?,?,?,?)', (fingerprint, number, safe_id, message))
        self.write((safe_id + '.rejected' if safe_id else 'invalid-row-' + str(number)) + '.json',
            {'state': 'rejected', 'row': number, 'request_id': safe_id,
             'candidate_id': (row.get('Candidate ID') or '').strip(), 'error': message})

    def accept(self, number, row):
        spec = specification(row)
        rid, fingerprint = spec['request_id'], digest(encoded(spec))
        old = self.db.execute('SELECT * FROM entries WHERE id=?', (rid,)).fetchone()
        if old:
            if old['fingerprint'] != fingerprint:
                self.write(rid + '.conflict.json', {'state': 'rejected', 'error': 'REQUEST_ID_CONFLICT: accepted inputs are immutable; use a new Request ID',
                    'request_id': rid, 'original_source_sha256': old['source_sha'], 'job_id': old['job_id'], 'row': number})
            return False
        if self.db.execute("SELECT count(*) FROM entries WHERE state NOT IN ('completed','failed','timed_out','interrupted','cancelled','rejected')").fetchone()[0] >= 32:
            return False  # Leave excess ready rows on the Sheet until space opens.
        raw = validate_notebook(spec['inline'].encode() if spec['inline'] is not None else self.fetch(spec['url'], MAX_NOTEBOOK))
        target = self.root / 'inputs' / (rid + '.ipynb')
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            os.chmod(temporary, 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # Durable snapshot and ledger precede POST. A lost response retries the
        # identical sheet-<id> request, which the API handles idempotently.
        with self.db:
            self.db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)',
                (rid, fingerprint, encoded(spec).decode(), digest(raw), 'pending', None, None, None))
        self.receipt(rid)
        return True

    def receipt(self, rid):
        row = dict(self.db.execute('SELECT * FROM entries WHERE id=?', (rid,)).fetchone())
        self.write(rid + '.json', {'schema_version': 1, 'transport': 'google-sheet', 'request_id': rid,
            'api_request_id': 'sheet-' + rid, 'source_sha256': row['source_sha'], 'state': row['state'],
            'candidate_id': json.loads(row['spec'])['candidate_id'],
            'job_id': row['job_id'], 'error': row['error'], 'job': json.loads(row['job_json']) if row['job_json'] else None})

    def publish_results(self):
        if self.publisher is None:
            return 'disabled'
        try:
            from sheets_results import project_receipts
            receipts = []
            for path in sorted((self.root / 'outbox').glob('*.json')):
                if path.name == 'publisher-status.json':
                    continue
                value = json.loads(path.read_text())
                value['receipt_name'] = path.name
                receipts.append(value)
            result = self.publisher.publish(project_receipts(receipts))
            if result.get('state') == 'retry':
                self.write('publisher-status.json', {'state': 'retrying', 'error': result.get('error')})
                return 'retrying'
            self.write('publisher-status.json', {'state': 'synchronized'})
            return 'synchronized'
        except Exception:
            # Credential/API trouble cannot stop GPU intake or lose results.
            # Exception text may contain Google request/credential details.
            self.write('publisher-status.json', {'state': 'retrying',
                'error': 'Google Sheets result write failed; stored results will retry automatically'})
            return 'retrying'

    def save_job(self, row, job):
        require(isinstance(job, dict) and isinstance(job.get('id'), str) and re.fullmatch(r'n_[0-9a-f]{32}', job['id']) and
                job.get('source_sha') == row['source_sha'] and job.get('request_id') == 'sheet-' + row['id'] and
                job.get('state') in TERMINAL | {'queued', 'running'}, 'Local API returned mismatched job evidence')
        with self.db:
            self.db.execute('UPDATE entries SET state=?,job_id=?,job_json=?,error=NULL WHERE id=?',
                (job['state'], job['id'], encoded(job).decode(), row['id']))
        self.receipt(row['id'])

    def advance(self, submit=False):
        count = 0
        for row in self.db.execute('SELECT * FROM entries ORDER BY rowid').fetchall():
            if row['state'] in TERMINAL or (row['job_id'] is None) != submit:
                continue
            try:
                if row['job_id']:
                    job = self.api.request('GET', '/v1/notebooks/' + row['job_id'])
                else:
                    spec = json.loads(row['spec'])
                    raw = (self.root / 'inputs' / (row['id'] + '.ipynb')).read_bytes()
                    require(digest(raw) == row['source_sha'], 'SOURCE_INTEGRITY_ERROR: immutable snapshot changed')
                    job = self.api.request('POST', '/v1/notebooks', {'candidate_id': spec['candidate_id'],
                        'request_id': 'sheet-' + row['id'], 'notebook_base64': base64.b64encode(raw).decode(),
                        'contains_secrets': False, 'timeout_s': spec['timeout_s'], 'cell_timeout_s': spec['cell_timeout_s']})
                self.save_job(row, job)
                count += 1
            except (RemoteError, InvalidInput, OSError) as exc:
                transient = isinstance(exc, RemoteError) and exc.retryable
                with self.db:
                    self.db.execute('UPDATE entries SET state=?,error=? WHERE id=?',
                        (row['state'] if transient else 'rejected', str(exc) if isinstance(exc, (RemoteError, InvalidInput)) else 'Snapshot unavailable', row['id']))
                self.receipt(row['id'])
                if submit and transient:
                    break  # Queue capacity/network failure applies to subsequent submissions too.
        return count

    def tick(self):
        counts = {'rows': 0, 'accepted': 0, 'submitted': 0, 'errors': 0}
        self.advance()  # Existing jobs progress even if the Sheet is unavailable.
        try:
            rows = parse_sheet(self.fetch(self.url, MAX_CSV))
            ready = [(n, r) for n, r in rows if (r.get('Ready') or '').strip().lower() == 'true']
            ids = [(r.get('Request ID') or '').strip() for _, r in ready]
            duplicates = {rid for rid in ids if ids.count(rid) > 1}
            counts['rows'] = len(rows)
            # Notices describe the current input rows. Once an input is fixed or
            # removed, an old rejection must not contradict its accepted job.
            # The audit table retains history; immutable job receipts stay intact.
            for notice in (self.root / 'outbox').glob('*.json'):
                if notice.name.endswith(('.rejected.json', '.conflict.json')) or notice.name.startswith('invalid-row-'):
                    notice.unlink()
            self.write('sheet-status.json', {'state': 'available', 'rows': len(rows), 'ready': len(ready), 'checked_at': int(time.time() // 60) * 60})
            for number, row in ready:
                try:
                    require((row.get('Request ID') or '').strip() not in duplicates, 'Duplicate ready Request ID in Sheet')
                    if counts['accepted'] < 32:
                        counts['accepted'] += int(self.accept(number, row))
                except (InvalidInput, RemoteError) as exc:
                    self.reject(number, row, str(exc))
                    counts['errors'] += 1
        except (InvalidInput, RemoteError, ValueError, csv.Error) as exc:
            self.write('sheet-status.json', {'state': 'unavailable', 'checked_at': int(time.time() // 60) * 60,
                'error': str(exc) if isinstance(exc, (InvalidInput, RemoteError)) else 'Invalid Sheet CSV'})
            counts['errors'] += 1
        counts['submitted'] = self.advance(submit=True)
        counts['results'] = self.publish_results()
        return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soup-token-file', required=True)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--sheet-url', default=SHEET_URL)
    parser.add_argument('--google-credentials-file', help='Optional service-account JSON, never mounted into notebooks')
    parser.add_argument('--google-access-token-file', help='Short-lived Sheets-only token supplied through systemd credentials')
    parser.add_argument('--google-metadata-auth', action='store_true', help='Use the attached VM writer identity; no private key')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require(5 <= args.poll_seconds <= 3600, 'Polling interval must be 5..3600 seconds')
    os.umask(0o077)
    Path(args.state_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (Path(args.state_dir) / 'worker.lock').open('a')
    if os.name == 'posix':
        import fcntl
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise SystemExit('Another Sheet queue worker is already running')
    publisher = None
    require(sum(bool(v) for v in (args.google_credentials_file, args.google_metadata_auth, args.google_access_token_file)) <= 1, 'Choose only one Google authentication mode')
    if args.google_credentials_file or args.google_metadata_auth or args.google_access_token_file:
        from sheets_result_publisher import LazyGoogleSheetsAPI, SheetsResultPublisher
        publisher = SheetsResultPublisher(LazyGoogleSheetsAPI(args.google_credentials_file, access_token_file=args.google_access_token_file), args.state_dir)
    worker = Worker(args.state_dir, SoupAPI(read_token(args.soup_token_file)), args.sheet_url, publisher=publisher)
    try:
        while True:
            print(json.dumps(worker.tick()), flush=True)
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        worker.close()
        lock.close()


if __name__ == '__main__':
    main()
