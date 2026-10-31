import base64
import hashlib
import http.client
import io
import json
import os
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from api import LocalServer
from notebook_queue import NotebookQueue, clean_notebook, validate_outputs
from notebook_worker import Worker
from soup_queue import QueueError

TOKENS = {role: 'TEST_ONLY_NOTEBOOK_' + role.upper() + '_' + 'x' * 32 for role in ('submitter', 'reviewer', 'worker', 'viewer')}
CONFIG = {'image_id': 'sha256:' + 'a' * 64, 'host_memory_gib': 29.37, 'python': '3.13.15', 'pytorch': '2.11.0+cu128', 'triton': '3.6.0'}


def notebook(source='print(42)'):
    return json.dumps({'nbformat': 4, 'cells': [{'cell_type': 'code', 'source': source,
        'outputs': [{'output_type': 'stream', 'text': 'stale'}], 'execution_count': 99,
        'metadata': {'tags': ['skip-execution', 'raises-exception']}}], 'metadata': {'kernelspec': {'name': 'untrusted'}}}).encode()


def submission(request_id='test-request', source='print(42)'):
    return {'candidate_id': 'test-candidate', 'request_id': request_id, 'contains_secrets': False,
            'notebook_base64': base64.b64encode(notebook(source)).decode()}


def outputs(extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('execution.log', 'test')
        archive.writestr('runner.json', '{}')
        if extra:
            archive.writestr(extra, 'candidate data')
    return stream.getvalue()


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue = NotebookQueue(self.root, CONFIG)

    def tearDown(self):
        self.temp.cleanup()

    def test_magics_shell_allowed_and_stale_execution_metadata_removed(self):
        value = json.loads(clean_notebook(notebook('%time print(42)\n!pwd')))
        self.assertIn('!pwd', value['cells'][0]['source'])
        self.assertEqual(value['cells'][0]['outputs'], [])
        self.assertEqual(value['cells'][0]['metadata'], {})
        self.assertIsNone(value['cells'][0]['execution_count'])
        self.assertEqual(value['metadata']['kernelspec']['name'], 'python3')
        with self.assertRaises(QueueError):
            clean_notebook(b'{"nbformat":4,"cells":[]}')

    def test_automatic_idempotent_submission_survives_restart(self):
        first = self.queue.submit(submission(), 'api-submitter')
        self.assertEqual(first['state'], 'queued')
        self.assertEqual(first['id'], NotebookQueue(self.root, CONFIG).submit(submission(), 'api-submitter')['id'])
        with self.assertRaises(QueueError):
            self.queue.submit(submission(source='different()'), 'api-submitter')
        with self.assertRaises(QueueError):
            self.queue.submit({**submission(), 'contains_secrets': True}, 'api-submitter')

    def test_serial_claim_cancel_and_no_expired_lease_rerun(self):
        first = self.queue.submit(submission(), 'api-submitter')
        second = self.queue.submit(submission('second'), 'api-submitter')
        self.assertEqual(self.queue.claim('api-worker')['id'], first['id'])
        self.assertIsNone(self.queue.claim('second-worker'))
        self.assertEqual(self.queue.cancel(second['id'])['state'], 'cancelled')
        self.queue.cancel(first['id'])
        self.assertTrue(self.queue.heartbeat(first['id'], 'api-worker')['cancel_requested'])
        with self.assertRaises(QueueError):
            self.queue.heartbeat(first['id'], 'other-worker')

    def test_completion_requires_output_verified_cleanup_and_zero_exit(self):
        job = self.queue.submit(submission(), 'api-submitter')
        self.queue.claim('api-worker')
        result = {'state': 'completed', 'exit_code': 0, 'elapsed_s': 1.0,
            'container_cleanup_confirmed': True, 'reason': 'Container exited', 'image_id': CONFIG['image_id']}
        with self.assertRaises(QueueError):
            self.queue.finish(job['id'], 'api-worker', result)
        raw = outputs()
        self.queue.upload(job['id'], 'api-worker', raw)
        self.assertEqual(self.queue.upload(job['id'], 'api-worker', raw)['sha256'], hashlib.sha256(raw).hexdigest())
        for override in ({'container_cleanup_confirmed': False}, {'exit_code': 1}, {'score': 100}):
            with self.assertRaises(QueueError):
                self.queue.finish(job['id'], 'api-worker', {**result, **override})
        done = self.queue.finish(job['id'], 'api-worker', result)
        self.assertEqual(done['state'], 'completed')
        self.assertIsNone(done['result']['score'])
        self.assertEqual(done['result']['admission'], 'UNKNOWN')

    def test_output_archive_validation(self):
        validate_outputs(outputs('results/result.json'))
        for name in ('../secrets', 'results/../../secrets', '/etc/passwd', 'results\\evil', 'runner.json'):
            with self.assertRaises(QueueError):
                validate_outputs(outputs(name))

    def test_collector_rejects_symlink_ancestors_symlinks_and_fifos(self):
        run = self.root / 'run'
        delivery = run / 'output' / 'delivery'
        delivery.mkdir(parents=True)
        (delivery / 'executed.ipynb').write_bytes(notebook())
        secret = self.root / 'host-data'
        (secret / 'results').mkdir(parents=True)
        (secret / 'results' / 'secret.txt').write_text('DO_NOT_COLLECT')
        (run / 'output' / 'work').symlink_to(secret, target_is_directory=True)
        worker = Worker({'image_id': CONFIG['image_id'], 'queue_url': 'http://127.0.0.1:8765', 'workspace_root': str(self.root / 'worker')}, TOKENS['worker'])
        raw = worker.collect(run, b'log', {'state': 'failed'})
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertNotIn('results/secret.txt', archive.namelist())
        (run / 'output' / 'work').unlink()
        results = run / 'output' / 'work' / 'results'
        results.mkdir(parents=True)
        (results / 'link').symlink_to(secret / 'results' / 'secret.txt')
        os.mkfifo(results / 'fifo')
        (results / 'normal.json').write_text('{}')
        with zipfile.ZipFile(io.BytesIO(worker.collect(run, b'log', {}))) as archive:
            self.assertIn('results/normal.json', archive.namelist())
            self.assertNotIn('results/link', archive.namelist())
            self.assertNotIn('results/fifo', archive.namelist())

    def test_container_resource_and_mount_policy(self):
        worker = Worker({'image_id': CONFIG['image_id'], 'queue_url': 'http://127.0.0.1:8765',
            'workspace_root': str(self.root / 'worker'), 'repo': '/public/repo', 'model_cache': '/public/model'}, TOKENS['worker'])
        argv = worker.argv('soupnb-test', self.root, {'cell_timeout': 10})
        for flag in ('--network=none', '--memory=30g', '--memory-swap=30g', '--cpuset-cpus=0,1,2,3', '--gpus=device=0', '--read-only', '--user=10001:10001'):
            self.assertIn(flag, argv)
        text = ' '.join(argv)
        self.assertNotIn(TOKENS['worker'], text)
        self.assertNotIn('docker.sock', text)
        self.assertIn('dst=/task,readonly', text)
        self.assertIn('--shm-size=30g', argv)
        self.assertFalse(any(arg.startswith('--tmpfs=') for arg in argv))
        self.assertIn('type=bind,src=' + str(self.root / 'output' / 'tmp') + ',dst=/tmp', argv)


class NotebookApiTests(unittest.TestCase):
    def test_rerun_auth_original_preserved_and_retry_idempotent(self):
        q = self.server.notebooks
        original = q.submit({**submission(), 'timeout_s':123,'cell_timeout_s':45}, 'api-submitter')
        route = '/v1/notebooks/' + original['id'] + '/rerun'
        self.assertEqual(self.request('POST',route,'viewer',{'request_id':'again'})[0],403)
        with self.assertRaises(QueueError):
            q.rerun(original['id'],{'request_id':'again'},'api-submitter')
        q.cancel(original['id'])
        status, raw, _ = self.request('POST',route,'submitter',{'request_id':'again'})
        created = json.loads(raw)
        self.assertEqual(status,201)
        self.assertNotEqual(created['id'],original['id'])
        self.assertEqual(created['rerun_of'],original['id'])
        for key in ('candidate','source_sha','timeout','cell_timeout'):
            self.assertEqual(created[key],original[key])
        self.assertEqual(q.status(original['id'])['state'],'cancelled')
        self.assertEqual(q.rerun(original['id'],{'request_id':'again'},'api-submitter')['id'],created['id'])
        self.assertEqual(len(q.status()),2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = LocalServer(Path(self.temp.name) / 'state', TOKENS, notebook_config=CONFIG)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, role=None, body=None, binary=False):
        headers = {}
        if role:
            headers['Authorization'] = 'Bearer ' + TOKENS[role]
        if body is not None:
            headers['Content-Type'] = 'application/zip' if binary else 'application/json'
            body = body if binary else json.dumps(body)
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        data, status, response_headers = response.read(), response.status, dict(response.getheaders())
        conn.close()
        return status, data, response_headers

    def test_role_boundaries_and_automatic_worker_claim(self):
        self.assertEqual(self.request('GET', '/v1/notebook-runtime')[0], 401)
        self.assertEqual(self.request('POST', '/v1/notebooks', 'viewer', submission())[0], 403)
        status, raw, _ = self.request('POST', '/v1/notebooks', 'submitter', submission())
        self.assertEqual(status, 201)
        job = json.loads(raw)
        self.assertEqual(job['state'], 'queued')
        self.assertEqual(self.request('POST', '/v1/notebook-worker/claim', 'submitter', {})[0], 403)
        claimed = json.loads(self.request('POST', '/v1/notebook-worker/claim', 'worker', {})[1])['job']
        self.assertEqual(claimed['id'], job['id'])
        path = '/v1/notebooks/' + job['id']
        self.assertEqual(self.request('GET', path + '/input', 'submitter')[0], 403)
        self.assertEqual(self.request('GET', path + '/input', 'worker')[0], 200)
        self.assertEqual(self.request('POST', path + '/outputs', 'submitter', outputs(), True)[0], 403)
        self.assertEqual(self.request('POST', path + '/outputs', 'worker', outputs(), True)[0], 200)
        status, raw, headers = self.request('GET', path + '/outputs', 'submitter')
        self.assertEqual(status, 200)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), headers['X-Artifact-SHA256'])
        self.assertIn('attachment;', headers['Content-Disposition'])
        self.assertEqual(self.request('GET', '/v1/notebook-worker/running', 'submitter')[0], 403)

    def test_page_has_no_data_or_persistent_credentials(self):
        status, html, headers = self.request('GET', '/notebooks')
        self.assertEqual(status, 200)
        self.assertIn(b'GPU notebook runner', html)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        js = self.request('GET', '/notebooks.js')[1].decode()
        for dangerous in ('innerHTML', 'localStorage', 'sessionStorage', 'document.cookie'):
            self.assertNotIn(dangerous, js)


if __name__ == '__main__':
    unittest.main(verbosity=2)
