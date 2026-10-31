#!/usr/bin/env python3
"""Submit, poll and download notebooks over verified HTTPS. Token stays in memory."""
import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Redirect refused; use the canonical HTTPS endpoint')


class Client:
    def __init__(self, url, token):
        parsed = urllib.parse.urlsplit(url.rstrip('/'))
        if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('A canonical HTTPS origin is required')
        if not re.fullmatch(r'[A-Za-z0-9_-]{32,256}', token):
            raise ValueError('A strong submitter token is required')
        self.url, self.token = url.rstrip('/'), token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, body=None, download=False):
        request = urllib.request.Request(self.url + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=90) as response:
            raw = response.read(64 * 1024**2 + 1)
            if len(raw) > 64 * 1024**2:
                raise ValueError('Response exceeded limit')
            if download:
                expected = response.headers.get('X-Artifact-SHA256')
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError('Output integrity check failed')
                return raw
        return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://35.200.1.242')
    commands = parser.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('submit')
    submit.add_argument('notebook')
    submit.add_argument('--candidate', default='agent-notebook')
    submit.add_argument('--request-id')
    submit.add_argument('--timeout', type=int, default=3600)
    submit.add_argument('--cell-timeout', type=int, default=1800)
    submit.add_argument('--wait', action='store_true')
    submit.add_argument('--out')
    for verb in ('status', 'download', 'cancel', 'wait'):
        command = commands.add_parser(verb)
        command.add_argument('job_id')
        if verb in ('download', 'wait'):
            command.add_argument('--out')
    commands.add_parser('runtime')
    commands.add_parser('list')
    args = parser.parse_args()
    token = os.environ.get('SOUP_NOTEBOOK_TOKEN') or getpass.getpass('Submitter token: ')
    client = Client(args.url, token)
    if args.command == 'runtime':
        print(json.dumps(client.request('GET', '/v1/notebook-runtime'), indent=2))
        return
    if args.command == 'list':
        print(json.dumps(client.request('GET', '/v1/notebooks'), indent=2))
        return
    if args.command == 'submit':
        raw = Path(args.notebook).read_bytes()
        if not 0 < len(raw) <= 8 * 1024**2:
            raise ValueError('Notebook must be at most 8 MiB')
        request_id = args.request_id or uuid.uuid4().hex
        print('request_id=' + request_id, flush=True)
        job = client.request('POST', '/v1/notebooks', {'candidate_id': args.candidate,
            'request_id': request_id, 'notebook_base64': base64.b64encode(raw).decode(),
            'contains_secrets': False, 'timeout_s': args.timeout,
            'cell_timeout_s': min(args.cell_timeout, args.timeout)})
        jid = job['id']
        print(json.dumps(job, indent=2), flush=True)
    else:
        jid = args.job_id
    if not re.fullmatch(r'n_[0-9a-f]{32}', jid):
        raise ValueError('Invalid job ID')
    path = '/v1/notebooks/' + jid
    if args.command == 'cancel':
        print(json.dumps(client.request('POST', path + '/cancel', {}), indent=2))
        return
    waiting = args.command == 'wait' or args.command == 'submit' and args.wait
    if waiting:
        previous = None
        while True:
            job = client.request('GET', path)
            if job['state'] != previous:
                print(jid + ': ' + job['state'], flush=True)
                previous = job['state']
            if job['state'] not in ('queued', 'running'):
                print(json.dumps(job, indent=2), flush=True)
                break
            time.sleep(5)
    if args.command == 'status':
        print(json.dumps(client.request('GET', path), indent=2))
    if args.command == 'download' or waiting and args.out:
        job = client.request('GET', path)
        if job['output_sha']:
            target = Path(args.out or jid + '-outputs.zip')
            raw = client.request('GET', path + '/outputs', download=True)
            with target.open('xb') as handle:
                handle.write(raw)
            print('Saved ' + str(target))
        else:
            print('No output archive is available for this job')
    if waiting and job['state'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as error:
        print('HTTP ' + str(error.code) + ': ' + error.read(2000).decode(errors='replace'), file=sys.stderr)
        sys.exit(2)
