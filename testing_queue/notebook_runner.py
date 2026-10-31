"""Runs only INSIDE the isolated candidate container; outputs are untrusted evidence."""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError
from notebook_failfast import FAILURE_PATH, classify_exception, error as failure, write_json

OUTPUT = Path('/output')
DELIVERY = OUTPUT / 'delivery'
WORK = Path('/kaggle/working')


def write_file(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)
        os.fchmod(handle.fileno(), 0o644)


def main():
    os.umask(0o022)
    DELIVERY.mkdir(exist_ok=True)
    (WORK / 'results').mkdir(exist_ok=True)
    shutil.copytree('/task', WORK / 'repo')
    nb = nbformat.read('/input/input.ipynb', as_version=4)
    started = time.monotonic()
    status, code, error = 'completed', 0, None

    def checkpoint(**kwargs):
        data = nbformat.writes(nb).encode()
        if len(data) > 32 * 1024**2:
            raise RuntimeError('Executed notebook exceeds 32 MiB output limit')
        write_file(DELIVERY / 'executed.ipynb', data)

    def after_cell(**kwargs):
        checkpoint()
        if FAILURE_PATH.exists():
            detail = json.loads(FAILURE_PATH.read_text())
            raise RuntimeError(detail.get('code', 'EXECUTION_ERROR') + ': ' +
                               detail.get('message', 'Screening failed'))

    checkpoint()
    client = NotebookClient(nb, kernel_name='python3', timeout=int(sys.argv[1]),
        resources={'metadata': {'path': str(WORK)}}, allow_errors=False,
        force_raise_errors=True, record_timing=True, on_cell_executed=after_cell)
    try:
        client.execute()
    except CellTimeoutError as exc:
        status, code, error = 'timed_out', 124, str(exc)[:2000]
        write_json(FAILURE_PATH, failure('TIME_LIMIT_EXCEEDED',
            'Notebook cell deadline exceeded', 'cell', limit=int(sys.argv[1]), unit='seconds'), first=True)
        traceback.print_exc()
    except BaseException as exc:
        status, code, error = 'failed', 1, type(exc).__name__ + ': ' + str(exc)[:2000]
        write_json(FAILURE_PATH, classify_exception(type(exc).__name__ + ': ' + str(exc)), first=True)
        traceback.print_exc()
    finally:
        try:
            checkpoint()
        except Exception as exc:
            status, code, error = 'failed', 1, 'Cannot save notebook: ' + str(exc)[:500]
        write_file(DELIVERY / 'runner.json', json.dumps({
            'status': status, 'exit_code': code, 'elapsed_s': time.monotonic() - started,
            'error': error, 'executed_code_cells': sum(c.get('execution_count') is not None for c in nb.cells if c.cell_type == 'code'),
            'admission': 'UNKNOWN', 'score': None, 'scope': 'untrusted_notebook_runner_output'
        }, allow_nan=False).encode())
    return code


if __name__ == '__main__':
    sys.exit(main())
