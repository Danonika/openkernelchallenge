#!/usr/bin/env python3
"""Refuse active work, then quiesce the queue under its shared GPU lock."""
import argparse
import os
import sqlite3
import subprocess
from pathlib import Path

SERVICES = ['soup-sheets-queue', 'soup-api', 'soup-notebook-worker']
DB = Path('/var/lib/soup-queue/notebooks/notebooks.sqlite3')
LOCK = Path('/tmp/soup-queue-gpu0.lock')


def has_work():
    with sqlite3.connect('file:' + str(DB) + '?mode=ro', uri=True) as db:
        return bool(db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running') LIMIT 1").fetchone())


def quiesce(run, busy, check_only=False):
    if busy():
        raise RuntimeError('Queue has queued/running jobs. Nothing was stopped; retry after they finish.')
    if check_only:
        return
    try:
        run(['systemctl', 'stop', *SERVICES[:2]], check=True)
        if busy():
            raise RuntimeError('A request arrived during preflight. Services restored; retry after it finishes.')
        run(['systemctl', 'stop', SERVICES[2]], check=True)

        run(['systemd-run', '--unit=soup-keyless-recover-' + str(os.getpid()),
             '--on-active=5m', '/bin/systemctl', 'start', *SERVICES], check=True)
    except Exception:
        run(['systemctl', 'start', *SERVICES], check=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    import fcntl
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('GPU is busy. Nothing was stopped; retry after the job finishes.')
        quiesce(subprocess.run, has_work, args.check)
    print('Queue is idle.' if args.check else 'Queue quiesced safely for keyless configuration.')


if __name__ == '__main__':
    main()

