"""Web JSON-file adapter for the unchanged original directory-output RoPE gate.

No kernel or arithmetic change. The original target checks run synchronously in
this short-lived child; a failure, incomplete result, or wrong status fails shut.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
import uuid


def _write(path, report):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)


def run(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    details = destination.with_name(destination.name + '.rotary_details')
    result = {'status': 'FAILED', 'gpu_executed': False, 'score': None,
              'admission': 'UNKNOWN', 'original_preflight_details': str(details)}
    _write(destination, result)
    previous_argv = sys.argv
    exit_code = None
    try:
        from . import gpu_preflight
        sys.argv = [gpu_preflight.__file__, '--out', str(details)]
        try:
            gpu_preflight.main()
            exit_code = 0
        except SystemExit as exc:
            exit_code = 0 if exc.code is None else exc.code
        report = json.loads((details / 'preflight.json').read_text())
        if exit_code != 0 or report.get('status') != 'PASSED_PREFLIGHT_ONLY' or report.get('gpu_executed') is not True:
            raise RuntimeError('Original target preflight failed or did not complete: ' + repr(exit_code))
        result = dict(report, original_preflight_details=str(details), original_exit_code=exit_code,
                      score=None, admission='UNKNOWN')
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                      original_exit_code=exit_code)
    finally:
        sys.argv = previous_argv
        _write(destination, result)
    return 0 if result['status'] == 'PASSED_PREFLIGHT_ONLY' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.out))


if __name__ == '__main__':
    main()
