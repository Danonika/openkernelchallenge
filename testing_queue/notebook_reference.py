"""Operator-selected public baseline, pinned to runtime, task, data and harness."""
import hashlib
import json
from pathlib import Path


def load_reference(path=None):
    try:
        return json.loads(Path(path or Path(__file__).with_name('public-baseline.json')).read_text())
    except (OSError, ValueError):
        return {}


def matching_reference(reference, fingerprint, image_id=None, eval_hash=None):
    return bool(reference and fingerprint == reference.get('task_fingerprint') and
                (image_id is None or image_id == reference.get('image_id')) and
                (eval_hash is None or eval_hash == reference.get('eval_data_sha256')))


def harness_matches(reference, repo):
    try:
        return bool(reference.get('harness_files')) and all(
            hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest() == sha
            for name, sha in reference['harness_files'].items())
    except OSError:
        return False
