"""Proposed next-version log reader; does not alter any run/completion decision.

Accept a structured JSON object only after an exact line-anchored event prefix,
ignoring terminal ANSI escape formatting and JSON whitespace line wrapping.
Malformed/truncated/oversize/duplicate-key/nonfinite objects fail closed.
"""
import json
import re

# CSI, OSC (BEL or ST termination), and standard two-character ANSI escapes.
_ANSI = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]')

def _reject(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def events(text, prefix, *, max_event_chars=32768):
    """Yield unambiguous dict events; never infer full-run or CUDA completion."""
    if not isinstance(prefix, str) or not prefix or '\n' in prefix or '\r' in prefix:
        raise ValueError('A nonempty single-line exact prefix is required')
    clean = _ANSI.sub('', text).replace('\r\n', '\n').replace('\r', '\n')
    decoder = json.JSONDecoder(object_pairs_hook=_unique_pairs, parse_constant=_reject)
    for match in re.finditer(r'(?m)^[ \t]*' + re.escape(prefix), clean):
        fragment = clean[match.end():match.end() + max_event_chars].lstrip()
        try:
            record, end = decoder.raw_decode(fragment)
        except (ValueError, TypeError, RecursionError):
            continue
        if not isinstance(record, dict):
            continue
        if fragment[end:].split('\n', 1)[0].strip():
            continue
        yield record
