"""CPU mocks and source-AST proof; no Torch/candidate import or GPU execution."""
import ast
import base64
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = Path('/workspace/shared/soup_web_wave2_20261030/head')
checks = []


def passed(name, **extra):
    checks.append({'test': name, 'pass': True, **extra})


with zipfile.ZipFile(BASE / 'candidate.zip') as z:
    original_files = {name: z.read(name) for name in z.namelist()}
with zipfile.ZipFile(ROOT / 'candidate.zip') as z:
    files = {name: z.read(name) for name in z.namelist()}
    assert len(z.namelist()) == len(files)
assert set(files) == set(original_files)
for name in files:
    if name != 'solution/native_head.py':
        assert files[name] == original_files[name]
    assert files[name] == (ROOT / name).read_bytes()
    compile(files[name], name, 'exec')
source = ast.parse(files['solution/native_head.py'])
old = ast.parse(original_files['solution/native_head.py'])


def without_telemetry(tree):
    return [ast.dump(n) for n in tree.body
            if not (isinstance(n, ast.ClassDef) and n.name == 'Telemetry')
            and not (isinstance(n, ast.Import) and len(n.names) == 1 and n.names[0].name == 'os')]


assert without_telemetry(source) == without_telemetry(old)
assert [ast.dump(n) for n in source.body if isinstance(n, ast.Import) and any(x.name == 'os' for x in n.names)] == [ast.dump(ast.parse('import os').body[0])]
passed('exact_nontelemetry_ast_identical_and_only_os_import_added')


def telemetry_class(tree, write=None, print_fn=None):
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Telemetry')
    scope = {'Counter': Counter, 'json': json, 'os': SimpleNamespace(write=write)}
    if print_fn is not None:
        scope['print'] = print_fn
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'isolated-Telemetry-only', 'exec'), scope)
    return scope['Telemetry']


written, reference = [], []


def writer(fd, raw):
    assert fd == 1 and isinstance(raw, bytes)
    written.append(raw)
    return len(raw)


def ref_print(*values, **kwargs):
    assert kwargs == {'flush': True}
    reference.append(' '.join(values))


Current = telemetry_class(source, writer)
Previous = telemetry_class(old, print_fn=ref_print)
current, previous = Current(), Previous()
hot = ['native_sum_forward_enter', 'native_sum_forward_return',
       'native_sum_backward_enter', 'native_sum_backward_return']
for event in ['installed', 'model_loss_attached']:
    current.emit(event); previous.emit(event)
for step in range(1, 65):
    for event in hot:
        kwargs = {'rows': 2048, 'vocabulary': 151936, 'dtype': 'torch.float16'} if event.endswith('forward_enter') else {}
        current.emit(event, **kwargs); previous.emit(event, **kwargs)
for reason in ['alternate_loss_arguments'] * 3 + ['dtype_layout_or_device'] * 2:
    current.emit('native_fallback', reason=reason)
    previous.emit('native_fallback', reason=reason)
current.emit('unused_output'); previous.emit('unused_output')
assert current.snapshot() == previous.snapshot()
records = [json.loads(line[len('SOUP_HEAD_NATIVE '):]) for line in b''.join(written).decode().splitlines() if line.startswith('SOUP_HEAD_NATIVE ')]
for event in hot:
    assert [r['count'] for r in records if r['event'] == event] == [1, 2, 4, 8, 16, 32, 64]
assert sum(r['event'] == 'native_fallback' for r in records) == 5
assert all(len(raw) < 4096 for raw in written)
assert sum(r['event'] in hot for r in records) == 28
assert len(reference) == 264 and len(records) == 36
passed('all_counters_and_failure_records_preserved_geometric_hotpath', hot_writes_before=256, hot_writes_after=28)

# Direct metadata writes do not alter the native stdout object or intercept its
# ordinary output. The mock sees fd1 writes while native print remains untouched.
native = io.StringIO()
with contextlib.redirect_stdout(native):
    owner = sys.stdout
    print('NATIVE SOUP OUTPUT BEFORE')
    Current().emit('installed')
    print('NATIVE SOUP OUTPUT AFTER')
    assert sys.stdout is owner
assert native.getvalue() == 'NATIVE SOUP OUTPUT BEFORE\nNATIVE SOUP OUTPUT AFTER\n'
passed('no_native_stdout_replacement_or_interception')

from rich.console import Console
from rich.file_proxy import FileProxy
rendered = io.StringIO()
console = Console(file=rendered, force_terminal=False, width=80)
proxy = FileProxy(console, io.StringIO())
before_count = len(written)
with contextlib.redirect_stdout(proxy):
    owner = sys.stdout
    print('native Soup before', flush=True)
    Current().emit('native_sum_backward_return', higher_order=False)
    print('native Soup after', flush=True)
    assert sys.stdout is owner
assert rendered.getvalue() == 'native Soup before\nnative Soup after\n'
assert len(written) == before_count + 1
assert written[-1].startswith(b'\nSOUP_HEAD_NATIVE ') and len(written[-1]) < 4096
passed('real_rich_fileproxy_only_own_compact_record_bypasses_rendering')

parts = []
def short_writer(fd, raw):
    assert fd == 1
    parts.append(raw[:7])
    return min(7, len(raw))
Short = telemetry_class(source, short_writer)
Short().emit('native_fallback', reason='fixture')
text = b''.join(parts).decode()
assert len([line for line in text.splitlines() if line.startswith('SOUP_HEAD_NATIVE ')]) == 1
assert json.loads(text.split('SOUP_HEAD_NATIVE ')[1])['reason'] == 'fixture'
def failed_writer(fd, raw):
    raise OSError('fixture write failure')
for fn in (failed_writer, lambda fd, raw: 0):
    Failing = telemetry_class(source, fn)
    try:
        Failing().emit('installed')
    except OSError:
        pass
    else:
        raise AssertionError('Metadata write failure was hidden')
silent = telemetry_class(source, failed_writer)(stream=False)
silent.emit('native_sum_forward_return')
assert silent.snapshot()['counts']['native_sum_forward_return'] == 1
passed('short_writes_completed_no_duplicates_errors_fail_closed_silent_counts')

manifest = json.loads((ROOT / 'source_manifest.json').read_text())
assert {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()} == manifest
nbpath = ROOT / '08_head_bounded_telemetry_v2.ipynb'
nb = json.loads(nbpath.read_text())
assert nbpath.stat().st_size < 8 * 1024**2
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    assert cell['outputs'] == [] and cell['execution_count'] is None
    code = ''.join(cell['source']); compile(code, f'cell{i}', 'exec')
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            key = getattr(node.targets[0], 'id', '')
            if key == 'PAYLOAD': payload = ast.literal_eval(node.value)
            if key == 'PAYLOAD_SHA256': hashes = ast.literal_eval(node.value)
for name, encoded in payload.items():
    raw = base64.b64decode(encoded, validate=True)
    assert raw == (ROOT / 'web_adapter' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hashes[name]
assert (ROOT / 'web_adapter/head_telemetry_v2.zip').read_bytes() == (ROOT / 'candidate.zip').read_bytes()
passed('notebook_zip_manifest_exact_embedded_payload_cleared_ast')

sys.path.insert(0, str(ROOT / 'web_adapter'))
import web_runner as w
base_runner = (BASE / 'web_adapter/web_runner.py').read_bytes()
local_tree=ast.parse((ROOT / 'web_adapter/web_runner.py').read_bytes())
base_tree=ast.parse(base_runner)
local_nodes={n.name:n for n in local_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
base_nodes={n.name:n for n in base_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(local_nodes)==set(base_nodes)
for name in local_nodes:
    if name!='normalize':assert ast.dump(local_nodes[name])==ast.dump(base_nodes[name]),name
assert (ROOT / 'web_adapter/evidence_reader.py').read_bytes()==Path('/workspace/shared/soup_web_ready_20261030_v2/support/evidence_reader.py').read_bytes()

for name in ['process_guard.py', 'owned_process_tree.py', 'runtime_probe.py', 'official_manifest.json', 'prepare_asset_manifest.py']:
    assert (ROOT / 'web_adapter' / name).read_bytes() == (BASE / 'web_adapter' / name).read_bytes()
assert w.NATIVE_TIMEOUT == 80 and w.FUSIONS['head_telemetry_v2']['requires_original_kernels'] is False
# Reuse only the earlier stdlib fixture constructor, not its runnable tests.
fixture_tree = ast.parse((BASE / 'test_preparation.py').read_text())
fixture_node = next(n for n in fixture_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fixture')
scope = {'json': json}
exec(compile(ast.Module(body=[fixture_node], type_ignores=[]), 'synthetic_fixture_only', 'exec'), scope)
data, logs = scope['fixture']()
proc = {'exit_code': 0, 'timed_out': False, 'interrupted': False, 'cleanup_confirmed': True}
for side in ('solution_1', 'solution_2'):
    logs[side] = b''.join(written).decode()
assert w.normalize(data, 'head_telemetry_v2', proc, logs)['complete_fixed_task']
for invalid in ('1', '32', '63', '65', '"64"', 'null'):
    altered = {name: value.replace('"count":64', '"count":' + invalid) for name, value in logs.items()}
    result = w.normalize(data, 'head_telemetry_v2', proc, altered)
    assert not result['complete_fixed_task'] and result['complete_measurements'] is None
    assert result['score'] is None
assert 'PAIRED_COMPLETE_RULES_FAILED' in base_runner.decode()
passed('same_accepted_runner_cap80_count64_route_and_negative_controls')

from evidence_reader import events
actual_path = Path('/workspace/shared/soup_head_research_20261030/evidence/fusion_b_solution_1.log')
actual_bytes = actual_path.read_bytes()
assert hashlib.sha256(actual_bytes).hexdigest() == '9ea6cae9ba02b55f58bbd806bb8b217bd10a6bc6d5f734bd996618f857654fb3'
actual = actual_bytes.decode(errors='replace')
head_events = list(events(actual, 'SOUP_HEAD_NATIVE '))
assert len(head_events) == 214
assert max(r['count'] for r in head_events if r['event'] == 'native_sum_backward_return') == 53
bad = w.normalize(data, 'head_telemetry_v2', proc,
                  {'solution_1': actual, 'solution_2': actual})
assert not bad['complete_fixed_task'] and bad['complete_measurements'] is None
# Actual count53 cannot become a complete count64 run even when the remaining
# fixture fields deliberately describe synthetic complete training.
for event in hot:
    wrapped = '\x1b[2K\x1b[1ASOUP_HEAD_NATIVE {"event":\n"' + event + '","count":64}\n'
    assert list(events(wrapped, 'SOUP_HEAD_NATIVE ')) == [{'event': event, 'count': 64}]
wrapped_log = '\n'.join('\x1b[2K\x1b[1ASOUP_HEAD_NATIVE {"event":\n"' + event + '","count":64}' for event in hot)
assert w.normalize(data, 'head_telemetry_v2', proc,
                   {'solution_1': wrapped_log, 'solution_2': wrapped_log})['complete_fixed_task']
for record in ('{"event":"native_sum_backward_return","count":64,"count":64}',
               '{"event":"native_sum_backward_return","count":NaN}',
               '{"event":"native_sum_backward_return","count":64',
               '{"event":"native_sum_backward_return","count":64} garbage'):
    assert list(events('SOUP_HEAD_NATIVE ' + record, 'SOUP_HEAD_NATIVE ')) == []
passed('actual_source_bound_rich_log_and_synthetic_wrapped_json_fail_closed',
       actual_head_records=214, actual_backward_return_count=53)

assert 'torch' not in sys.modules and 'triton' not in sys.modules
report = {'status': 'CPU_MOCK_SOURCE_AST_PASS', 'checks': checks,
          'notebook_sha256': hashlib.sha256(nbpath.read_bytes()).hexdigest(),
          'source_zip_sha256': hashlib.sha256((ROOT / 'candidate.zip').read_bytes()).hexdigest(),
          'arithmetic_source_ast_identical': True, 'native_soup_stdout_untouched': True,
          'gpu_executed': False, 'candidate_module_imported': False,
          'performance_benefit': 'HYPOTHESIS_NOT_MEASURED', 'admission': 'BLOCKED_ZERO_OWNED_TRITON'}
(ROOT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
