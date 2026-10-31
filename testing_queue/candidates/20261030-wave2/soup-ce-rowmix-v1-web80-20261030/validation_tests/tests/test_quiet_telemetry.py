"""CPU-only logging protocol fixtures; no model, CUDA, task execution or timing."""
import contextlib
import io
import json
from unittest.mock import patch
import solution
from solution import telemetry as t

class Proxy(io.StringIO):
    def fileno(self):return 1

def records(data):
    return [json.loads(line[len(t.RouteStats.PREFIX):]) for line in data.splitlines()
            if line.startswith(t.RouteStats.PREFIX)]

def test_geometric_hotpath_all_counts_all_fallbacks_and_final_summary():
    stats=t.RouteStats();stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        for i in range(65):
            stats.record('loss_boundary',training=True)
            stats.record('triton_forward',rows=356)
            stats.record('triton_backward',phase='enqueued_not_completion')
        for event in sorted(t.RouteStats.EMIT-t.RouteStats.HOTPATH):
            for i in range(3):stats.record(event,reason=f'reason{i}')
        stats.record('triton_prepass_started');stats.record('triton_prepass_started')
        for i in range(65):stats.record('candidate_loss_call')
    rows=records(stream.getvalue())
    for event in t.RouteStats.HOTPATH:
        assert stats.events[event]==65
        assert [x['event_count'] for x in rows if x['event']==event]==[1,2,4,8,16,32,64]
    for event in t.RouteStats.EMIT-t.RouteStats.HOTPATH:
        assert [x['event_count'] for x in rows if x['event']==event]==[1,2,3]
    assert sum(x['event']=='triton_prepass_started' for x in rows)==1
    assert stats.events['triton_prepass_started']==2 and stats.events['candidate_loss_call']==65
    assert stats.sequence==37
    with patch.object(solution,'_stats',stats),contextlib.redirect_stdout(io.StringIO()) as final:
        solution._report()
    summary=json.loads(final.getvalue().split('SOUP_CE_V3_SUMMARY ',1)[1])
    assert summary['events']==dict(stats.events)
    assert summary['telemetry']['emitted_records']==37
    assert not summary['telemetry']['kernel_completion_or_share_claim']

def test_direct_write_one_atomic_sized_ascii_record_and_native_stdout_untouched():
    proxy=Proxy();calls=[];stats=t.RouteStats()
    def write(fd,data):
        assert fd==1;calls.append(data);return len(data)
    with patch.object(t.sys,'stdout',proxy),patch.object(t.os,'fpathconf',return_value=4096),patch.object(t.os,'write',side_effect=write):
        before=t.sys.stdout
        print('native Soup before',flush=True)
        stats.record('loss_boundary',name='caf\u00e9',shape=[4,512,151936],training=True)
        print('native Soup after',flush=True)
        assert t.sys.stdout is before
    assert proxy.getvalue()=='native Soup before\nnative Soup after\n'
    assert len(calls)==1 and len(calls[0])<=4096
    assert calls[0].startswith(b'\nSOUP_CE_V3_EVENT ') and calls[0].endswith(b'\n')
    assert records(calls[0].decode('ascii'))[0]['metadata']['name']=='caf\u00e9'

def test_short_write_completes_suffix_no_duplicate():
    proxy=Proxy();writes=[]
    def write(fd,data):
        size=min(11,len(data));writes.append(data[:size]);return size
    with patch.object(t.sys,'stdout',proxy),patch.object(t.os,'fpathconf',return_value=4096),patch.object(t.os,'write',side_effect=write):
        t.emit_line('SOUP_CE_V3_EVENT {"event":"fixture"}')
    assert b''.join(writes)==b'\nSOUP_CE_V3_EVENT {"event":"fixture"}\n'
    assert not proxy.getvalue()

def test_partial_then_ioerror_fallback_only_unwritten_suffix():
    proxy=Proxy();written=[]
    def write(fd,data):
        if written:raise OSError('fixture fd failure')
        written.append(data[:9]);return 9
    with patch.object(t.sys,'stdout',proxy),patch.object(t.os,'fpathconf',return_value=4096),patch.object(t.os,'write',side_effect=write):
        t.emit_line('SOUP_CE_V3_EVENT {"event":"fixture"}')
    combined=b''.join(written)+proxy.getvalue().encode('ascii')
    assert combined==b'\nSOUP_CE_V3_EVENT {"event":"fixture"}\n'
    assert len(records(combined.decode('ascii')))==1

def test_unavailable_fd_and_pipe_bound_and_write_failure_fallback():
    for failure in ('no_fileno','fpathconf','too_long','write','zero_write'):
        proxy=io.StringIO() if failure=='no_fileno' else Proxy()
        bound=1 if failure=='too_long' else 4096
        fpathconf=patch.object(t.os,'fpathconf',side_effect=OSError('fixture unavailable')) if failure=='fpathconf' else patch.object(t.os,'fpathconf',return_value=bound)
        effect=OSError('fixture write failure') if failure=='write' else (lambda *args:0) if failure=='zero_write' else AssertionError('unexpected direct write')
        with patch.object(t.sys,'stdout',proxy),fpathconf,patch.object(t.os,'write',side_effect=effect):
            t.emit_line('SOUP_CE_V3_EVENT {"event":"fixture"}')
        assert proxy.getvalue()=='\nSOUP_CE_V3_EVENT {"event":"fixture"}\n'

def test_no_benchmark_special_case_and_same_event_fields():
    import ast
    from pathlib import Path
    source=Path(t.__file__).read_text()
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Name) and n.id in ('torch','triton','time') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('environ','getenv','cuda') for n in ast.walk(tree))
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        stats=t.RouteStats();stats.record('triton_backward',phase='enqueued_not_completion')
    record=records(stream.getvalue())[0]
    assert set(record)=={'schema','sequence','event','event_count','loss_calls','own_forward_routes','own_backward_enqueues','native_adapter_routes','native_core_routes','native_backward_routes','metadata'}
    assert record['own_backward_enqueues']==1 and record['metadata']['phase']=='enqueued_not_completion'


def test_real_rich_fileproxy_only_own_record_bypasses_console():
    from rich.console import Console
    from rich.file_proxy import FileProxy
    rendered=io.StringIO();console=Console(file=rendered,force_terminal=False,width=80)
    proxy=FileProxy(console,Proxy());calls=[]
    def write(fd,data):calls.append(data);return len(data)
    with patch.object(t.sys,'stdout',proxy),patch.object(t.os,'fpathconf',return_value=4096),patch.object(t.os,'write',side_effect=write):
        print('native Soup before',flush=True)
        t.RouteStats().record('triton_forward',rows=356)
        print('native Soup after',flush=True)
    assert rendered.getvalue()=='native Soup before\nnative Soup after\n'
    assert len(calls)==1 and records(calls[0].decode('ascii'))[0]['event']=='triton_forward'


def test_real_rich_live_child_preserves_native_prints_and_single_line_records():
    import os
    from pathlib import Path
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    code='''import sys
from rich.console import Console
from rich.live import Live
from solution.telemetry import RouteStats
console=Console(file=sys.stdout,force_terminal=True,width=40)
with Live('NATIVE PANEL',console=console,auto_refresh=False,redirect_stdout=True,redirect_stderr=False):
    print('NATIVE_SOUP_BEFORE',flush=True)
    RouteStats().record('loss_boundary',denominator_type='Tensor',logits_shape=[4,235,151936],training=True)
    print('NATIVE_SOUP_AFTER',flush=True)
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=root,
                          env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES=''),
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=10,text=True)
    lines=[line for line in result.stdout.splitlines() if line.startswith(t.RouteStats.PREFIX)]
    assert len(lines)==1
    row=json.loads(lines[0][len(t.RouteStats.PREFIX):])
    assert row['metadata']['denominator_type']=='Tensor'
    assert row['metadata']['logits_shape']==[4,235,151936]
    assert result.stdout.count('NATIVE_SOUP_BEFORE')==1
    assert result.stdout.count('NATIVE_SOUP_AFTER')==1


def test_mixed_counts_accumulate_all_calls_with_only_geometric_emission():
    stream=io.StringIO();stats=t.RouteStats()
    with contextlib.redirect_stdout(stream):
        for i in range(5):
            stats.record('triton_forward',route='mixed_rows',native_supervised_rows=2,
                         safe_supervised_rows=5,ignored_rows=5)
    rows=records(stream.getvalue())
    assert [r['event_count'] for r in rows]==[1,2,4]
    assert rows[-1]['metadata']['cumulative_mixed_rows']==dict(calls=4,native_supervised_rows=8,safe_supervised_rows=20,ignored_rows=20)
    assert stats.snapshot()['mixed_row_totals']==dict(calls=5,native_supervised_rows=10,safe_supervised_rows=25,ignored_rows=25)
    assert stats.sequence==3
