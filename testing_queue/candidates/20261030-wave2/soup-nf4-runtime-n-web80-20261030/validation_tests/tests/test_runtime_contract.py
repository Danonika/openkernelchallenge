"""CPU-only metadata/tail/source/receipt tests; no CUDA query or model."""
import ast,contextlib,importlib.util,io,json,os,re,sys,tempfile
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TRITON_INTERPRET']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OLD=Path('/redacted/local/soup_web_wave2_20261030/nf4')
sys.path.insert(0,str(ROOT))
import torch
from solution import routes
from solution.packed_nf4 import raw_supported

def test_guards_and_native_wrappers_unchanged():
 def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
 a,b=functions(OLD/'solution/packed_nf4.py'),functions(ROOT/'solution/packed_nf4.py')
 for name in ('_nf4_value','_tensor','raw_supported','state_supported','decode'):assert a[name]==b[name]
 a,b=functions(OLD/'solution/routes.py'),functions(ROOT/'solution/routes.py')
 for name in ('forward_supported','forward_wrapper','functional_wrapper','install'):assert a[name]==b[name]
 assert (OLD/'solution/gpu_preflight.py').read_bytes()==(ROOT/'solution/gpu_preflight.py').read_bytes()

def test_lazy_negative_guards_and_plain_control():
 shape=(256,1536);n=shape[0]*shape[1];B=torch.zeros(n//2,1,dtype=torch.uint8)
 data=[torch.zeros(n//64,dtype=torch.uint8),torch.linspace(-1,1,256),torch.ones((n//64+255)//256),torch.tensor([.1])]
 with patch.object(torch.Tensor,'is_cuda',new=property(lambda _:True)),patch.object(torch.cuda,'get_device_capability',return_value=(7,5)):
  assert raw_supported(B,shape,*data,torch.float16)
 for index in range(5):
  tensors=[B,*data];tensors[index]=torch._neg_view(tensors[index])
  with patch.object(torch.Tensor,'is_cuda',new=property(lambda _:True)),patch.object(torch.cuda,'get_device_capability',side_effect=AssertionError('No CUDA query on lazy view')):
   assert not raw_supported(tensors[0],shape,*tensors[1:],torch.float16)

def test_direct_fd_geometric_metadata_receipts_bypass_stdout_proxy():
 class ForbiddenProxy:
  def write(self,*a):raise AssertionError('Python/Rich stdout proxy must not see own receipt')
  def flush(self):pass
 routes._COUNTS.update(forward=0,functional=0)
 with tempfile.TemporaryFile() as capture:
  sys.stdout.flush();saved=os.dup(1)
  try:
   os.dup2(capture.fileno(),1)
   with patch.object(sys,'stdout',ForbiddenProxy()):
    for route in ('forward','functional'):
     for _ in range(1025):routes._activated(route,(256,1536),torch.float16)
  finally:os.dup2(saved,1);os.close(saved)
  capture.seek(0);data=capture.read().decode()
 lines=[line for line in data.splitlines() if line]
 assert len(lines)==22
 records=[json.loads(line.removeprefix('SOUP_NF4_ROUTE ')) for line in lines]
 for route in ('forward','functional'):
  subset=[r for r in records if r['route']==route]
  assert [r['calls'] for r in subset]==[2**n for n in range(11)]
  assert all(r['candidate']=='nf4_runtime_n_v2' and r['kernel']=='soup_nf4_blockpair_runtime_n' and r['shape']==[256,1536] for r in subset)
 (ROOT/'validation/direct_fd_receipts.json').write_text(json.dumps({'status':'CPU_RECEIPT_PASS','attempted_calls':2050,'lines':len(lines),'exact_geometric_counts':True,'stdout_proxy_bypassed':True,'metadata_only':True,'gpu_executed':False},indent=2)+'\n')

def test_old_receipt_under_rich_live_is_wrapped():
 from rich.console import Console
 from rich.live import Live
 # Execute only the old Python telemetry function, not its module or any tensors.
 tree=ast.parse((OLD/'solution/routes.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_activated')
 scope={'json':json,'_COUNTS':{'forward':0,'functional':0}};exec(compile(ast.Module(body=[fn],type_ignores=[]),'old_telemetry','exec'),scope)
 capture=io.StringIO();console=Console(file=capture,force_terminal=True,width=80)
 with Live('fixture',console=console,auto_refresh=False):scope['_activated']('forward',(1536,1536),torch.float16)
 plain=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',capture.getvalue())
 lines=[line for line in plain.splitlines() if line.startswith('SOUP_NF4_ROUTE ')]
 assert len(lines)==1
 try:json.loads(lines[0].removeprefix('SOUP_NF4_ROUTE '))
 except json.JSONDecodeError:pass
 else:raise AssertionError('Expected default width80 Rich to wrap own long JSON receipt')
 (ROOT/'validation/rich_receipt_witness.json').write_text(json.dumps({'status':'CPU_RICH_WRAPPING_WITNESS','force_terminal':True,'width':80,'raw_output_after_ANSI_strip':plain,'single_line_JSON_parse':False,'does_not_measure_server_wall':True},indent=2)+'\n')
