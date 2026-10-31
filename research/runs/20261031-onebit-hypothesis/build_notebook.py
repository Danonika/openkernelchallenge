from pathlib import Path
import base64,hashlib,json

ROOT=Path(__file__).resolve().parent
payload={n:(ROOT/n).read_bytes() for n in ('codec.py','gpu_weight_probe.py')}
for n,raw in payload.items():compile(raw,n,'exec')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
code="import base64,hashlib,json,os,runpy,sys,tempfile\nfrom pathlib import Path\n"
code+="assert os.environ.get('SOUP_RESULTS') and os.environ.get('SOUP_REPO')\n"
code+="os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
code+='PAYLOAD='+repr({n:base64.b64encode(v).decode() for n,v in payload.items()})+'\n'
code+='HASHES='+repr({n:sha(v) for n,v in payload.items()})+'\n'
code+="stage=Path(tempfile.mkdtemp(prefix='onebit-weight-probe-',dir='/kaggle/working'))\n"
code+="for name,value in PAYLOAD.items():\n raw=base64.b64decode(value,validate=True)\n assert hashlib.sha256(raw).hexdigest()==HASHES[name]\n (stage/name).write_bytes(raw)\n"
code+="sys.path.insert(0,str(stage))\nrunpy.run_path(str(stage/'gpu_weight_probe.py'),run_name='__main__')\n"
compile(code,'notebook','exec')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
 {'cell_type':'markdown','id':'scope','metadata':{},'source':'# Selective one-bit weight research\n\nRead-only component study on the prepared public Qwen checkpoint. No task, harness, model checkpoint or other job is modified. No training, admission, speed, peak-memory or 45-second/1359-MiB claim. Tests a sign/scale/exception storage format against native NF4 and FP16 values. All precomputation happens in this diagnostic; no model tensors or caches are exported.\n'},
 {'cell_type':'code','id':'probe','metadata':{},'execution_count':None,'outputs':[],'source':code}]}
path=ROOT/'Soup_OneBit_Weight_Probe.ipynb';path.write_text(json.dumps(nb,indent=1)+'\n')
report={'notebook':path.name,'notebook_sha256':sha(path.read_bytes()),'source_sha256':{n:sha(v) for n,v in payload.items()},'scope':'COMPONENT_PROBE_NOT_BENCHMARK','status':'PREPARED_NOT_RUN'}
(ROOT/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
