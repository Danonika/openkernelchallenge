from pathlib import Path
import base64,hashlib,json
ROOT=Path(__file__).resolve().parent
raw=(ROOT/'first_batch_probe_v2.py').read_bytes();compile(raw,'first_batch_probe_v2.py','exec')
digest=hashlib.sha256(raw).hexdigest()
code="import base64,hashlib,os,runpy,tempfile\nfrom pathlib import Path\n"
code+="assert os.environ.get('SOUP_REPO') and os.environ.get('SOUP_RESULTS')\n"
code+='raw=base64.b64decode('+repr(base64.b64encode(raw).decode())+',validate=True)\n'
code+='assert hashlib.sha256(raw).hexdigest()=='+repr(digest)+'\n'
code+="stage=Path(tempfile.mkdtemp(prefix='onebit-first-loss-',dir='/kaggle/working'))\n(stage/'probe.py').write_bytes(raw)\nrunpy.run_path(str(stage/'probe.py'),run_name='__main__')\n"
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
 {'cell_type':'markdown','id':'scope','metadata':{},'source':'# One-bit first-native-batch sensitivity\n\nStandalone diagnostic, stopped before backward and optimizer update. Native Soup task/loader/collator/loss; same first batch and replayed RNG for NF4 and selective/all binary projections. No complete training, eval, speed, memory, admission or target claim. The fixed task file and harness are not edited. Expanded binary weights are numerical references only; all precomputation occurs in the diagnostic.\n'},
 {'cell_type':'code','id':'probe','metadata':{},'execution_count':None,'outputs':[],'source':code}]}
path=ROOT/'Soup_OneBit_First_Batch_Probe_v2.ipynb';path.write_text(json.dumps(nb,indent=1)+'\n')
report={'notebook':path.name,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'source_sha256':digest,'scope':'FIRST_BATCH_FORWARD_DIAGNOSTIC_NOT_BENCHMARK','status':'PREPARED_NOT_RUN'}
(ROOT/'first_batch_v2_build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
