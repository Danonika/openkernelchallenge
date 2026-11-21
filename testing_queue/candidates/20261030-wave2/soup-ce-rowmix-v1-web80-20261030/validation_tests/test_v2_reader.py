"""V2 evidence-reader fixtures only; no notebook or training execution."""
import copy,json,runpy
from pathlib import Path
R=Path(__file__).resolve().parent
ns=runpy.run_path(str(R/'test_web_packet.py'));w=ns['w'];base,logs=ns['fixture']();proc=ns['process']
from evidence_reader import events
prefix='SOUP_CE_V3_EVENT '
valid='\x1b[2K\r'+prefix+'{\n"event":"triton_forward",\n"own_forward_routes":64\n}\n'
assert list(events(valid,prefix))[0]['own_forward_routes']==64
invalid=[prefix+'{"event":"triton_forward","own_forward_\nroutes":64}',
 prefix+'{"event":"triton_forward","own_forward_routes":64,"own_forward_routes":1}',
 prefix+'{"event":"triton_forward","own_forward_routes":NaN}',
 prefix+'{"event":"triton_forward","own_forward_routes":64']
for s in invalid:assert list(events(s,prefix))==[]
wrapped={name:'\n'.join('\x1b[2K\r'+prefix+json.dumps(json.loads(line[len(prefix):]),indent=2) for line in text.splitlines()) for name,text in logs.items()}
assert w.normalize(base,'ce_rowmix_v1',proc,wrapped)['complete_fixed_task']
for count in (0,-1,True,'64',None):
 altered=dict(wrapped);altered['solution_1']=prefix+json.dumps({'event':'triton_forward','own_forward_routes':count})
 assert not w.normalize(base,'ce_rowmix_v1',proc,altered)['complete_fixed_task']
censored=dict(proc,timed_out=True)
assert not w.normalize(base,'ce_rowmix_v1',censored,wrapped)['complete_fixed_task']
report={'status':'V2_READER_PROFILE_CPU_PASS','valid_wrapped_ANSI_evidence':True,'inside_string_wrap_rejected':True,'duplicate_nonfinite_truncated_rejected':True,'bad_positive_fields_rejected':5,'censored_remains_incomplete':True,'gpu_executed':False}
(R/'validation/v2_reader_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
