"""Original CPU-only tiny attention routing probe; no pretrained/full-model load."""
import hashlib, importlib.util, inspect, json
from pathlib import Path
import torch
import transformers
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2 import modeling_qwen2 as qmod

torch.set_num_threads(2)
kernels_present=importlib.util.find_spec('kernels') is not None
if kernels_present: raise RuntimeError('Route probe requires no hub-kernel runtime; stop instead of enabling it')
source=Path(inspect.getfile(qmod)).read_bytes()
blob=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
config=Qwen2Config(hidden_size=8,num_attention_heads=2,num_key_value_heads=1,intermediate_size=16,num_hidden_layers=1,attention_dropout=0.,rope_parameters={'rope_type':'default','rope_theta':1000000.})
config._attn_implementation='eager'
module=qmod.Qwen2Attention(config,0)
x=torch.randn(1,3,8,requires_grad=True)
cos=torch.ones(1,3,4);sin=torch.zeros_like(cos)
orig=qmod.apply_rotary_pos_emb
calls=[]
def observed(*args,**kwargs):
 calls.append({'q_shape':list(args[0].shape),'k_shape':list(args[1].shape),'q_stride':list(args[0].stride()),'k_stride':list(args[1].stride()),'dtype':str(args[0].dtype)})
 return orig(*args,**kwargs)
qmod.apply_rotary_pos_emb=observed
try:
 y,_=module(x,(cos,sin),None,past_key_values=None)
 y.sum().backward()
finally:
 qmod.apply_rotary_pos_emb=orig
assert len(calls)==1
r=dict(scope='Tiny CPU Qwen2Attention only; eager attention, no hub runtime/full model/pretrained download',torch=torch.__version__,torch_git=torch.version.git_version,transformers=transformers.__version__,installed_qwen2_git_blob_sha=blob,expected_source_blob_sha='b4fed98fafae8e4ee39be3b8cd766b1c85b079cd',source_blob_matches=blob=='b4fed98fafae8e4ee39be3b8cd766b1c85b079cd',kernels_package_present=kernels_present,module_global_patch_called=len(calls),calls=calls,output_shape=list(y.shape),finite_input_grad=bool(torch.isfinite(x.grad).all()),gpu=False,performance_measured=False)
Path('/workspace/shared/soup_cycle/rope_cycle3_route_probe.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
