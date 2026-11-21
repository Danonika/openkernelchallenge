"""Fresh native W16 materialization per call; metadata views, never a weight cache."""
import hashlib,inspect
import torch
from bitsandbytes import functional as F

EXPECTED={
    'dequantize_4bit':('bitsandbytes.functional','dequantize_4bit','c928daa1321ac80321093537a64d2339751690b5e53b96aa4b302898197d9a15'),
    'quantstate_init':('bitsandbytes.functional','QuantState.__init__','7fb835bb9d8e708a4c2407f006e57e03dea9b4978e2f3778121572b2ab3eae26'),
}


def verify_materialize_owners():
    for key,value in (('dequantize_4bit',F.dequantize_4bit),('quantstate_init',F.QuantState.__init__)):
        module,qualname,digest=EXPECTED[key]
        if (not inspect.isfunction(value) or hasattr(value,'__wrapped__') or (value.__module__,value.__qualname__)!=(module,qualname)
                or hashlib.sha256(inspect.getsource(value).encode()).hexdigest()!=digest):
            raise RuntimeError('Native materialization source/owner changed: '+key)


def materialize_native_w16(A,B,shapeB,absmax,blocksize,quant_type,absmax_8bit,absmax_code,absmax_offset):
    # Reconstruct only native metadata containers; tensors/storage remain native.
    secondary=F.QuantState(absmax=absmax,code=absmax_code,blocksize=256,dtype=torch.float32)
    state=F.QuantState(absmax=absmax_8bit,shape=shapeB,blocksize=blocksize,
                       quant_type=quant_type,dtype=A.dtype,offset=absmax_offset,state2=secondary)
    weight=F.dequantize_4bit(B.view(-1,1),quant_state=state)
    if (type(weight) is not torch.Tensor or weight.dtype!=torch.float16 or tuple(weight.shape)!=tuple(shapeB)
            or not weight.is_contiguous() or weight.data_ptr()%4):
        raise RuntimeError('Unexpected native materialized W16 contract')
    return weight
