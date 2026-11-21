"""CPU source-function witness, not an import of the full target PEFT package.

Execute exact downloaded PEFT0.20.0 methods with nn.Module/BaseTunerLayer marker
shims. Only FP16/BF16/FP32 adapters and ModuleDict paths are claimed; no CUDA,
quantization, real PEFT constructor or target integration is exercised here.
"""
import ast
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reference/tuners_utils.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='2f8f343737348014b71d775dda15a06dbe34df4190ed20afba8ad58ba471a5cf'
tree=ast.parse(SOURCE.read_text())
base=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseTunerLayer')
names=('_get_base_layer_device_and_dtype','_move_adapter_to_device_of_base_layer')
nodes=[n for n in base.body if isinstance(n,ast.FunctionDef) and n.name in names]
nodes.append(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cast_adapter_dtype'))
class BaseTunerLayer:pass
class BufferDict(nn.Module):pass  # unreachable in these ModuleDict-only cases
scope={'torch':torch,'nn':nn,'BaseTunerLayer':BaseTunerLayer,'BufferDict':BufferDict,'UPCAST_DTYPES':()}
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(SOURCE),'exec'),scope)
cast_adapter_dtype=scope['cast_adapter_dtype']
class Layer(nn.Module,BaseTunerLayer):
    adapter_layer_names=('lora_A','lora_B');other_param_names=()
    _get_base_layer_device_and_dtype=scope['_get_base_layer_device_and_dtype']
    _move_adapter_to_device_of_base_layer=scope['_move_adapter_to_device_of_base_layer']
    def __init__(self,compute_dtype):
        super().__init__();self.base_layer=nn.Module()
        self.base_layer.weight=nn.Parameter(torch.arange(32,dtype=torch.uint8),requires_grad=False)
        self.base_layer.compute_dtype=compute_dtype
        self.lora_A=nn.ModuleDict({'default':nn.Linear(8,4,bias=False),'inactive':nn.Linear(8,4,bias=False).half()})
        self.lora_B=nn.ModuleDict({'default':nn.Linear(4,8,bias=False),'inactive':nn.Linear(4,8,bias=False).half()})
    def get_base_layer(self):return self.base_layer
class TestNativeAdapterUpcast(unittest.TestCase):
    def test_native_move_then_native_upcast(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            with self.subTest(dtype=dtype):
                layer=Layer(dtype)
                with patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA forbidden')):
                    layer._move_adapter_to_device_of_base_layer('default')
                    self.assertEqual(layer.lora_A['default'].weight.dtype,dtype)
                    values={n:p.detach().clone() for n,p in layer.named_parameters()}
                    identities={n:id(p) for n,p in layer.named_parameters()};rng=torch.get_rng_state().clone()
                    cast_adapter_dtype(layer,adapter_name='default',autocast_adapter_dtype=True)
                    self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                    for n,p in layer.named_parameters():
                        self.assertEqual(id(p),identities[n])
                        expected=values[n].float() if '.default.' in n else values[n]
                        self.assertEqual(p.dtype,expected.dtype);self.assertTrue(torch.equal(p,expected))
                    self.assertEqual(layer.lora_A['default'].weight.dtype,torch.float32)
                    self.assertEqual(layer.lora_B['default'].weight.dtype,torch.float32)
    def test_opt_out_retains_half(self):
        layer=Layer(torch.float16);layer._move_adapter_to_device_of_base_layer('default')
        cast_adapter_dtype(layer,adapter_name='default',autocast_adapter_dtype=False)
        self.assertEqual(layer.lora_A['default'].weight.dtype,torch.float16)
    def test_fixture_calls_native_helper_before_unchanged_assertions(self):
        source=(ROOT/'solution/gpu_preflight.py').read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='cast_adapter_dtype']
        self.assertEqual(len(calls),1);call=calls[0]
        self.assertEqual(ast.unparse(call),'cast_adapter_dtype(layer, adapter_name=\'default\', autocast_adapter_dtype=True)')
        checks=[n for n in ast.walk(tree) if isinstance(n,ast.Assert) and '.weight.dtype == torch.float32' in ast.unparse(n)]
        self.assertEqual(len(checks),2);self.assertTrue(all(n.lineno>call.lineno for n in checks))
if __name__=='__main__':unittest.main(verbosity=2)
