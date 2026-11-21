"""Surgical PEFT integration; vendor forward is inspected, never bundled/credited.

Only a pinned vanilla scale/add pair is substituted in memory. The original
forward owns base NF4, clone, dropout, adapters, GEMMs, casts and dispatch.
"""
import ast
import functools
import hashlib
import importlib.metadata
import inspect
import os
import textwrap

import torch

from .epilogue import note, scale_add

EXPECTED_BNB_SHA256 = '96ed6fda63d7c67a4646faea8ba38483e8be8670059a3ac3dd740a99bdab6950'
_OWNER = None


def transform_forward(source):
    """Transform exactly one recognized pair; reject unexpected source layouts."""
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != 'forward' or function.decorator_list:
        raise RuntimeError('Unexpected PEFT forward definition')
    scaled, summed = [], []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == 'output' and ast.unparse(node.value) == 'lora_B(lora_A(dropout(x))) * scaling':
            scaled.append(node)
        if name == 'result' and ast.unparse(node.value) == 'result + output':
            summed.append(node)
    if len(scaled) != 1 or len(summed) != 1:
        raise RuntimeError('Pinned vanilla scale/add pair is not unique')
    scaled[0].value = scaled[0].value.left
    summed[0].value = ast.Call(
        func=ast.Name(id='_soup_original_scale_add', ctx=ast.Load()),
        args=[ast.Name(id=n, ctx=ast.Load()) for n in ('result', 'output', 'scaling')], keywords=[])
    return ast.fix_missing_locations(tree)


def install():
    global _OWNER
    if _OWNER is not None:
        cls, original, wrapped = _OWNER
        if cls.forward is not wrapped:
            raise RuntimeError('LoRA wrapper ownership changed')
        return
    if os.environ.get('TRITON_INTERPRET') not in (None, '', '0'):
        raise RuntimeError('Interpreter mode is diagnostic-only, never a target solution run')
    if importlib.metadata.version('peft') != '0.20.0':
        raise RuntimeError('LoRA candidate requires exact PEFT 0.20.0')
    from peft.tuners.lora import bnb as peft_bnb
    # Check only the installed library source, never model/data/results.
    module_source = inspect.getsource(peft_bnb)
    if hashlib.sha256(module_source.encode()).hexdigest() != EXPECTED_BNB_SHA256:
        raise RuntimeError('Pinned PEFT bnb.py bytes differ; no patch installed')
    cls = peft_bnb.Linear4bit
    original = cls.forward
    if original.__module__ != peft_bnb.__name__ or original.__qualname__ != 'Linear4bit.forward':
        raise RuntimeError('PEFT forward already patched by another component')
    tree = transform_forward(inspect.getsource(original))
    scope = dict(original.__globals__)
    scope['_soup_original_scale_add'] = scale_add
    exec(compile(tree, original.__code__.co_filename, 'exec'), scope)
    transformed = scope['forward']

    @functools.wraps(original)
    def wrapped(layer, x, *args, **kwargs):
        # The AST removes scaling before the vendor's optional output cast.
        # Activate ONLY under actual CUDA fp16 autocast, where that cast branch
        # is absent. All non-AMP and unrelated modules keep the native forward.
        enabled = (type(layer) is cls and type(x) is torch.Tensor and x.is_cuda
                   and torch.is_autocast_enabled('cuda')
                   and torch.get_autocast_dtype('cuda') == torch.float16
                   and not torch.is_inference_mode_enabled()
                   and not torch.compiler.is_compiling()
                   and not torch.jit.is_tracing() and not torch.jit.is_scripting()
                   and layer.training and torch.is_grad_enabled()
                   and not layer.disable_adapters and not layer.merged
                   and len(layer.active_adapters) == 1 and not layer.lora_variant)
        if enabled:
            adapter = layer.active_adapters[0]
            enabled = (adapter in layer.lora_A and adapter in layer.lora_B
                       and layer.r[adapter] == 16 and layer.lora_alpha[adapter] == 32
                       and layer.scaling[adapter] == 2.0
                       and layer.lora_A[adapter].weight.shape == (16, 1536)
                       and layer.lora_B[adapter].weight.shape in ((1536, 16), (256, 16))
                       and layer.lora_A[adapter].weight.dtype == torch.float32
                       and layer.lora_B[adapter].weight.dtype == torch.float32
                       and type(layer.lora_dropout[adapter]) is torch.nn.Dropout
                       and layer.lora_dropout[adapter].p == 0.05
                       and not layer.lora_dropout[adapter].inplace
                       and not args and not kwargs)
        if not enabled:
            note('native_forward_fallback')
            return original(layer, x, *args, **kwargs)
        return transformed(layer, x, *args, **kwargs)

    cls.forward = wrapped
    _OWNER = (cls, original, wrapped)
    note('installed', clone_retained=True, native_dropout_retained=True,
         kernel='lora_half_scale_add', own_share='UNMEASURED')


def uninstall():
    global _OWNER
    if _OWNER is None:
        return
    cls, original, wrapped = _OWNER
    if cls.forward is not wrapped:
        raise RuntimeError('Refusing to overwrite a foreign PEFT wrapper')
    cls.forward = original
    _OWNER = None
