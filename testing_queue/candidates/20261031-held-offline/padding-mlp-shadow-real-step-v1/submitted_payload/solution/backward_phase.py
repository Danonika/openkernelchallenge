"""Explicit after-forward ownership phase for frozen-MLP adjoints only.

The source-pinned dense forward must have already finished. Only exact native
MatMul4BitBackward context objects bound by the producer observer can compact;
every other context calls the captured native backward. No global row mask.
"""
import hashlib,inspect
from collections import Counter
from .row_adjoint import FrozenLinearTicket,conditional_adjoint

NATIVE_BACKWARD_SHA='053e8719319a526c61f288a37221ba5200de9eaa20dae7c7e45386315f284cd2'
NATIVE_DECODE_SHA='c928daa1321ac80321093537a64d2339751690b5e53b96aa4b302898197d9a15'


class FrozenMLPBackwardPhase:
    @classmethod
    def from_observations(cls,observations,*,shadow_only=True):
        from .forward_observer import DenseObservation,ProjectionBinding
        pairs=[]
        for observation in observations:
            if type(observation) is not DenseObservation:
                raise TypeError('Exact dense producer observation required')
            for b in observation.bindings:
                if type(b) is not ProjectionBinding:
                    raise TypeError('Exact producer projection binding required')
                pairs.append((b.node,FrozenLinearTicket(b.plan,b.output_shape,b.input_width,
                    b.state,b.packed,b.packed_version,b.packed_pointer,b.label,
                    b.weight_ref,b.weight_version,b.quant_metadata)))
        return cls(pairs,shadow_only=shadow_only)

    def __init__(self,bindings,*,shadow_only=True):
        from bitsandbytes.autograd._functions import MatMul4Bit
        self.native=MatMul4Bit;self.original=MatMul4Bit.backward
        if (not inspect.isfunction(self.original) or hasattr(self.original,'__wrapped__')
                or hashlib.sha256(inspect.getsource(self.original).encode()).hexdigest()!=NATIVE_BACKWARD_SHA):
            raise RuntimeError('Exact native BNB backward source required before phase')
        if MatMul4Bit._backward_cls._forward_cls is not MatMul4Bit:
            raise RuntimeError('Native autograd context owner changed')
        self.original_code=self.original.__code__
        self.functional=self.original.__globals__['F']
        self.decode=self.functional.dequantize_4bit
        if (not inspect.isfunction(self.decode) or hasattr(self.decode,'__wrapped__')
                or hashlib.sha256(inspect.getsource(self.decode).encode()).hexdigest()!=NATIVE_DECODE_SHA):
            raise RuntimeError('Exact native BNB dequantizer source required before phase')
        self.decode_code=self.decode.__code__
        if type(shadow_only) is not bool:raise TypeError('Explicit diagnostic return mode required')
        self.shadow_only=shadow_only
        self.bindings={};self.events=[];self.counts=Counter();self.active=False;self.used=False
        for node,ticket in bindings:
            if type(node) is not MatMul4Bit._backward_cls or type(ticket) is not FrozenLinearTicket:
                raise RuntimeError('Producer must bind exact native context and ticket')
            if node in self.bindings:raise RuntimeError('Duplicate native context binding')
            self.bindings[node]=ticket
        if not self.bindings:raise RuntimeError('No owned MLP contexts bound')
        phase=self
        def backward(ctx,grad):
            if not phase.active or phase.native.backward is not backward or backward.__code__ is not phase.installed_code:
                raise RuntimeError('Conditional backward phase ownership changed')
            if not phase._native_code_intact():
                raise RuntimeError('Native backward/dequantization code or dispatch changed within phase')
            ticket=phase.bindings.get(ctx)
            return conditional_adjoint(ctx,grad,ticket,original_backward=phase.original,
                decode_weight=phase.decode,record=phase.record,shadow_only=phase.shadow_only)
        self.installed=backward
        self.installed_code=backward.__code__

    def _native_code_intact(self):
        return (self.original.__code__ is self.original_code
            and self.original.__globals__['F'] is self.functional
            and self.functional.dequantize_4bit is self.decode and self.decode.__code__ is self.decode_code
            and self.native._backward_cls._forward_cls is self.native)

    def record(self,event):
        self.events.append(event);self.counts[(event['route'],event['reason'])]+=1

    def __enter__(self):
        if self.used or self.active or self.native.backward is not self.original or not self._native_code_intact():
            raise RuntimeError('Backward phase must start once after native dense forward')
        self.native.backward=staticmethod(self.installed);self.active=True;self.used=True
        return self

    def __exit__(self,*error):
        owned=self.native.backward is self.installed
        if owned:self.native.backward=staticmethod(self.original)
        self.active=False;self.bindings.clear()
        if not owned:raise RuntimeError('Foreign backward mutation was not overwritten')
        return False
