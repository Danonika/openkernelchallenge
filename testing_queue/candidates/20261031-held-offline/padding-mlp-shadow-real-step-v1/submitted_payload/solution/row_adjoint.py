"""Conditional row compaction of a frozen native linear's actual adjoint.

DIAGNOSTIC native-shadow version: no speed or production claim.
No forward, producer, activation derivative, random mask, or kernel is defined
here. The owned caller supplies one original-batch mask and an exact native
MatMul4Bit context ticket. Unsupported cases call its original backward.
"""
from dataclasses import dataclass,field
import torch


_PLAN_FACTORY=object()


@dataclass(frozen=True,eq=False)
class OriginalBatchMask:
    mask:torch.Tensor
    version:int
    pointer:int
    shape:tuple
    live:torch.Tensor
    padded:torch.Tensor
    _factory:object=field(default=None,repr=False,compare=False)
    _mask_owner:object=field(default=None,repr=False,compare=False)
    _live_owner:object=field(default=None,repr=False,compare=False)
    _padded_owner:object=field(default=None,repr=False,compare=False)
    _index_versions:tuple=field(default=(),repr=False,compare=False)
    _index_pointers:tuple=field(default=(),repr=False,compare=False)
    _index_shapes:tuple=field(default=(),repr=False,compare=False)

    @classmethod
    def prepare(cls,mask):
        if type(mask) is not torch.Tensor or mask.ndim!=2 or mask.dtype!=torch.int64 or not mask.is_contiguous():
            raise ValueError('Original contiguous 2D int64 batch attention mask required')
        if mask.is_neg() or mask.is_conj() or mask.requires_grad:raise ValueError('Ordinary mask required')
        if not bool(((mask==0)|(mask==1)).all()):raise ValueError('Original binary attention mask required')
        lengths=mask.sum(-1)
        if not bool((lengths>0).all()):raise ValueError('Nonempty original rows required')
        prefix=torch.arange(mask.shape[1],device=mask.device)[None,:]<lengths[:,None]
        if not torch.equal(prefix,mask.bool()):raise ValueError('Original right-padding geometry required')
        flat=mask.reshape(-1)
        live=flat.nonzero().reshape(-1);padded=flat.eq(0).nonzero().reshape(-1)
        return cls(mask,mask._version,mask.untyped_storage().data_ptr(),tuple(mask.shape),live,padded,
            _PLAN_FACTORY,mask,live,padded,(live._version,padded._version),
            (live.untyped_storage().data_ptr(),padded.untyped_storage().data_ptr()),
            (tuple(live.shape),tuple(padded.shape)))

    def unchanged(self):
        # This is ordinary framework version/identity ownership, not a promise
        # against .data/raw-storage writes that bypass native version counters.
        return (self._factory is _PLAN_FACTORY and self.mask is self._mask_owner
            and self.live is self._live_owner and self.padded is self._padded_owner
            and type(self.mask) is torch.Tensor and tuple(self.mask.shape)==self.shape
            and self.mask.dtype==torch.int64 and self.mask.is_contiguous()
            and self.mask._version==self.version and self.mask.untyped_storage().data_ptr()==self.pointer
            and all(type(x) is torch.Tensor and x.dtype==torch.int64 and x.device==self.mask.device
                and x.is_contiguous() and x._version==version and x.untyped_storage().data_ptr()==pointer
                and tuple(x.shape)==shape for x,version,pointer,shape in
                zip((self.live,self.padded),self._index_versions,self._index_pointers,self._index_shapes))
            and len(self._index_versions)==len(self._index_pointers)==len(self._index_shapes)==2)


@dataclass(frozen=True)
class FrozenLinearTicket:
    plan:OriginalBatchMask
    output_shape:tuple
    input_width:int
    state:object
    packed:torch.Tensor
    packed_version:int
    packed_pointer:int
    label:str
    weight_ref:object
    weight_version:int
    quant_metadata:tuple


def _scalar_same(actual,expected):
    # Never invoke tensor/QuantState value equality for ownership snapshots.
    return (type(actual) is type(expected) and type(expected) in
            (type(None),bool,int,str,torch.dtype) and actual==expected)


def _integer_shape_same(actual,expected):
    return (type(actual) in (tuple,list,torch.Size) and len(actual)==len(expected)
            and all(type(a) is int and type(b) is int and a==b for a,b in zip(actual,expected)))


def _tensor_stamp_same(actual,stamp):
    if stamp is None:return actual is None
    return (actual is stamp.tensor and isinstance(actual,torch.Tensor)
        and actual._version==stamp.version and actual.untyped_storage().data_ptr()==stamp.pointer
        and tuple(actual.shape)==stamp.shape and tuple(actual.stride())==stamp.stride
        and actual.dtype==stamp.dtype and actual.device==stamp.device)


def _quant_metadata_same(state,snapshot):
    shape,blocksize,kind,dtype,nested,child,stamps,child_snapshot=snapshot
    if (not _integer_shape_same(getattr(state,'shape',None),shape)
            or not all(_scalar_same(getattr(state,name,None),value) for name,value in
                (('blocksize',blocksize),('quant_type',kind),('dtype',dtype),('nested',nested)))
            or getattr(state,'state2',None) is not child):return False
    if not all(_tensor_stamp_same(getattr(state,name,None),stamp) for name,stamp in stamps):return False
    if child is None:return child_snapshot is None
    cblock,ckind,cdtype,cstamps=child_snapshot
    return (all(_scalar_same(getattr(child,name,None),value) for name,value in
            (('blocksize',cblock),('quant_type',ckind),('dtype',cdtype)))
        and all(_tensor_stamp_same(getattr(child,name,None),stamp) for name,stamp in cstamps))


def conditional_adjoint(ctx,grad,ticket,*,original_backward,decode_weight,record,shadow_only=False):
    """Preserve actual half cotangent; compact ONLY a verified frozen dX GEMM.

    Native BNB's frozen/no-bias backward reads no saved activation: only actual
    grad_output and decoded W. Activation backward and FP32 branch fan-in stay
    in the caller's unchanged dense graph. Every unsupported condition invokes
    original_backward(ctx, grad), never an approximate replacement fallback.
    """
    def fallback(reason):
        record({'route':'native','reason':reason,'label':None if ticket is None else ticket.label})
        return original_backward(ctx,grad)
    if ticket is None:return fallback('unowned_context')
    p=ticket.plan
    if not p.unchanged():return fallback('original_mask_changed')
    if p.padded.numel()==0:return fallback('no_padding_work')
    if (getattr(ctx,'is_empty',True) or tuple(getattr(ctx,'needs_input_grad',()))!=(True,False,False,False,False)
            or getattr(ctx,'dtype_A',None)!=torch.float16 or getattr(ctx,'dtype_B',None)!=torch.uint8
            or getattr(ctx,'dtype_bias',False) is not None):return fallback('native_context_contract')
    if getattr(ctx,'state',None) is not ticket.state:return fallback('quant_state_identity_changed')
    tensors=getattr(ctx,'tensors',None)
    if not isinstance(tensors,tuple) or len(tensors)!=2 or tensors[0] is not None or tensors[1] is not ticket.packed:
        return fallback('packed_context_identity_changed')
    if (ticket.packed.requires_grad or ticket.packed._version!=ticket.packed_version
            or ticket.packed.untyped_storage().data_ptr()!=ticket.packed_pointer):return fallback('packed_weight_changed')
    weight_owner=ticket.weight_ref()
    if (weight_owner is None or weight_owner._version!=ticket.weight_version
            or weight_owner.requires_grad or getattr(weight_owner,'quant_state',None) is not ticket.state
            or weight_owner.untyped_storage().data_ptr()!=ticket.packed_pointer
            or weight_owner.data_ptr()!=ticket.packed.data_ptr()):return fallback('producer_weight_changed')
    if not _quant_metadata_same(ticket.state,ticket.quant_metadata):return fallback('quant_metadata_changed')
    if (type(grad) is not torch.Tensor or grad.dtype!=torch.float16 or grad.device!=p.mask.device
            or tuple(grad.shape)!=ticket.output_shape or grad.ndim!=3
            or tuple(grad.shape[:2])!=p.shape or not grad.is_contiguous()
            or grad.is_neg() or grad.is_conj()):return fallback('actual_half_adjoint_geometry')
    if torch.is_grad_enabled():return fallback('higher_order_not_owned')
    flat=grad.reshape(-1,grad.shape[-1])
    # Actual incoming values are checked, not inferred from ignored labels or
    # from the upstream scale. Both +/-zero are numeric zero, not a bit claim.
    if not bool(torch.isfinite(grad).all()):return fallback('nonfinite_actual_adjoint')
    if bool((flat.index_select(0,p.padded)!=0).any()):return fallback('nonzero_padded_adjoint')
    weight=decode_weight(ticket.packed,ticket.state).to(grad.dtype)
    if (type(weight) is not torch.Tensor or weight.dtype!=torch.float16 or weight.device!=grad.device
            or tuple(weight.shape)!=(grad.shape[-1],ticket.input_width)):
        return fallback('decoded_weight_contract')
    # The extra decode is diagnostic overhead on this rare fallback; the
    # original native backward is still used verbatim and may decode again.
    if not bool(torch.isfinite(weight).all()):return fallback('nonfinite_decoded_weight')
    # First bounded target observation retains the actual full-M native result.
    # Finite dy/W alone does not certify reduced-precision GEMM output range.
    # This extra native work is why this version is DIAGNOSTIC ONLY, not a
    # claimed optimized training candidate or a timing benchmark.
    native_result=original_backward(ctx,grad)
    native_dx=native_result[0]
    if (type(native_dx) is not torch.Tensor or native_dx.dtype!=grad.dtype
            or tuple(native_dx.shape)!=(*p.shape,ticket.input_width)):
        record({'route':'native','reason':'native_output_contract','label':ticket.label})
        return native_result
    if not bool(torch.isfinite(native_dx).all()):
        record({'route':'native','reason':'nonfinite_native_output_witness','label':ticket.label})
        return native_result
    native_padded=native_dx.reshape(-1,ticket.input_width).index_select(0,p.padded)
    if bool((native_padded!=0).any()):
        record({'route':'native','reason':'nonzero_native_padded_output_witness','label':ticket.label})
        return native_result
    selected=flat.index_select(0,p.live)
    compact_dx=torch.matmul(selected,weight)
    if not bool(torch.isfinite(compact_dx).all()):
        record({'route':'native','reason':'nonfinite_compact_output','label':ticket.label})
        return native_result
    native_live=native_dx.reshape(-1,ticket.input_width).index_select(0,p.live)
    error=(compact_dx.double()-native_live.double())
    reference_norm=float(native_live.double().norm())
    relative=float(error.norm())/max(reference_norm,1e-30)
    max_dy=float(grad.abs().max());max_w=float(weight.abs().max())
    # Native half dX is scattered BEFORE the original independent half->FP32
    # branch cast edges. Never join gate/up adjoints in half or predivide q.
    dx=torch.zeros((*p.shape,ticket.input_width),device=grad.device,dtype=grad.dtype)
    dx.reshape(-1,ticket.input_width).index_copy_(0,p.live,compact_dx)
    record({'route':'shadow_compact' if shadow_only else 'compact','reason':'actual_finite_zero_pad_adjoint_and_finite_weight','label':ticket.label,
        'full_rows':flat.shape[0],'live_rows':p.live.numel(),'skipped_rows':p.padded.numel(),
        'input_width':ticket.input_width,'output_width':grad.shape[-1],'adjoint_dtype':str(grad.dtype),
        'original_mask_identity_matched':True,'finite_GEMM_geometry_tradeoff':True,
        'native_output_witness_required':True,'native_dX_finite':True,'compact_dX_finite':True,
        'native_padded_dX_numerically_zero':True,
        'local_live_dx_relative_l2':relative,'local_live_dx_max_abs':float(error.abs().max()),
        'max_abs_actual_dy':max_dy,'max_abs_decoded_weight':max_w,'K':grad.shape[-1],
        'unrounded_absolute_dot_upper_bound':grad.shape[-1]*max_dy*max_w,'range_certificate':False,
        'returned_native_result':shadow_only})
    return native_result if shadow_only else (dx,None,None,None,None)
