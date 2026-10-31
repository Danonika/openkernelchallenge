"""Per-call detached metric ownership. No model/Trainer patch or tensor values read.

Caller must first verify the native Qwen->PEFT->Accelerate source/callable chain.
That proof, one producer, and one direct HF-parent return bind half source logits
to the public FP32 tensor. PEFT bypasses inner Qwen module hooks, so none are used.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import weakref
import torch

_ACTIVE=ContextVar('soup_ce_metric_transaction',default=None)
_FIELDS=('token_entropy','first_argmax','token_nll')


def metadata(t):
    return (tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t.storage_offset(),t._version)


@dataclass(frozen=True)
class DetachedSummary:
    token_nll: torch.Tensor
    token_entropy: torch.Tensor
    first_argmax: torch.Tensor
    route: str


class Transaction:
    def __init__(self,owner,labels,vocabulary=None):
        self.owner_ref=weakref.ref(owner)
        self.labels_ref=weakref.ref(labels)
        self.labels_metadata=metadata(labels)
        self.raw_ref=None;self.raw_metadata=None;self.public_ref=None;self.public_metadata=None
        self.source_kind=None;self.source_shape=None;self.source_device=None;self.vocabulary=vocabulary
        self.loss_ref=None;self.loss_metadata=None;self.completed=False
        self.summary=None;self.summary_metadata={};self.requested=False;self.offered=False
        self.bound=False;self.closed=False;self.invalid_reason=None;self.consumed=set()

    def invalidate(self,reason):
        self.invalid_reason=reason
        self.summary=None;self.summary_metadata.clear()
        return None

    def offer(self,*,token_nll,token_entropy,first_argmax,route):
        if self.closed or _ACTIVE.get() is not self or self.invalid_reason is not None:
            return False
        if self.offered:
            self.invalidate('duplicate_producer_offer');return False
        self.offered=True
        labels=self.labels_ref()
        if self.source_shape is None or labels is None or metadata(labels)!=self.labels_metadata:
            self.invalidate('producer_label_or_source_changed');return False
        expected_shape=self.source_shape[:-1];expected_device=self.source_device
        values={'token_nll':token_nll,'token_entropy':token_entropy,'first_argmax':first_argmax}
        for name,value in values.items():
            dtype=torch.int64 if name=='first_argmax' else torch.float32
            if (type(value) is not torch.Tensor or value.requires_grad or value.grad_fn is not None
                    or tuple(value.shape)!=expected_shape or value.device!=expected_device
                    or value.dtype!=dtype or not value.is_contiguous() or value.is_neg() or value.is_conj()):
                self.invalidate('summary_metadata:'+name);return False
        if route not in ('all_safe','mixed_rows','streaming'):
            self.invalidate('summary_route');return False
        self.summary=DetachedSummary(**values,route=route)
        self.summary_metadata={n:metadata(v) for n,v in values.items()}
        return True

    def complete(self,loss):
        if self.closed or _ACTIVE.get() is not self or self.invalid_reason is not None:return False
        if self.completed:
            self.invalidate('duplicate_loss_completion');return False
        self.completed=True
        if (self.summary is None or type(loss) is not torch.Tensor or loss.dtype!=torch.float32
                or loss.ndim!=0 or not loss.requires_grad or loss.device!=self.source_device):
            self.invalidate('producer_loss_contract');return False
        self.loss_ref=weakref.ref(loss)
        return True

    def bind_outputs(self,loss,outputs):
        if self.closed or _ACTIVE.get() is not self or self.invalid_reason is not None:return False
        if self.bound:
            self.invalidate('duplicate_public_bind');return False
        self.bound=True
        if (self.summary is None or self.loss_ref is None or self.loss_ref() is not loss
                or getattr(outputs,'loss',None) is not loss):
            self.invalidate('no_completed_producer_or_loss_identity');return False
        labels=self.labels_ref();public_logits=getattr(outputs,'logits',None)
        if labels is None or metadata(labels)!=self.labels_metadata:
            self.invalidate('public_or_label_metadata');return False
        if self.source_kind=='materialized':
            if (type(public_logits) is not torch.Tensor or public_logits.dtype!=torch.float32
                    or tuple(public_logits.shape)!=self.source_shape or public_logits.device!=self.source_device
                    or not public_logits.is_contiguous() or public_logits.is_neg() or public_logits.is_conj()):
                self.invalidate('public_or_label_metadata');return False
            self.public_ref=weakref.ref(public_logits);self.public_metadata=metadata(public_logits)
        elif self.source_kind=='streaming':
            if public_logits is not None:
                self.invalidate('streaming_unexpected_full_logits');return False
        else:
            self.invalidate('unknown_source_kind');return False
        # Native Accelerate preserves FP32 scalar Tensor identity. HF's optional
        # native in-place loss scale retains that identity; stamp its current
        # metadata now, not before that known native operation.
        self.loss_metadata=metadata(loss)
        return True

    def take(self,name,public_logits,labels):
        if (name not in _FIELDS or self.closed or _ACTIVE.get() is not self
                or self.invalid_reason is not None or self.summary is None):return None
        if name in self.consumed:
            return self.invalidate('duplicate_consumer:'+name)
        original_labels=self.labels_ref()
        public_ok=(self.source_kind=="streaming" and public_logits is None) or (
            self.source_kind=="materialized" and self.public_ref is not None
            and self.public_ref() is public_logits and metadata(public_logits)==self.public_metadata)
        loss=self.loss_ref() if self.loss_ref is not None else None
        if (not public_ok or loss is None or metadata(loss)!=self.loss_metadata
                or original_labels is not labels or metadata(labels)!=self.labels_metadata):
            return self.invalidate('consumer_identity_or_version')
        for field in _FIELDS:
            if metadata(getattr(self.summary,field))!=self.summary_metadata[field]:
                return self.invalidate('summary_modified:'+field)
        self.consumed.add(name)
        return getattr(self.summary,name)

    def close(self):
        self.closed=True;self.summary=None;self.summary_metadata.clear()
        self.raw_ref=None;self.public_ref=None;self.loss_ref=None
        self.producer_owner_ref=None
        if hasattr(self,'chunked_packet'):del self.chunked_packet


@contextmanager
def scope(owner,labels,*,vocabulary=None):
    if _ACTIVE.get() is not None:raise RuntimeError('Nested metric ownership requires native suspended fallback')
    transaction=Transaction(owner,labels,vocabulary);token=_ACTIVE.set(transaction)
    try:yield transaction
    finally:transaction.close();_ACTIVE.reset(token)


@contextmanager
def suspended():
    token=_ACTIVE.set(None)
    try:yield
    finally:_ACTIVE.reset(token)


def request_sink(owner,raw_logits,labels):
    transaction=_ACTIVE.get()
    if transaction is None or transaction.closed or transaction.invalid_reason is not None:return None
    if transaction.requested:
        transaction.invalidate('multiple_loss_producers');return None
    if (transaction.owner_ref() is not owner or transaction.labels_ref() is not labels
            or type(raw_logits) is not torch.Tensor or raw_logits.dtype!=torch.float16
            or raw_logits.ndim!=3 or raw_logits.shape[:-1]!=labels.shape
            or raw_logits.device!=labels.device or not raw_logits.is_contiguous()
            or raw_logits.is_neg() or raw_logits.is_conj() or metadata(labels)!=transaction.labels_metadata):
        transaction.invalidate('producer_identity_or_metadata');return None
    if transaction.vocabulary is not None and raw_logits.shape[-1]!=transaction.vocabulary:
        transaction.invalidate("producer_vocabulary");return None
    transaction.requested=True;transaction.raw_ref=weakref.ref(raw_logits)
    transaction.raw_metadata=metadata(raw_logits);transaction.source_kind="materialized"
    transaction.source_shape=tuple(raw_logits.shape);transaction.source_device=raw_logits.device
    return ProducerSink(transaction)


def current():return _ACTIVE.get()


class ProducerSink:
    def __init__(self,transaction):self.transaction=transaction
    def __call__(self,**fields):return self.transaction.offer(**fields)
    def complete(self,loss):return self.transaction.complete(loss)


def request_streaming_sink(owner,labels,*,vocabulary):
    transaction=_ACTIVE.get()
    if transaction is None or transaction.closed or transaction.invalid_reason is not None:return None
    if transaction.requested:
        transaction.invalidate('multiple_loss_producers');return None
    if (transaction.owner_ref() is not owner or transaction.labels_ref() is not labels
            or metadata(labels)!=transaction.labels_metadata or type(vocabulary) is not int
            or vocabulary<=1 or transaction.vocabulary!=vocabulary):
        transaction.invalidate('streaming_producer_identity_or_metadata');return None
    transaction.requested=True;transaction.source_kind='streaming'
    transaction.source_shape=tuple(labels.shape)+(vocabulary,);transaction.source_device=labels.device
    return ProducerSink(transaction)
