"""Explicit tiny target-only row-mixed CE gate; not training or a benchmark.

Imported by verify_integration only. Uses native CE as the oracle, requiring
bitwise native near-row gradients while retaining the existing safe-row gates.
"""
import torch
import torch.nn.functional as F
from . import materialized_ce as ce
from . import mixed_rows as mixed
from .numerical_gates import _scalar_check, _gradient_check


def _same_classes_and_finite(actual, expected):
    classes = (torch.equal(actual.isnan(), expected.isnan())
               and torch.equal(actual.isposinf(), expected.isposinf())
               and torch.equal(actual.isneginf(), expected.isneginf()))
    finite = actual.isfinite() & expected.isfinite()
    return classes and torch.equal(actual[finite], expected[finite])


def _bits_equal(actual, expected):
    return torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                       expected.contiguous().reshape(-1).view(torch.uint8))


def run(report):
    device = torch.device('cuda')
    labels = torch.tensor([[0,1,2], [0,3,4], [0,-100,5], [0,6,7]], device=device)
    target = ce._targets(labels).view(-1)
    logits = torch.zeros(4,3,ce.VOCABULARY, device=device, dtype=torch.float16)
    flat = logits.view(-1,ce.VOCABULARY)
    # With 151936 classes: target19 => native NLL about 8.5e-4; target30 =>
    # exact-zero/very-small native NLL. Both remain behind the fixed 1e-3 guard.
    flat[0,1] = 19.
    flat[4,4] = 30.
    native_rows = torch.tensor([0,4],device=device)
    denominator = torch.tensor(13,device=device,dtype=torch.int64)
    before_logits, before_labels = logits.clone(), labels.clone()
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    pairs = []
    for upstream in (1., 65536.):
        a, b = logits.clone().requires_grad_(), logits.clone().requires_grad_()
        diagnostic = ce.Diagnostics()
        actual = ce.materialized_causal_loss(a,labels,denominator,diagnostics=diagnostic)
        expected = ce.native_causal_loss(b,labels,denominator)
        q = torch.tensor(upstream,device=device,dtype=torch.float32)
        actual.backward(q); expected.backward(q)
        loss_gate = _scalar_check(actual.detach(),expected.detach())
        gradient_gate = _gradient_check(a.grad,b.grad)
        near_exact = _bits_equal(a.grad.view(-1,ce.VOCABULARY).index_select(0,native_rows),
                                 b.grad.view(-1,ce.VOCABULARY).index_select(0,native_rows))
        ignored_zero = bool((a.grad.view(-1,ce.VOCABULARY)[target==-100]==0).all().item())
        forward = [e for e in diagnostic.events if e['event']=='triton_forward']
        backward = [e for e in diagnostic.events if e['event']=='triton_backward']
        active = (len(forward)==len(backward)==1 and forward[0].get('route')=='mixed_rows'
                  and backward[0].get('route')=='mixed_rows'
                  and forward[0]['native_supervised_rows']==2
                  and forward[0]['safe_supervised_rows']==5 and forward[0]['ignored_rows']==5
                  and not any('fallback' in e['event'] for e in diagnostic.events))
        pairs.append({'upstream':upstream,'loss':loss_gate,'FP16_gradient':gradient_gate,
                      'native_near_rows_bitwise_exact':near_exact,'ignored_gradients_zero':ignored_zero,
                      'nonvacuous_mixed_route':active,'diagnostics':diagnostic.events})
        report['row_mixed_pairs'] = pairs
        assert loss_gate['passed'] and gradient_gate['passed'] and near_exact and ignored_zero and active

    # Compare the subset native selected values against a full native CE call;
    # the one-column table must then retain the full row count and one reduction.
    with torch.no_grad():
        native_values = -F.cross_entropy(flat.float(),target,reduction='none',ignore_index=-100)
        selected = native_values.clone()
        selected.index_fill_(0,native_rows,-123.)  # Require actual replacement.
        values = mixed.mixed_selected_values(logits,target,selected,native_rows)
        near_value_exact = _bits_equal(values.index_select(0,native_rows),
                                      native_values.index_select(0,native_rows))
        internal_target = torch.where(target!=-100,torch.zeros_like(target),target)
        full = torch.zeros_like(flat,dtype=torch.float32)
        full[:,0] = values
        one_column_sum = mixed.ordered_nll_sum(values,target)
        full_geometry_sum = F.nll_loss(full,internal_target,reduction='sum',ignore_index=-100)
        same_sum = _bits_equal(one_column_sum,full_geometry_sum)
    report['row_mixed_forward_geometry'] = {'native_near_selected_bitwise_exact':near_value_exact,
                                             'full_N_rows_NLL_sum_bitwise_exact':same_sum,
                                             'does_not_claim_all_custom_values_bitwise_exact':True}
    assert near_value_exact and same_sum

    fallbacks = []
    for kind in ('all_near','all_ignored','nonfinite_ignored'):
        z,y=logits.clone(),labels.clone()
        if kind=='all_near':
            for row in range(target.numel()):
                index=int(target[row].item())
                if index!=-100:z.view(-1,ce.VOCABULARY)[row,index]=30.
            reason='all_supervised_rows_near_certain'
        elif kind=='all_ignored':
            y.fill_(-100);reason='no_supervised_rows'
        else:
            z[0,2,0]=float('nan');reason='nonfinite_logits_or_invalid_target'
        a,b=z.clone().requires_grad_(),z.clone().requires_grad_()
        diagnostic=ce.Diagnostics()
        actual=ce.materialized_causal_loss(a,y,denominator,diagnostics=diagnostic)
        expected=ce.native_causal_loss(b,y,denominator)
        actual.backward();expected.backward()
        same = _same_classes_and_finite(actual,expected) and _same_classes_and_finite(a.grad,b.grad)
        native = diagnostic.events[-1]=={'event':'native_forward_fallback','reason':reason}
        no_custom_route = not any(e['event'] in ('triton_forward','triton_backward') for e in diagnostic.events)
        fallbacks.append({'case':kind,'native_values_and_nonfinite_positions':same,
                          'native_route':native,'no_custom_loss_route':no_custom_route})
        report['row_mixed_fallbacks']=fallbacks
        assert same and native and no_custom_route
    unchanged = (torch.equal(logits,before_logits) and torch.equal(labels,before_labels)
                 and torch.equal(rng_cpu,torch.get_rng_state())
                 and torch.equal(rng_cuda,torch.cuda.get_rng_state()))
    report['row_mixed_inputs_and_RNG_unchanged']=unchanged
    assert unchanged
