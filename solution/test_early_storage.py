"""Mock PEFT preparation contract for original early tied FP16 storage.
Not a real CUDA/Qwen model test; physical GPU preflight is still required.
"""
from types import SimpleNamespace
import torch
from . import early_storage, frozen_storage

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(20, 16, dtype=torch.float16)
        self.head = torch.nn.Linear(16, 20, bias=False)
        self.head.weight = self.emb.weight
    def get_input_embeddings(self):
        return self.emb
    def get_output_embeddings(self):
        return self.head

def test_preserves_tied_half_parameter_and_other_peft_updates(monkeypatch):
    import peft
    model = Dummy()
    param = model.emb.weight
    before = param.detach().clone()
    seen = []
    def pretend_prepare(m, use_gradient_checkpointing=False):
        seen.append((tuple(m.emb.weight.shape),m.emb.weight is m.head.weight))
        for p in m.parameters():
            p.requires_grad_(False)
            if p.dtype == torch.float16:
                p.data = p.data.float()
        return m
    def capture(m, use_gc):
        return SimpleNamespace(model=m, embedding=m.emb, head=m.head,
                               parameter=m.emb.weight, half_storage=m.emb.weight.detach())
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", pretend_prepare)
    monkeypatch.setattr(frozen_storage, "_capture_candidate", capture)
    applied = []
    monkeypatch.setattr(frozen_storage, "_install_storage", applied.append)
    assert early_storage.patch_early_storage()
    out = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert out is model
    assert seen == [((0,), True)]
    assert model.emb.weight is param is model.head.weight
    assert param.dtype == torch.float16 and not param.requires_grad
    assert torch.equal(param, before)
    assert len(applied) == 1

def test_original_data_restored_when_prepare_raises(monkeypatch):
    import peft
    model = Dummy()
    before = model.emb.weight.detach().clone()
    def fails(m, use_gradient_checkpointing=False):
        assert m.emb.weight.numel() == 0
        raise RuntimeError("mock PEFT failure")
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", fails)
    monkeypatch.setattr(frozen_storage, "_capture_candidate",
                        lambda m,g: SimpleNamespace(model=m, embedding=m.emb,
                             head=m.head, parameter=m.emb.weight,
                             half_storage=m.emb.weight.detach()))
    assert early_storage.patch_early_storage()
    import pytest
    with pytest.raises(RuntimeError, match="mock PEFT failure"):
        peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert model.emb.weight is model.head.weight
    assert torch.equal(model.emb.weight, before)
