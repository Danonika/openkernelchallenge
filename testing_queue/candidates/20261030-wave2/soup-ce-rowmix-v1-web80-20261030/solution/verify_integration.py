"""Explicitly authorized tiny actual-class T4 integration gate, not training."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch

import solution
from .numerical_gates import GATES, _scalar_check, _gradient_check


def run(report):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.loss.loss_utils import ForCausalLMLoss
    from accelerate.utils.operations import convert_outputs_to_fp32

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability() != (7, 5)
            or "T4" not in torch.cuda.get_device_name()):
        raise RuntimeError("Requires one visible T4; no dependency installation is attempted")
    report["device"] = torch.cuda.get_device_name()
    native_init = Qwen2ForCausalLM.__init__
    settings = dict(vocab_size=151936, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=32, attention_dropout=0.0,
                    tie_word_embeddings=True, use_cache=False)
    torch.manual_seed(9030801)
    reference = Qwen2ForCausalLM(Qwen2Config(**settings)).cuda()
    assert reference.loss_function is ForCausalLMLoss
    assert solution.apply() is True
    installed_init = Qwen2ForCausalLM.__init__
    assert solution.apply() is True and Qwen2ForCausalLM.__init__ is installed_init
    candidate = Qwen2ForCausalLM(Qwen2Config(**settings)).cuda()
    candidate.load_state_dict(reference.state_dict())
    assert reference.loss_function is ForCausalLMLoss
    assert candidate.loss_function.native_loss is ForCausalLMLoss
    assert solution.diagnostics()["events"]["model_loss_installed"] == 1
    for model in (reference, candidate):
        model.train()
        model.requires_grad_(False)
    report["constructor_and_per_model_install"] = {"passed": True, "repeated_apply_idempotent": True}

    data = torch.randn(4, 3, 32, device="cuda", dtype=torch.float32)
    labels = torch.tensor([[-100, 0, 151935], [-100, 151552, 1],
                           [-100, -100, 151551], [-100, -100, -100]], device="cuda")
    mask = torch.ones_like(labels)
    denominator = torch.tensor(13, device="cuda", dtype=torch.int64)
    label_copy, data_copy = labels.clone(), data.clone()
    d_version = denominator._version
    observations = {}

    def training_call(name, model):
        seen = {"head_calls": 0}
        def capture(_module, _args, output):
            seen["head_calls"] += 1
            seen["raw"] = output
            output.retain_grad()
        hook = model.lm_head.register_forward_hook(capture)
        inputs = data.detach().clone().requires_grad_(True)
        wrapped = convert_outputs_to_fp32(model.forward)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                output = wrapped(inputs_embeds=inputs, attention_mask=mask, labels=labels,
                                 num_items_in_batch=denominator, use_cache=False,
                                 output_attentions=False, output_hidden_states=False)
            scaler = torch.amp.GradScaler("cuda", init_scale=65536.0)
            scaler.scale(output.loss).backward()
        finally:
            hook.remove()
        assert seen["head_calls"] == 1
        assert seen["raw"].dtype == torch.float16 and seen["raw"].shape == (4, 3, 151936)
        assert output.logits.dtype == torch.float32 and output.loss.dtype == torch.float32
        assert output.attentions is None and output.hidden_states is None
        assert torch.equal(output.logits, seen["raw"].float())
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert seen["raw"].grad.dtype == torch.float16
        observations[name] = {"loss": output.loss.detach(), "logits": output.logits.detach(),
                              "raw_gradient": seen["raw"].grad.detach(), "input_gradient": inputs.grad.detach()}
        report[name + "_boundary"] = {"head_calls": 1, "raw_dtype": "torch.float16",
                                       "returned_dtype": "torch.float32", "grad_scaler_scale": scaler.get_scale(),
                                       "RNG_unchanged": True}

    training_call("native", reference)
    training_call("candidate", candidate)
    a, b = observations["candidate"], observations["native"]
    loss_gate = _scalar_check(a["loss"], b["loss"])
    grad_gate = _gradient_check(a["raw_gradient"], b["raw_gradient"])
    input_diagnostic = _gradient_check(a["input_gradient"], b["input_gradient"])
    logits_equal = torch.equal(a["logits"], b["logits"])
    routes = solution.diagnostics()
    own_active = (routes["events"].get("triton_forward", 0) == 1
                  and routes["events"].get("triton_backward", 0) == 1
                  and routes["events"].get("native_forward_fallback", 0) == 0
                  and routes["events"].get("native_backward_fallback", 0) == 0
                  and routes["denominators"].get("int64_cuda:13", 0) == 1
                  and routes["native_ignored_output_controls"].get(
                      "output_attentions=False,output_hidden_states=False", 0) == 1)
    report["training_pair"] = {"loss": loss_gate, "final_FP16_logit_gradient": grad_gate,
                                "input_FP32_gradient_diagnostic_only": input_diagnostic,
                                "returned_FP32_logits_exact": logits_equal,
                                "nonvacuous_own_forward_backward": own_active, "routes": routes}
    assert loss_gate["passed"] and grad_gate["passed"] and logits_equal and own_active
    assert input_diagnostic["nonfinite_class_mismatches"] == 0
    assert denominator._version == d_version and int(denominator) == 13
    assert torch.equal(labels, label_copy) and torch.equal(data, data_copy)

    # One real evaluation forward. Native expected loss uses its already-produced
    # head output; it never invokes the model/head a second time.
    candidate.eval()
    eval_seen = {"head_calls": 0}
    def capture_eval(_module, _args, output):
        eval_seen["head_calls"] += 1
        eval_seen["raw"] = output
    hook = candidate.lm_head.register_forward_hook(capture_eval)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            evaluated = convert_outputs_to_fp32(candidate.forward)(
                inputs_embeds=data, attention_mask=mask, labels=labels,
                num_items_in_batch=denominator, use_cache=False,
                output_attentions=False, output_hidden_states=False)
            expected_eval = ForCausalLMLoss(eval_seen["raw"], labels, 151936,
                                            num_items_in_batch=denominator)
    finally:
        hook.remove()
    assert eval_seen["head_calls"] == 1 and torch.equal(evaluated.loss, expected_eval)
    assert torch.equal(evaluated.logits, eval_seen["raw"].float())
    candidate.train()
    unknown_input = eval_seen["raw"].detach().requires_grad_(True)
    unknown = candidate.loss_function(unknown_input, labels, 151936,
                                      num_items_in_batch=denominator, adapter_probe=True)
    expected_unknown = ForCausalLMLoss(unknown_input, labels, 151936,
                                      num_items_in_batch=denominator, adapter_probe=True)
    assert torch.equal(unknown, expected_unknown) and eval_seen["head_calls"] == 1
    after = solution.diagnostics()
    assert after["events"].get("triton_forward", 0) == 1 and after["events"].get("triton_backward", 0) == 1
    assert after["reasons"].get("owner_or_evaluation", 0) == 1
    assert after["reasons"].get("alternate_arguments:adapter_probe", 0) == 1
    report["evaluation_and_unknown_keyword"] = {"passed": True, "extra_head_forward": False, "routes": after}

    # A separate deterministic raw-logit fixture MUST activate the new mixed
    # route; the random tiny model above only validates the original safe path.
    from .verify_rowmix import run as verify_rowmix
    verify_rowmix(report)

    # A second small CPU instance exercises foreign-loss preservation without
    # another CUDA forward or numerical sweep.
    foreign_model = Qwen2ForCausalLM(Qwen2Config(**settings))
    foreign = lambda *args, **kwargs: None
    foreign_model.loss_function = foreign
    restored = solution.restore()
    assert Qwen2ForCausalLM.__init__ is native_init
    assert candidate.loss_function is ForCausalLMLoss and reference.loss_function is ForCausalLMLoss
    assert foreign_model.loss_function is foreign
    assert restored["model_losses"] == {"restored": 1, "foreign_loss_preserved": 1}
    assert solution.restore()["constructor"] == "already_restored"
    report["restore"] = {"passed": True, "details": restored}
    report["status"] = "passed_gpu_integration_preflight"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    root = Path(__file__).resolve().parent
    report = {"status": "running", "gates": GATES,
              "scope": "Tiny actual-class loss/return plus nonvacuous native-near/custom-safe row-mix integration; no full task, optimizer step or training benchmark",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}}
    try:
        run(report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}), flush=True)
    return 0 if report["status"] == "passed_gpu_integration_preflight" else 1


if __name__ == "__main__":
    raise SystemExit(main())
