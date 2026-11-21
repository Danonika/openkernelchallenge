"""Unexecuted target-only native-head preflight; not measured training.

This newly added module does not change the three frozen candidate modules.
It constructs only a tiny random Qwen with the real vocabulary. It proves a
narrow native-head route when eventually run, not full NF4/PEFT/TRL integration.
"""
import argparse
import json
import os
from pathlib import Path
import traceback


def run():
    if os.environ.get("TRITON_INTERPRET"):
        raise RuntimeError("Interpreter mode is not target evidence")
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.loss.loss_utils import ForCausalLMLoss
    from . import apply, diagnostics, restore
    from .native_head import NativeHeadLoss

    if (torch.__version__ != "2.11.0+cu128"
            or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != "Tesla T4"
            or torch.cuda.get_device_capability(0) != (7, 5)):
        raise RuntimeError("Exact target Torch and one Tesla T4/SM75 required")
    torch.manual_seed(1)
    if not apply():
        raise RuntimeError("Pinned installation did not activate")
    try:
        config = Qwen2Config(vocab_size=151936, hidden_size=32,
                            intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=512,
                            tie_word_embeddings=True, attention_dropout=0.0,
                            use_cache=False)
        model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.float16)
        model.requires_grad_(False).train()
        if not isinstance(model.loss_function, NativeHeadLoss):
            raise RuntimeError("Production constructor failed to attach native loss")
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise RuntimeError("Tied Parameter identity changed")

        # Real model forward and native head, with small preflight geometry.
        # This is explicitly NOT a shortened substitute for the official task.
        embeds = torch.randn(4, 8, 32, device="cuda", dtype=torch.float16,
                             requires_grad=True)
        labels = torch.randint(0, config.vocab_size, (4, 8), device="cuda")
        labels[:, 0] = -100
        denominator = torch.tensor(28, device="cuda", dtype=torch.int64)
        outputs = model(inputs_embeds=embeds, labels=labels,
                        num_items_in_batch=denominator, use_cache=False)
        logits = outputs.logits
        if logits.dtype != torch.float16 or tuple(logits.shape) != (4, 8, 151936):
            raise RuntimeError("Actual returned head logits contract differs")
        native = ForCausalLMLoss(logits=logits, labels=labels,
                                vocab_size=config.vocab_size,
                                num_items_in_batch=denominator)
        if not torch.equal(outputs.loss, native):
            raise RuntimeError("Native forward value differs")

        # Compare the actual returned-logit gradient, including nonunit incoming
        # scale before narrowing. Avoid assumptions about unrelated model ops.
        scale = torch.tensor(32.0, device="cuda")
        before = torch.cuda.get_rng_state().clone()
        expected, = torch.autograd.grad(native, logits, grad_outputs=scale,
                                        retain_graph=True)
        actual, = torch.autograd.grad(outputs.loss, logits, grad_outputs=scale,
                                      retain_graph=True)
        if not torch.equal(before, torch.cuda.get_rng_state()):
            raise RuntimeError("Loss replay consumed CUDA RNG")
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            raise RuntimeError("Native replay logit gradient differs")
        outputs.loss.backward()
        if embeds.grad is None or not torch.isfinite(embeds.grad).all():
            raise RuntimeError("Real native-head/model backward did not complete")
        torch.cuda.synchronize()
        report = diagnostics()
        if (report.get("counts", {}).get("native_sum_forward_return") != 1
                or report.get("counts", {}).get("native_sum_backward_return") != 2
                or report.get("counts", {}).get("native_fallback", 0) != 0):
            raise RuntimeError("Native route was inactive, partial or fell back")
        return {"geometry": [4, 8, 151936], "tiny_model_only": True,
                "full_nf4_peft_trl_integration": False,
                "native_loss_equal": True, "native_logit_gradient_equal": True,
                "real_model_backward_completed": True,
                "loss_cuda_rng_unchanged": True, "diagnostics": report,
                "original_triton_kernels": [], "structural_own_share": 0.0}
    finally:
        restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {"status": "FAIL", "gpu_execution_attempted": True,
              "gpu_executed": None,
              "score": None, "admission": "NOT_ELIGIBLE_ZERO_OWNED_KERNELS"}
    try:
        report["checks"] = run()
        report.update(status="PASS_NATIVE_HEAD_PREFLIGHT_ONLY", gpu_executed=True)
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": args.out}), flush=True)
    raise SystemExit(report["status"] != "PASS_NATIVE_HEAD_PREFLIGHT_ONLY")


if __name__ == "__main__":
    main()
