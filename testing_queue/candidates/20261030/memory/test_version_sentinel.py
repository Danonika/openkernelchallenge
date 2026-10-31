"""New safety regression fixtures; explicitly CPU-only, no model execution."""
import contextlib
import gc
import io
import json
import traceback
import weakref
from pathlib import Path

import torch
from solution import adapter
from solution.version_sentinel import make_witness, runtime_contract, UnsupportedVersionWitness


def sentinel_storage_and_rng():
    before_rng = torch.random.get_rng_state().clone()
    records = []
    for shape in ((0,), (4,), (2, 3)):
        source = torch.arange(max(1, torch.tensor(shape).prod().item()), dtype=torch.float32)[:int(torch.tensor(shape).prod())].reshape(shape)
        original = source.clone()
        version, pointer = source._version, source.data_ptr()
        witness = make_witness(source)
        assert source._version == version and source.data_ptr() == pointer
        assert torch.equal(source, original)
        assert witness.untyped_storage().nbytes() == 0
        source.add_(1)
        assert witness._version == version + 1
        records.append({"shape": list(shape), "witness_storage_bytes": 0})
    assert torch.equal(before_rng, torch.random.get_rng_state())
    return {"cases": records, "runtime": runtime_contract()}


def escaped_alias_after_python_release():
    x = torch.arange(8., requires_grad=True)
    half = x.half()
    wide = half.float()
    result = torch.nn.functional.silu(wide)
    escaped = wide.detach()
    reference = weakref.ref(wide)
    assert adapter._register_one(result.grad_fn._raw_saved_self, wide, half)
    del wide
    gc.collect()
    assert reference() is None, "test must actually cover collected wide object"
    escaped.add_(1)
    try:
        result.sum().backward()
    except RuntimeError as exc:
        assert "version witness changed" in str(exc), str(exc)
    else:
        raise AssertionError("escaped detached alias mutation was not detected")
    return {"wide_object_collected": True, "mutation_detected_before_restore": True}


def payload_mutation_and_native_parity():
    x = torch.arange(8., requires_grad=True)
    half = x.half(); wide = half.float()
    result = torch.nn.functional.silu(wide)
    assert adapter._register_one(result.grad_fn._raw_saved_self, wide, half)
    half.detach().add_(1)
    try:
        result.sum().backward()
    except RuntimeError as exc:
        assert "saved producer payload was modified" in str(exc)
    else:
        raise AssertionError("half producer mutation undetected")
    return {"payload_mutation_detected": True}


def sentinel_retains_no_wide_storage():
    class Slot:
        packet = None
        def register_hooks(self, pack, unpack):
            self.packet = pack(wide)
    half = torch.arange(8., dtype=torch.float16)
    wide = half.float()
    old_storage = weakref.ref(wide.untyped_storage())
    ref = weakref.ref(wide)
    slot = Slot()
    assert adapter._register_one(slot, wide, half)
    del wide
    gc.collect()
    assert ref() is None
    assert slot.packet.wide_witness.untyped_storage().nbytes() == 0
    assert slot.packet.payload.dtype == torch.float16
    assert old_storage() is None
    with torch.no_grad():
        restored = adapter._unpack(slot.packet)
    assert torch.equal(restored, half.float())
    return {"old_wide_storage_python_wrapper_collected": True,
            "zero_storage_witness": True, "not_allocator_or_peak_evidence": True}


def unsupported_witness_refuses_slot():
    original = adapter.make_witness
    class Slot:
        calls = 0
        def register_hooks(self, pack, unpack):
            self.calls += 1
    slot = Slot(); half = torch.ones(3, dtype=torch.float16); wide = half.float()
    pointer, version = wide.data_ptr(), wide._version
    def unsupported(t):
        raise UnsupportedVersionWitness("synthetic unsupported runtime")
    adapter.make_witness = unsupported
    try:
        assert adapter._register_one(slot, wide, half) is False
    finally:
        adapter.make_witness = original
    assert slot.calls == 0 and wide.data_ptr() == pointer and wide._version == version
    return {"raw_slot_untouched_on_decline": True}


def timeout_progress_is_flushed():
    # Exercise the actual logging code without target install or any CUDA call.
    previous = adapter._STATE
    adapter._STATE = {"mode": "production"}
    adapter.reset_stats()
    class FlushBuffer(io.StringIO):
        flushes = 0
        def flush(self): self.flushes += 1
    output = FlushBuffer()
    try:
        with contextlib.redirect_stdout(output):
            for _ in range(7): adapter._count("restores")
    finally:
        adapter._STATE = previous
    lines = output.getvalue().splitlines()
    values = [json.loads(line.split(" ", 1)[1]) for line in lines]
    assert [x["cumulative_lower_bound"]["restores"] for x in values] == [1, 2, 4]
    assert output.flushes == 3 and all(x["completed_training"] is False for x in values)
    return {"records": 3, "flushes": output.flushes, "cumulative_lower_bounds": [1, 2, 4]}


def main():
    torch.set_num_threads(2)
    records = []
    for function in (sentinel_storage_and_rng, escaped_alias_after_python_release,
                     payload_mutation_and_native_parity, sentinel_retains_no_wide_storage,
                     unsupported_witness_refuses_slot, timeout_progress_is_flushed):
        try:
            records.append({"name": function.__name__, "status": "PASS", "details": function()})
        except Exception as exc:
            records.append({"name": function.__name__, "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()})
    result = {"scope": "new zero-storage-version and telemetry CPU fixtures", "torch": torch.__version__,
              "results": records, "passed": all(r["status"] == "PASS" for r in records),
              "target_cuda_validated": False}
    Path(__file__).with_name('version_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(not result['passed'])

if __name__ == '__main__': main()
