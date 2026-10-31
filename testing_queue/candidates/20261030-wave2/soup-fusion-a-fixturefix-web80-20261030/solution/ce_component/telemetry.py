"""Bounded route-only stdout telemetry; no tensors, hooks, files, or CUDA queries.

Each selected event is immediately flushed to the harness-owned train.log. It
survives a child SIGKILL once the write completed; a partial last line is ignored.
Counters describe Python route/enqueue observations, never kernel time or share.
"""
from collections import Counter
import json


class RouteStats:
    PREFIX = "SOUP_CE_V3_EVENT "
    MAX_KEYS = 128
    EMIT = frozenset({"loss_boundary", "adapter_native_fallback", "native_forward_fallback",
                      "triton_forward", "triton_backward", "native_backward_fallback",
                      "model_loss_installed", "model_not_installed"})

    def __init__(self):
        self.events = Counter()
        self.reasons = Counter()
        self.denominators = Counter()
        self.boundaries = Counter()
        self.output_controls = Counter()
        self.sequence = 0

    def observe(self, counter, key):
        key = str(key)[:240]
        values = getattr(self, counter)
        if key not in values and len(values) >= self.MAX_KEYS:
            key = "<other>"
        values[key] += 1

    def record(self, event, **values):
        self.observe("events", event)
        if "reason" in values:
            self.observe("reasons", values["reason"])
        if event == "denominator_validated":
            self.observe("denominators", f"{values['kind']}:{values['count']}")
        # A start marker before the first JIT can explain a compilation timeout.
        # Subsequent loss_boundary records identify later in-flight calls.
        if event in self.EMIT or (event == "triton_prepass_started" and self.events[event] == 1):
            self.sequence += 1
            record = {"schema": 1, "sequence": self.sequence, "event": event,
                      "event_count": self.events[event],
                      "loss_calls": self.events.get("loss_boundary", 0),
                      "own_forward_routes": self.events.get("triton_forward", 0),
                      "own_backward_enqueues": self.events.get("triton_backward", 0),
                      "native_adapter_routes": self.events.get("adapter_native_fallback", 0),
                      "native_core_routes": self.events.get("native_forward_fallback", 0),
                      "native_backward_routes": self.events.get("native_backward_fallback", 0),
                      "metadata": values}
            print(self.PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)

    def snapshot(self):
        return {"events": dict(self.events), "reasons": dict(self.reasons),
                "denominators": dict(self.denominators),
                "loss_boundaries": dict(self.boundaries),
                "native_ignored_output_controls": dict(self.output_controls),
                "telemetry": {"schema": 1, "emitted_records": self.sequence,
                              "bounded_keys_per_counter": self.MAX_KEYS + 1,
                              "kernel_completion_or_share_claim": False}}
