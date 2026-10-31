"""Bounded route-only stdout telemetry; no tensors, hooks, files, or CUDA queries.

Each selected event is immediately flushed to the harness-owned train.log. It
survives a child SIGKILL once the write completed; a partial last line is ignored.
Counters describe Python route/enqueue observations, never kernel time or share.
"""
from collections import Counter
import json
import os
import sys


def emit_line(line):
    """Send only our own record to the same captured stdout, bypassing Rich.

    Leading newline separates a route from an in-progress carriage-return bar.
    On the normal fd1 path one PIPE_BUF-bounded write emits the whole line.
    Alternate/captured streams, unavailable fd/PIPE_BUF and overlong records use
    ordinary flushed stdout. Short writes finish only the unwritten suffix.
    No native output stream is replaced, redirected, closed or suppressed.
    """
    payload = ("\n" + line + "\n").encode("ascii")
    try:
        direct = sys.stdout.fileno() == 1 and len(payload) <= os.fpathconf(1, "PC_PIPE_BUF")
    except (AttributeError, OSError, ValueError):
        direct = False
    if direct:
        offset = 0
        try:
            while offset < len(payload):
                written = os.write(1, payload[offset:])
                if written <= 0:
                    raise OSError("stdout write made no progress")
                offset += written
            return
        except (AttributeError, OSError):
            # Never replay a written prefix (which would duplicate a route).
            sys.stdout.write(payload[offset:].decode("ascii"))
            sys.stdout.flush()
            return
    sys.stdout.write(payload.decode("ascii"))
    sys.stdout.flush()


class RouteStats:
    PREFIX = "SOUP_CE_V3_EVENT "
    MAX_KEYS = 128
    HOTPATH = frozenset({"loss_boundary", "triton_forward", "triton_backward"})
    EMIT = frozenset({"loss_boundary", "adapter_native_fallback", "native_forward_fallback",
                      "triton_forward", "triton_backward", "native_backward_fallback",
                      "model_loss_installed", "model_not_installed"})

    def __init__(self):
        self.events = Counter()
        self.reasons = Counter()
        self.denominators = Counter()
        self.boundaries = Counter()
        self.output_controls = Counter()
        self.mixed_row_totals = Counter()
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
        if event == "triton_forward" and values.get("route") == "mixed_rows":
            self.mixed_row_totals["calls"] += 1
            for key in ("native_supervised_rows", "safe_supervised_rows", "ignored_rows"):
                self.mixed_row_totals[key] += values[key]
            # Python counters only. A geometric record contains exact totals
            # through that call, not eventual counts after a killed process.
            values["cumulative_mixed_rows"] = dict(self.mixed_row_totals)
        # Count every event. Emit routine routes at 1,2,4,...; keep all selected
        # fallback/install metadata and the first marker before a potential JIT.
        count = self.events[event]
        selected = event in self.EMIT or (event == "triton_prepass_started" and count == 1)
        if selected and (event not in self.HOTPATH or count & (count - 1) == 0):
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
            emit_line(self.PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def snapshot(self):
        return {"events": dict(self.events), "reasons": dict(self.reasons),
                "denominators": dict(self.denominators),
                "loss_boundaries": dict(self.boundaries),
                "native_ignored_output_controls": dict(self.output_controls),
                "mixed_row_totals": dict(self.mixed_row_totals),
                "telemetry": {"schema": 1, "emitted_records": self.sequence,
                              "bounded_keys_per_counter": self.MAX_KEYS + 1,
                              "hotpath_emit_policy": "positive_powers_of_two",
                              "counter_policy": "count_every_event",
                              "kernel_completion_or_share_claim": False}}
