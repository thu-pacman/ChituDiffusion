"""Merge a serial and an overlap chrome trace into ONE small comparison file.

Extracts a single clean iteration of GPU kernels from each input trace, aligns
both to t=0, and places them in two separate process lanes ("serial" and
"overlap") so they can be compared side by side in perfetto/chrome://tracing.

Usage:
    python merge_traces.py SERIAL.json OVERLAP.json OUT.json
"""
import json
import sys


def load_gpu_kernels(path):
    """Return device-side kernel events (kernel / memcpy / memset) from a trace."""
    with open(path) as f:
        ev = json.load(f).get("traceEvents", [])
    ker = [
        e for e in ev
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    ker.sort(key=lambda e: e["ts"])
    return ker


def one_iteration(ker):
    """Slice out a single representative iteration.

    Each iter = projections + (gathers) + attention. We detect iteration
    boundaries by the gap before an attention/SDPA kernel, but a simpler robust
    heuristic: take the window between the 2nd and 3rd NCCL-or-attention anchor
    so warmup/teardown edges are excluded. Falls back to a fixed time slice.
    """
    if not ker:
        return []

    def is_anchor(e):
        n = e["name"].lower()
        return "nccl" in n or "sdpa" in n or "attention" in n or "flash" in n or "fmha" in n

    anchors = [i for i, e in enumerate(ker) if is_anchor(e)]
    t0 = ker[0]["ts"]
    # Prefer a mid-trace window to dodge warmup/cooldown.
    if len(anchors) >= 3:
        # window around the middle anchor: from previous lull to next lull
        mid = anchors[len(anchors) // 2]
        start_ts = ker[max(0, mid - 6)]["ts"]
        end_ts = ker[min(len(ker) - 1, mid + 6)]["ts"] + ker[min(len(ker) - 1, mid + 6)]["dur"]
    else:
        start_ts = t0
        end_ts = t0 + 4000.0  # 4ms fallback

    win = [e for e in ker if start_ts <= e["ts"] <= end_ts]
    base = min(e["ts"] for e in win)
    out = []
    for e in win:
        c = dict(e)
        c["ts"] = e["ts"] - base
        out.append(c)
    return out


def relane(kernels, pid, label):
    """Reassign all events to one process pid, keep per-stream tid lanes."""
    out = []
    # remap each distinct source stream (tid) to a stable sub-lane
    tids = sorted({e.get("tid") for e in kernels})
    for e in kernels:
        c = dict(e)
        c["pid"] = pid
        c["tid"] = f"{label}/stream{tids.index(e.get('tid'))}"
        out.append(c)
    return out


def main():
    serial_path, overlap_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    serial = one_iteration(load_gpu_kernels(serial_path))
    overlap = one_iteration(load_gpu_kernels(overlap_path))

    events = []
    events += relane(serial, pid="0_SERIAL", label="serial")
    events += relane(overlap, pid="1_OVERLAP", label="overlap")
    # process-name metadata so perfetto labels the lanes
    for pid, name in (("0_SERIAL", "SERIAL (no overlap)"), ("1_OVERLAP", "OVERLAP")):
        events.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                       "args": {"name": name}})

    def span(kers):
        if not kers:
            return 0.0
        return (max(e["ts"] + e["dur"] for e in kers) - min(e["ts"] for e in kers)) / 1000.0

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)

    print(f"serial  : {len(serial):3d} kernels, span {span(serial):.3f} ms")
    print(f"overlap : {len(overlap):3d} kernels, span {span(overlap):.3f} ms")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
