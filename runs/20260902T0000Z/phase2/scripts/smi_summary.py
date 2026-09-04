"""Summarize an nvidia-smi sampling CSV per GPU (max/mean utilization, max memory, sample count).

Usage:
    python smi_summary.py <csv> [...]

The CSV is the output of
``nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5``.
"""
import csv, sys


def summarize(path):
    per = {}
    with open(path) as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:]:
        if len(row) < 4:
            continue
        idx = row[1].strip()
        try:
            util = float(row[2].strip().split()[0])
            mem = float(row[3].strip().split()[0])
        except ValueError:
            continue
        d = per.setdefault(idx, {"n": 0, "util_max": 0.0, "util_sum": 0.0, "mem_max": 0.0, "busy": 0})
        d["n"] += 1
        d["util_max"] = max(d["util_max"], util)
        d["util_sum"] += util
        d["mem_max"] = max(d["mem_max"], mem)
        d["busy"] += util > 0
    first, last = rows[1][0].strip(), rows[-1][0].strip()
    print(f"{path}: {first} .. {last}")
    for idx, d in sorted(per.items()):
        print(f"  GPU {idx}: samples={d['n']} util_max={d['util_max']:.0f}% util_mean={d['util_sum'] / d['n']:.1f}% "
              f"samples_with_util>0={d['busy']} mem_max={d['mem_max']:.0f} MiB")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
