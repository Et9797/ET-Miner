"""Regenerate the two result figures of the paper from the reproduction artifacts.

Produces ``mining_campaign.pdf`` (frequent itemsets per threshold, log scale,
with wall-clock time per run on the original H100 and on the RTX 3090 of the
September 2026 re-execution) and ``k_distribution.pdf`` (itemsets per K of
the Opus run). Counts and RTX 3090 timings are read from the six-threshold
campaign JSON; the H100 timings are the original February 2026 measurements
quoted in the paper and are not available for the Base and Super thresholds.

Usage:
    python paper/figures/make_figures.py [--campaign JSON] [--out DIR]

Options:
    --campaign  campaign JSON written by experiment_full_campaign.py
                (default: runs/20260902T0000Z/phase3/2026_01/exp/
                experiment_full_campaign_20260902_091611.json)
    --out       output directory (default: the directory of this script)
    --png       also write PNG copies (200 dpi) next to the PDFs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = (
    ROOT / "runs/20260902T0000Z/phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json"
)
RUN_ORDER = ["Base", "Super", "Power", "Blitz", "Ultra", "Opus"]
SUPPORT_LABEL = {
    "Base": "0.1%",
    "Super": "0.01%",
    "Power": "0.001%",
    "Blitz": "0.0001%",
    "Ultra": "0.00002%",
    "Opus": "0.00001%",
}
H100_SECONDS = {"Power": 50.7, "Blitz": 2.0 * 60, "Ultra": 4.7 * 60, "Opus": 7.3 * 60}

BAR = "#3b82c4"
LINE_3090 = "#d1495b"
LINE_H100 = "#4d4d4d"


def load_campaign(path: Path) -> dict:
    """Return {run: {itemsets, max_k, seconds, k_distribution}} for the six thresholds."""
    data = json.loads(path.read_text())
    out = {}
    for name in RUN_ORDER:
        run = data["thresholds"][name]["runs"][0]
        out[name] = {
            "itemsets": int(run["itemsets"]),
            "max_k": int(run["max_k"]),
            "seconds": float(run["time_seconds"]),
            "k_distribution": {int(k): int(v) for k, v in run["k_distribution"].items()},
        }
    return out


def fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M".replace(".00M", "M")
    if n >= 1_000:
        return f"{n:,}"
    return str(n)


def save(fig, out: Path, stem: str, png: bool) -> None:
    """Write the PDF and, on request, a 200 dpi PNG copy."""
    fig.savefig(out / f"{stem}.pdf")
    if png:
        fig.savefig(out / f"{stem}.png", dpi=200)


def mining_campaign(camp: dict, out: Path, png: bool = False) -> None:
    """Bars of exhaustive itemset counts with per-run wall-clock lines on a twin axis."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    x = list(range(len(RUN_ORDER)))
    counts = [camp[r]["itemsets"] for r in RUN_ORDER]
    ax.bar(x, counts, width=0.55, color=BAR, label="Frequent itemsets (exhaustive)")
    ax.set_yscale("log")
    ax.set_ylim(1e2, 3e8)
    ax.set_ylabel("Frequent itemsets found")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{SUPPORT_LABEL[r]}" for r in RUN_ORDER])
    ax.set_xlabel("Run and minimum support threshold")
    for xi, r in zip(x, RUN_ORDER):
        ax.annotate(
            f"{fmt_count(camp[r]['itemsets'])}\nK={camp[r]['max_k']}",
            (xi, camp[r]["itemsets"]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax2 = ax.twinx()
    mins_3090 = [camp[r]["seconds"] / 60 for r in RUN_ORDER]
    ax2.plot(x, mins_3090, color=LINE_3090, marker="o", ms=4, lw=1.4, label="Time, RTX 3090 (Sept 2026)")
    h_x = [i for i, r in enumerate(RUN_ORDER) if r in H100_SECONDS]
    h_y = [H100_SECONDS[RUN_ORDER[i]] / 60 for i in h_x]
    ax2.plot(x[h_x[0]:], h_y, color=LINE_H100, marker="s", ms=4, lw=1.2, ls="--", label="Time, H100 (Feb 2026)")
    ax2.set_ylabel("Time per run (minutes)")
    ax2.set_ylim(0, 30)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, out, "mining_campaign", png)
    plt.close(fig)


def k_distribution(camp: dict, out: Path, png: bool = False) -> None:
    """Bar chart of the Opus K-distribution with the peak and the K=22 itemset annotated."""
    plt.rcParams.update({"font.size": 8})
    kd = camp["Opus"]["k_distribution"]
    total = camp["Opus"]["itemsets"]
    ks = sorted(kd)
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.bar(ks, [kd[k] for k in ks], width=0.75, color=BAR)
    ax.set_xlabel("Itemset size K")
    ax.set_ylabel("Number of frequent itemsets")
    ax.set_xticks(ks)
    ax.set_xlim(0.3, max(ks) + 0.7)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v else "0")
    )
    peak_k = max(ks, key=lambda k: kd[k])
    peak = kd[peak_k]
    ax.annotate(
        f"Peak: K={peak_k}, {peak / 1e6:.2f}M ({100 * peak / total:.2f}%)",
        (peak_k + 0.4, peak),
        textcoords="offset points",
        xytext=(60, -6),
        ha="left",
        va="top",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.6),
    )
    last = max(ks)
    ax.annotate(
        f"K={last}: {kd[last]} itemset\n(8 proteins)",
        (last, kd[last]),
        textcoords="offset points",
        xytext=(-6, 40),
        ha="right",
        va="bottom",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    ax.set_ylim(0, peak * 1.12)
    fig.tight_layout()
    save(fig, out, "k_distribution", png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    camp = load_campaign(args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    mining_campaign(camp, args.out, args.png)
    k_distribution(camp, args.out, args.png)
    for r in RUN_ORDER:
        print(f"{r:6s} {camp[r]['itemsets']:>12,} K={camp[r]['max_k']:2d} {camp[r]['seconds']:8.1f} s")
    print("wrote", args.out / "mining_campaign.pdf", "and", args.out / "k_distribution.pdf")


if __name__ == "__main__":
    main()
