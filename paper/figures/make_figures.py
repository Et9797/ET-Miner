"""Regenerate the result and pipeline figures of the paper from the run artifacts.

Produces ``mining_campaign.pdf`` (frequent itemsets per threshold, log scale,
with wall-clock time per run on a single RTX 3090, pinned with CUDA_VISIBLE_DEVICES=0), ``k_distribution.pdf``
(itemsets per K of the Opus run) and ``architecture.pdf`` (the data path from
the UniProt flat file to the frequent itemsets, annotated with the measured
sizes). Counts and timings are read from the six-threshold campaign JSON; the
sizes in the architecture figure are the values recorded in RESULTS.md
(rows U-007, M-001, X-010, X-011, X-014, X-018, X-023, P-002, P-003).

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
    ROOT / "runs/20260902T0000Z/phase3/2026_01/single_gpu/exp/experiment_full_campaign_20260902_200731.json"
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
BAR = "#3b82c4"
LINE_3090 = "#d1495b"
BOX_FACE = "#eef3fa"
BOX_EDGE = "#3b82c4"


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
    ax2.plot(x, mins_3090, color=LINE_3090, marker="o", ms=4, lw=1.4, label="Time per run, single RTX 3090")
    ax2.set_ylabel("Time per run (minutes)")
    ax2.set_ylim(0, 65)
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


def architecture(out: Path, png: bool = False) -> None:
    """Pipeline diagram: six stages with the measured sizes, left to right."""
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({"font.size": 7})
    stages = [
        ("Inputs", "UniProt TrEMBL 2026_01\nflat file, 149.8 GiB gz\nAlphaFold DB metadata\n214.7M rows"),
        ("Feature extraction\n(CPU)", "205.6M transactions\n76.9M multi-feature\nParquet, 0.89 GB"),
        ("Sparse matrix\n(CPU)", "316M non-zeros\n5.1 GB coordinate form\n(dense: 206 GB, 40.7x)"),
        ("GPU bitvectors\n(VRAM, resident)", "1,002 x 1,201,422 words\n9.6 GB\n1 x RTX 3090 (24 GB)"),
        ("GPU-resident\nApriori", "K = 1 ... 22\nfused count + filter\nscalar readback per level"),
        ("Frequent\nitemsets", "26.8M itemsets\nK <= 22\nmin_count 8"),
    ]
    arrows = ["", "", "H2D once:\n3.1 GB CSR arrays", "", ""]
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    w, h, gap, y0 = 1.02, 1.4, 0.11, 0.45
    for i, (title, body) in enumerate(stages):
        x0 = 0.08 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=BOX_FACE, ec=BOX_EDGE, lw=0.9))
        ax.text(x0 + w / 2, y0 + h - 0.08, title, ha="center", va="top", fontsize=6.6, weight="bold")
        ax.text(x0 + w / 2, y0 + 0.08, body, ha="center", va="bottom", fontsize=5.7, linespacing=1.3)
        if i < len(stages) - 1:
            ax.annotate("", (x0 + w + gap, y0 + h / 2), (x0 + w, y0 + h / 2),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color=BOX_EDGE))
            if arrows[i]:
                ax.text(x0 + w + gap / 2, y0 + h + 0.02, arrows[i], ha="center", va="bottom", fontsize=5.4)
    ax.text(0.08, 0.16, "CPU side", fontsize=6.5, color="#555555")
    ax.text(0.08 + 3 * (w + gap), 0.16, "GPU side (single device; bitvectors never leave VRAM)",
            fontsize=6.5, color="#555555")
    fig.tight_layout(pad=0.2)
    save(fig, out, "architecture", png)
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
    architecture(args.out, args.png)
    for r in RUN_ORDER:
        print(f"{r:6s} {camp[r]['itemsets']:>12,} K={camp[r]['max_k']:2d} {camp[r]['seconds']:8.1f} s")
    print("wrote mining_campaign.pdf, k_distribution.pdf and architecture.pdf in", args.out)


if __name__ == "__main__":
    main()
