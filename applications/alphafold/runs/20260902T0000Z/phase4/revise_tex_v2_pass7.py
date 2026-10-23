"""V2 pass 7: fixes from the final re-review of the pinned-run manuscript.

Exact-string replacements; each anchor must occur exactly once.

Usage:
    python revise_tex_v2_pass7.py <path-to-tex>
"""
import sys

path = sys.argv[1]
tex = open(path, encoding="utf-8").read()
orig = tex


def rep(old, new):
    global tex
    n = tex.count(old)
    assert n == 1, f"expected 1 occurrence, found {n}: {old[:90]!r}"
    tex = tex.replace(old, new)


# D1: early exit belongs to the shared-memory kernel that ran, not the legacy ballot kernel
rep(
    r"with a warp-level early exit that skips the remaining features of a 32-word window (2,048 proteins) once the running AND is zero in all of its lanes.",
    r"with an early exit that skips a 32-word tile (2,048 proteins) as soon as the staged prefix AND is zero across the whole tile.",
)
rep(r"Early termination    & No & Unknown & \textbf{Warp-level} (window AND $= 0$) \\", r"Early termination    & No & Unknown & \textbf{Tile-level} (tile AND $= 0$) \\")

# D2: set identity was verified at Base and Super only
rep(
    r"The streaming SON path, run at the Base, Super and Power thresholds, returned identical itemsets (Section~\ref{sec:results}).}",
    r"The streaming SON path, run at the Base, Super and Power thresholds, returned the same itemset counts (set identity verified at Base and Super; Section~\ref{sec:results}).}",
)
rep(
    r"at identical support (0.001\%) the released SON implementation returns exactly the exhaustive itemset table, but $99\times$ more slowly",
    r"at identical support (0.001\%) the released SON implementation returns the same itemset count and $K$-distribution as the exhaustive run, but $99\times$ more slowly",
)

# D3: name the GPU configuration of the two SON runs at Base and Super
rep(
    r"the permutation null model and the per-level itemset exports used both GPUs with the row-split path.",
    r"the permutation null model and the per-level itemset exports used both GPUs with the row-split path. The two streaming runs at the Base and Super thresholds were not pinned, but the streaming path counts on one device only.",
)

# D4: intermediate-K supports that the run can source
rep(
    r"\paragraph{$K{=}13$: Helicase-Recombinase DNA Repair Module} (${\sim}11{,}000$ proteins\footnote{Supporting-protein counts for the highlighted intermediate-$K$ patterns are order-of-magnitude estimates read from the per-$K$ itemset tables of the Blitz run.}).",
    r"\paragraph{$K{=}13$: Helicase-Recombinase DNA Repair Module} (itemsets at this level are shared by up to 11,521 proteins\footnote{Supporting-protein counts are read from the per-$K$ itemset tables of the Blitz run. Where the pattern is identified by named domains, the count is the support of the matching itemsets; the $K{=}13$ and $K{=}11$ descriptions name no domain identifiers, so the maximum support at that level is given instead.}).",
)
rep(
    r"\paragraph{$K{=}12$: Bacterial Cell Wall Synthase} (${\sim}10{,}500$ proteins).",
    r"\paragraph{$K{=}12$: Bacterial Cell Wall Synthase} (up to 16,185 proteins; 111 itemsets at this level contain both domains).",
)
rep(
    r"\paragraph{$K{=}11$: AAA+ ATPase Proteasome Complex} (${\sim}16{,}000$ proteins).",
    r"\paragraph{$K{=}11$: AAA+ ATPase Proteasome Complex} (up to 28,913 proteins).",
)

# D7: memory-aware chunk sizing belongs to the row-split path
rep(
    r"For candidate spaces exceeding a single device, ET-miner can distribute work across multiple GPUs with memory-aware chunk sizing that adapts to the installed hardware.",
    r"For candidate spaces exceeding a single device, ET-miner can distribute work across multiple GPUs either by candidate index or by row split, the latter with chunk sizing measured from the installed VRAM.",
)

# D8: one quantity, one value
rep(r"versus ${\sim}10$\,GB for the equivalent bit-packed dense bitmap of the 76.9M mining subset", r"versus 9.6\,GB for the equivalent bit-packed dense bitmap of the 76.9M mining subset")

# D9: the scale gap contradicted Table 6
rep(
    r"but all existing GPU implementations were evaluated on datasets of at most a few million transactions, three orders of magnitude smaller than our proteome dataset.",
    r"but all existing GPU implementations were evaluated on datasets of at most a few million real transactions, one to two orders of magnitude smaller than our proteome dataset.",
)
rep(
    r"and (2)~they were evaluated on datasets of at most a few million transactions, three orders of magnitude smaller than our proteome dataset.",
    r"and (2)~they were evaluated on datasets of at most a few million real transactions, one to two orders of magnitude smaller than our proteome dataset.",
)

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("ok pass7:", len(orig), "->", len(tex), "chars")
