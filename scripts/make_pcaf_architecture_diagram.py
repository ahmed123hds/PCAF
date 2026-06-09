#!/usr/bin/env python3
"""Generate the PCAF architecture diagram used in the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "pcaf_architecture_diagram.png"


def add_box(ax, xy, width, height, text, color, fontsize=12, lw=1.2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=color,
        edgecolor="#1d1d1f",
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return box


def add_arrow(ax, start, end, text=None, offset=(0, 0), color="#1d1d1f"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if text:
        mid = ((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1])
        ax.text(mid[0], mid[1], text, ha="center", va="center", fontsize=10, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    teal = "#5ec3bc"
    indigo = "#8897d3"
    amber = "#f6b24b"
    light = "#f1f2f4"

    ax.text(0.5, 0.955, "PCAF Architecture", ha="center", va="center", fontsize=17, weight="bold")
    ax.text(0.5, 0.915, r"Output: $\log p(y \mid x_{\leq T})$", ha="center", va="center", fontsize=11)

    add_box(
        ax,
        (0.34, 0.79),
        0.32,
        0.075,
        "Learned Cache Gate\n"
        r"$p(y)=(1-\lambda)p_{\rm param}+\lambda p_{\rm cache}$" "\n"
        r"$\lambda=\sigma(g_\theta(h_T))$",
        amber,
        fontsize=10.5,
    )

    add_box(ax, (0.14, 0.65), 0.29, 0.055, "MLP Head", teal, fontsize=12)
    add_box(
        ax,
        (0.14, 0.45),
        0.29,
        0.13,
        "Causal Depthwise Conv Blocks\n"
        r"LayerNorm $\rightarrow$ Conv1D $\rightarrow$ GELU" "\n"
        r"MLP $\rightarrow$ Residual",
        teal,
        fontsize=10.5,
    )
    add_box(ax, (0.14, 0.33), 0.29, 0.06, "Token Embedding", teal, fontsize=12)

    token_y = 0.215
    for i, x in enumerate([0.15, 0.225, 0.30, 0.375, 0.455]):
        label = [r"$x_1$", r"$x_2$", r"$x_3$", r"$\cdots$", r"$x_T$"][i]
        if label == r"$\cdots$":
            ax.text(x, token_y + 0.03, label, ha="center", va="center", fontsize=14)
            continue
        token = FancyBboxPatch(
            (x - 0.027, token_y),
            0.054,
            0.04,
            boxstyle="round,pad=0.006,rounding_size=0.025",
            facecolor=light,
            edgecolor="#1d1d1f",
            linewidth=1.0,
        )
        ax.add_patch(token)
        ax.text(x, token_y + 0.02, label, ha="center", va="center", fontsize=11)

    add_box(
        ax,
        (0.58, 0.62),
        0.28,
        0.075,
        "Key-Query Scoring\n"
        r"$\alpha_i=\mathrm{softmax}(q_T^\top k_i/\sqrt{d}+\rho_i)$",
        indigo,
        fontsize=10.2,
    )
    add_box(
        ax,
        (0.58, 0.515),
        0.28,
        0.065,
        "Bounded Bucket Lookup\n"
        r"select top-$K$ candidates $\mathcal{C}(T)$",
        indigo,
        fontsize=10.5,
    )
    add_box(
        ax,
        (0.58, 0.405),
        0.28,
        0.065,
        "Hash Address Computation\n" r"$a_i=H_n(x_{i-n+1:i})\,\mathrm{mod}\,B$",
        indigo,
        fontsize=10.4,
    )

    bucket = FancyBboxPatch(
        (0.58, 0.27),
        0.28,
        0.08,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=indigo,
        edgecolor="#1d1d1f",
        linewidth=1.2,
    )
    ax.add_patch(bucket)
    for i in range(9):
        ax.add_patch(Rectangle((0.605 + 0.025 * i, 0.292), 0.023, 0.024, facecolor="#d9deef", edgecolor="#333333", lw=0.7))
    for x, y in [(0.61, 0.327), (0.635, 0.338), (0.68, 0.332), (0.73, 0.34), (0.755, 0.329)]:
        ax.add_patch(Rectangle((x, y), 0.023, 0.014, facecolor="#7687c3", edgecolor="#333333", lw=0.6))
    ax.text(0.72, 0.235, "Hash Buckets (B buckets)", ha="center", va="center", fontsize=11)

    add_arrow(ax, (0.285, token_y + 0.055), (0.285, 0.33))
    add_arrow(ax, (0.285, 0.39), (0.285, 0.45), r"$h_{1:T}$", offset=(0.045, 0.0))
    add_arrow(ax, (0.285, 0.58), (0.285, 0.65), r"$h_{1:T}$", offset=(0.045, 0.0))
    add_arrow(ax, (0.285, 0.705), (0.39, 0.79), r"$p_{\rm param}$", offset=(-0.05, 0.03))

    add_arrow(ax, (0.43, 0.515), (0.58, 0.55))
    add_arrow(ax, (0.43, 0.48), (0.58, 0.435))
    add_arrow(ax, (0.72, 0.35), (0.72, 0.405))
    add_arrow(ax, (0.72, 0.47), (0.72, 0.515))
    add_arrow(ax, (0.72, 0.58), (0.72, 0.62))
    add_arrow(ax, (0.72, 0.695), (0.63, 0.79), r"$p_{\rm cache}$", offset=(0.05, 0.02))
    add_arrow(ax, (0.5, 0.865), (0.5, 0.905))

    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.18, 1.0)
    fig.tight_layout(pad=0.1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
