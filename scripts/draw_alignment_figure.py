"""Draw InATTo's (b) Alignment figure in the same style as FACE's example.

Renders to /home/koohy/cikm/InATTo/figures/inatto_alignment.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D


def rounded_box(ax, x, y, w, h, label, fc, ec="#444", fontsize=10,
                fontweight="normal", txt_color="black", boxstyle="round,pad=0.02",
                lw=1.2):
    box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=txt_color)


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.5, style="-|>", alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw, alpha=alpha)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 4.6), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(98.5, 33.6, "(b) Alignment", ha="right", va="center",
            fontsize=13, fontweight="bold")

    # ============================== TOP — Descriptor path ==============================
    # Prompt box
    rounded_box(ax, 1.0, 27.5, 16.5, 3.2,
                '"This toy can be described by these aspects:"',
                fc="#FFE9C2", fontsize=8.0, fontweight="bold")
    ax.text(9.25, 26.0, "Prompt", ha="center", va="top", fontsize=8.8, style="italic")

    # Descriptor codeword boxes (8 aspects, with some inactive)
    cw_y = 27.6
    cw_h = 3.0
    cw_w = 3.0
    gap  = 0.6
    cw_x0 = 21.5
    active_pattern = [True, True, False, True, True, True, False, True]  # 6/8 active demo
    aspect_labels  = ["c₁", "c₂", "c₃", "c₄", "c₅", "c₆", "c₇", "c₈"]
    for i, (lab, act) in enumerate(zip(aspect_labels, active_pattern)):
        x = cw_x0 + i * (cw_w + gap)
        if act:
            rounded_box(ax, x, cw_y, cw_w, cw_h, lab, fc="#CFE9FF",
                        ec="#2566A8", fontsize=10, fontweight="bold")
        else:
            # inactive (struck-through, ghosted)
            rounded_box(ax, x, cw_y, cw_w, cw_h, lab, fc="#F4F4F4",
                        ec="#BBBBBB", fontsize=10, txt_color="#BBBBBB")
            ax.plot([x+0.3, x+cw_w-0.3], [cw_y+0.4, cw_y+cw_h-0.4],
                    color="#BB6666", lw=1.4)
    ax.text(cw_x0 + 4*(cw_w+gap) - gap/2, 26.0, "Descriptors (active codewords; $\\mathbf{c}_k^{(l)}$)",
            ha="center", va="top", fontsize=8.8, style="italic")
    # legend
    ax.text(cw_x0 + 8*(cw_w+gap)-cw_w + 1.3, cw_y + cw_h + 0.7,
            "✗ inactive (mask=0)", color="#777", ha="left", va="bottom", fontsize=7.4)

    # arrow from Prompt + Descriptors → MiniLM
    minilm_top_x, minilm_top_y, minilm_top_w, minilm_top_h = 56.5, 26.0, 12.0, 6.0
    arrow(ax, 17.5, 29.1, minilm_top_x, 29.0)
    arrow(ax, cw_x0 + 8*(cw_w+gap)-gap, 29.1, minilm_top_x, 29.0)

    # MiniLM box (top)
    rounded_box(ax, minilm_top_x, minilm_top_y, minilm_top_w, minilm_top_h,
                "MiniLM\n(frozen, inputs-embeds)",
                fc="#E8F1FA", ec="#2566A8", fontsize=9.5, fontweight="bold")
    # snow flake (frozen) icon
    ax.text(minilm_top_x + 0.8, minilm_top_y + minilm_top_h - 0.8, "❄",
            fontsize=12, color="#2566A8", ha="left", va="top")

    # token pool / norm
    arrow(ax, minilm_top_x + minilm_top_w, 29.0, minilm_top_x + minilm_top_w + 3.5, 29.0)
    ax.text(minilm_top_x + minilm_top_w + 1.8, 30.6, "mean-pool",
            ha="center", va="bottom", fontsize=7.6)
    ax.text(minilm_top_x + minilm_top_w + 1.8, 27.4, "+ L2 norm",
            ha="center", va="top", fontsize=7.6)

    # h_d box
    hd_x, hd_y, hd_w, hd_h = 73.5, 27.4, 5.6, 3.2
    rounded_box(ax, hd_x, hd_y, hd_w, hd_h, "$\\mathbf{h}_d$",
                fc="#9DC8F2", ec="#1B4C7F", fontsize=12, fontweight="bold")

    # ============================== BOTTOM — Profile path ==============================
    # Raw metadata box
    rounded_box(ax, 1.0, 6.0, 16.5, 4.2,
                "Title; brand;\ncategories;\ndescription; price; rank",
                fc="#FFEDED", ec="#A85B5B", fontsize=7.4)
    ax.text(9.25, 4.6, "Raw item text", ha="center", va="top", fontsize=8.8, style="italic")

    # Arrow → GPT-4o-mini
    arrow(ax, 17.5, 8.1, 22.0, 8.1)

    # GPT-4o-mini box
    rounded_box(ax, 22.0, 6.0, 12.0, 4.2,
                "GPT-4o-mini\n(profile generator)",
                fc="#E7F6E1", ec="#3A8A2C", fontsize=8.8, fontweight="bold")

    # Arrow → profile sentence
    arrow(ax, 34.0, 8.1, 39.0, 8.1)

    # Profile sentence box
    rounded_box(ax, 39.0, 6.0, 15.5, 4.2,
                '"The toy attributes\nare xxx."',
                fc="#FFFAE0", ec="#8C7B22", fontsize=8.0, fontweight="bold")
    ax.text(46.75, 4.6, "Generated profile",
            ha="center", va="top", fontsize=8.8, style="italic")

    # Arrow → MiniLM (bottom)
    arrow(ax, 54.5, 8.1, minilm_top_x, 8.1)
    rounded_box(ax, minilm_top_x, 5.0, minilm_top_w, 6.0,
                "MiniLM\n(frozen)",
                fc="#E8F1FA", ec="#2566A8", fontsize=9.5, fontweight="bold")
    ax.text(minilm_top_x + 0.8, 5.0 + 6.0 - 0.8, "❄",
            fontsize=12, color="#2566A8", ha="left", va="top")

    # Arrow → h_raw
    arrow(ax, minilm_top_x + minilm_top_w, 8.0, hd_x, 8.0)
    rounded_box(ax, hd_x, 6.4, hd_w, 3.2, "$\\mathbf{h}_{raw}$",
                fc="#F2C2C2", ec="#7F1B1B", fontsize=12, fontweight="bold")

    # ============================== CENTER — Contrastive Alignment ==============================
    # Connect h_d and h_raw to a matrix-like panel
    ca_cx, ca_cy = 92.0, 18.0
    # mini cosine matrix (3x3 sample)
    n = 3
    cell = 1.8
    mat_x = ca_cx - cell*n/2
    mat_y = ca_cy - cell*n/2
    for i in range(n):
        for j in range(n):
            face = "#9DC8F2" if i == j else "#FFFFFF"
            ax.add_patch(Rectangle((mat_x + j*cell, mat_y + (n-1-i)*cell),
                                    cell, cell, lw=0.8, ec="#444", fc=face))
            if i == j:
                ax.text(mat_x + j*cell + cell/2, mat_y + (n-1-i)*cell + cell/2,
                        "✓", ha="center", va="center", fontsize=10, color="#1B4C7F")
    ax.text(ca_cx, mat_y + n*cell + 1.0, "Contrastive\nAlignment",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(ca_cx, mat_y - 0.6, "(in-batch InfoNCE)",
            ha="center", va="top", fontsize=8, style="italic")

    # arrows from h_d, h_raw to matrix
    arrow(ax, hd_x + hd_w, hd_y + hd_h/2, mat_x - 0.3, ca_cy + cell*0.8, style="->")
    arrow(ax, hd_x + hd_w, 6.4 + 3.2/2, mat_x - 0.3, ca_cy - cell*0.8, style="->")

    # ============================== Outside title ==============================
    fig.text(0.5, 0.965, "InATTo — (b) Alignment",
              ha="center", fontsize=12.5, fontweight="bold")

    out = Path("/home/koohy/cikm/InATTo/figures") / "inatto_alignment.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
