"""Draw InATTo's (c) Recommendation figure (Stage 2 RPG) in FACE-style."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


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
    fig, ax = plt.subplots(figsize=(14.5, 5.0), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(98.5, 37.6, "(c) Recommendation", ha="right", va="center",
            fontsize=13, fontweight="bold")

    # ================== USER HISTORY ==================
    ax.text(5.0, 32.5, "User history", ha="center", va="bottom",
            fontsize=9, style="italic", fontweight="bold")
    hist_y = 22.0
    hist_h = 8.5
    items = [("item_1", "#FFE9C2"), ("item_2", "#FFF6D6"),
             ("…", "#F8F8F8"),    ("item_S", "#FFE0E0")]
    for i, (lab, fc) in enumerate(items):
        x = 1.5 + i * 2.4
        rounded_box(ax, x, hist_y, 2.0, hist_h, lab, fc=fc,
                    fontsize=7.4, fontweight="bold")
    # show identifier token grid hint
    ax.text(5.0, hist_y - 1.3,
            "each item → identifier\n(variable-length tokens)",
            ha="center", va="top", fontsize=7.0, color="#555", style="italic")

    arrow(ax, 11.2, 26.0, 14.6, 26.0)

    # ================== ITEM-LEVEL POOL ==================
    pool_x, pool_y, pool_w, pool_h = 14.6, 19.0, 16.0, 14.0
    rounded_box(ax, pool_x, pool_y, pool_w, pool_h,
                "",
                fc="#E7F6E1", ec="#3A8A2C", fontsize=9.4)
    ax.text(pool_x + pool_w/2, pool_y + pool_h - 1.1, "Item-level pool",
            ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1B5E1B")

    # mini grid (K × L_max) inside the pool box, with mask shading
    grid_x = pool_x + 1.0
    grid_y = pool_y + 1.0
    cell_w = 0.95
    cell_h = 0.95
    K, L = 8, 4
    # active pattern per aspect (number of active levels)
    active_per_aspect = [4, 3, 2, 1, 3, 2, 4, 2]
    for k in range(K):
        for l in range(L):
            x = grid_x + k * (cell_w + 0.06)
            y = grid_y + l * (cell_h + 0.06)
            if l < active_per_aspect[k]:
                ax.add_patch(Rectangle((x, y), cell_w, cell_h,
                                        lw=0.5, ec="#2A7522", fc="#A6D8A0"))
            else:
                ax.add_patch(Rectangle((x, y), cell_w, cell_h,
                                        lw=0.5, ec="#BBBBBB", fc="#F2F2F2"))
    ax.text(grid_x - 0.4, grid_y + L*cell_h*0.5, "L",
            ha="right", va="center", fontsize=7, color="#555")
    ax.text(grid_x + K*(cell_w+0.06)*0.5 - 0.3, grid_y - 0.5, "K aspects",
            ha="center", va="top", fontsize=7, color="#555")
    # description text inside box
    ax.text(pool_x + pool_w/2, pool_y + 6.4,
            "mean over active codewords",
            ha="center", va="top", fontsize=7.6, color="#1B5E1B")
    ax.text(pool_x + pool_w/2, pool_y + 5.0,
            r"item_vec = $\sum_p m_p \cdot e_p / \sum_p m_p$",
            ha="center", va="top", fontsize=7.4, color="#222")

    arrow(ax, pool_x + pool_w, 26.0, pool_x + pool_w + 4.0, 26.0)

    # ================== GPT-2 BACKBONE ==================
    gpt_x, gpt_y, gpt_w, gpt_h = 34.6, 19.0, 14.0, 14.0
    rounded_box(ax, gpt_x, gpt_y, gpt_w, gpt_h,
                "GPT-2\n(2 layers, 4 heads,\nd=256, causal SA)",
                fc="#E8F1FA", ec="#2566A8", fontsize=9.5, fontweight="bold")
    ax.text(gpt_x + gpt_w/2, gpt_y - 1.0,
            "over item sequence",
            ha="center", va="top", fontsize=7.6, style="italic", color="#555")

    arrow(ax, gpt_x + gpt_w, 26.0, gpt_x + gpt_w + 4.0, 26.0)
    ax.text(gpt_x + gpt_w + 2.0, 27.0, r"$\mathbf{h}_{last}$",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ================== N PARALLEL HEADS ==================
    head_x, head_y, head_w, head_h = 52.6, 18.5, 13.0, 14.8
    rounded_box(ax, head_x, head_y, head_w, head_h,
                "",
                fc="#FFF3E5", ec="#C97818", fontsize=9.5)
    ax.text(head_x + head_w/2, head_y + head_h - 1.0,
            "N = K·L = 32 parallel\nResBlock heads",
            ha="center", va="top", fontsize=8.4, fontweight="bold", color="#7A4A0F")
    # mini "P" stack inside the head box
    s_x = head_x + 1.6
    s_y = head_y + 1.5
    for i in range(6):
        ax.add_patch(Rectangle((s_x + i*0.18, s_y + i*0.18), head_w-3.3, 1.2,
                                lw=0.6, ec="#A86420", fc="#FFD89A", alpha=0.7))
    ax.text(head_x + head_w/2, s_y + 7.0,
            "p = 1, 2, …, 32",
            ha="center", va="top", fontsize=7.6, color="#5A330B")
    ax.text(head_x + head_w/2, s_y + 5.2,
            r"state$_p$ = head$_p$($\mathbf{h}_{last}$)",
            ha="center", va="top", fontsize=7.4, color="#222")
    ax.text(head_x + head_w/2, s_y + 3.8,
            "★ no autoregressive\n  (all positions at once)",
            ha="center", va="top", fontsize=7.0, style="italic", color="#7A4A0F")

    arrow(ax, head_x + head_w, 26.0, head_x + head_w + 3.0, 26.0)

    # ================== LOGITS PANEL ==================
    log_x, log_y, log_w, log_h = 68.6, 18.5, 13.0, 14.8
    rounded_box(ax, log_x, log_y, log_w, log_h,
                "",
                fc="#F6E1F0", ec="#A8367A", fontsize=9.5)
    ax.text(log_x + log_w/2, log_y + log_h - 1.0,
            "Position-wise logits",
            ha="center", va="top", fontsize=9.0, fontweight="bold", color="#5A1F46")
    # mini matrix P × V
    m_x = log_x + 1.6
    m_y = log_y + 5.0
    m_w = log_w - 3.0
    m_h = 4.6
    ax.add_patch(Rectangle((m_x, m_y), m_w, m_h,
                            lw=0.8, ec="#5A1F46", fc="#FFFFFF"))
    # gradient stripes (P rows)
    for i in range(8):
        ax.add_patch(Rectangle((m_x, m_y + i*(m_h/8)),
                                m_w, m_h/8 - 0.02,
                                lw=0, fc=(0.9 - i*0.07, 0.6, 0.85 - i*0.06),
                                alpha=0.45))
    ax.text(log_x + log_w/2, m_y - 0.5,
            r"$\log p$ = log-softmax(logits)",
            ha="center", va="top", fontsize=7.4, color="#222")
    ax.text(log_x + log_w/2, m_y - 1.9,
            "(B, P=32, |V|=9340)",
            ha="center", va="top", fontsize=7.0, style="italic", color="#555")
    ax.text(m_x - 0.5, m_y + m_h/2, "P", ha="right", va="center",
            fontsize=8, color="#5A1F46")
    ax.text(m_x + m_w/2, m_y + m_h + 0.4, "vocab",
            ha="center", va="bottom", fontsize=7, color="#5A1F46")

    arrow(ax, log_x + log_w, 26.0, log_x + log_w + 3.0, 26.0)

    # ================== RANKING / TOP-K ==================
    rk_x, rk_y, rk_w, rk_h = 84.6, 18.5, 14.0, 14.8
    rounded_box(ax, rk_x, rk_y, rk_w, rk_h, "",
                fc="#FFEDED", ec="#A85B5B", fontsize=9.5)
    ax.text(rk_x + rk_w/2, rk_y + rk_h - 1.0,
            "Top-K Ranking",
            ha="center", va="top", fontsize=9.0, fontweight="bold", color="#7A2C2C")
    # formula
    ax.text(rk_x + rk_w/2, rk_y + 9.6,
            r"score$_i$ = $\frac{1}{|A_i|}$",
            ha="center", va="top", fontsize=8.6, color="#222")
    ax.text(rk_x + rk_w/2, rk_y + 8.0,
            r"$\sum_{p \in A_i} \log p_{i,p}$",
            ha="center", va="top", fontsize=8.6, color="#222")
    ax.text(rk_x + rk_w/2, rk_y + 6.0,
            r"$A_i$ = active positions of $i$",
            ha="center", va="top", fontsize=7.2, style="italic", color="#555")
    # top-K list (mock)
    ax.text(rk_x + rk_w/2, rk_y + 4.0,
            "1. item₁₂\n2. item₃₄\n3. item₂₁",
            ha="center", va="top", fontsize=7.6, color="#2A2A2A", fontweight="bold")

    # ====== bottom note ======
    fig.text(0.5, 0.045,
              "★ Variable-depth signal enters 3× — (1) pool denominator, "
              "(2) per-position loss mask, (3) ranking score $|A_i|$.",
              ha="center", fontsize=8.6, color="#444", style="italic")

    fig.text(0.5, 0.97, "InATTo — (c) Recommendation",
              ha="center", fontsize=12.5, fontweight="bold")

    out = Path("/home/koohy/cikm/InATTo/figures") / "inatto_recommendation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
