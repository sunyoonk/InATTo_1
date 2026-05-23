"""Draw InATTo's full pipeline: (b) Alignment → bridge → (c) Recommendation.

Reflects the latest design:
  - Variable-length descriptor (active codewords as soft tokens)
  - User-side u2i alignment (target = mean h_raw_item over history)
  - Item-side cross-text alignment (target = GPT-profile)
  - Stage 2 user_fuse=prepend (single mean-pooled user token at front)
"""
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


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.4, style="-|>", alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=12,
                        color=color, lw=lw, alpha=alpha)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    # ────────────────────────────────────────────────────────────────────
    # (b) ALIGNMENT — top half (y: 60–98)
    # ────────────────────────────────────────────────────────────────────
    ax.text(1.5, 96.0, "(b) Alignment — dual, variable-length",
            ha="left", va="center", fontsize=13, fontweight="bold")

    # === ITEM TOP TRUNK ===
    ax.text(1.5, 92.0, "ITEM", ha="left", va="center", fontsize=10,
            fontweight="bold", color="#1B4C7F")

    # Item prompt
    rounded_box(ax, 1.5, 86.5, 14.5, 3.0,
                '"This toy can be described by these aspects:"',
                fc="#FFE9C2", fontsize=7.4, fontweight="bold")
    ax.text(8.75, 85.6, "Prompt", ha="center", va="top",
            fontsize=7.4, style="italic")

    # Item codeword boxes (active/inactive)
    cw_y = 86.6; cw_h = 2.7; cw_w = 2.4; gap = 0.4
    cw_x0 = 18.0
    item_active = [True, True, False, True, True, False, True, True]
    aspect_labels = ["c₁", "c₂", "c₃", "c₄", "c₅", "c₆", "c₇", "c₈"]
    for i, (lab, act) in enumerate(zip(aspect_labels, item_active)):
        x = cw_x0 + i * (cw_w + gap)
        if act:
            rounded_box(ax, x, cw_y, cw_w, cw_h, lab, fc="#CFE9FF",
                        ec="#2566A8", fontsize=8.5, fontweight="bold")
        else:
            rounded_box(ax, x, cw_y, cw_w, cw_h, lab, fc="#F4F4F4",
                        ec="#BBBBBB", fontsize=8.5, txt_color="#BBBBBB")
            ax.plot([x+0.2, x+cw_w-0.2], [cw_y+0.3, cw_y+cw_h-0.3],
                    color="#BB6666", lw=1.2)
    ax.text(cw_x0 + 4*(cw_w+gap) - gap/2, 85.6,
            "Item codewords (active by depth_mask)",
            ha="center", va="top", fontsize=7.4, style="italic")

    # → MiniLM item
    ml_x, ml_y, ml_w, ml_h = 44.0, 84.5, 9.5, 5.0
    arrow(ax, 16.0, 88.0, ml_x, 87.0)
    arrow(ax, cw_x0 + 8*(cw_w+gap)-gap, 88.0, ml_x, 87.0)
    rounded_box(ax, ml_x, ml_y, ml_w, ml_h, "MiniLM (frozen)\ninputs-embeds",
                fc="#E8F1FA", ec="#2566A8", fontsize=8.5, fontweight="bold")
    ax.text(ml_x + 0.6, ml_y + ml_h - 0.5, "❄", fontsize=10, color="#2566A8")

    # h_d_item
    arrow(ax, ml_x + ml_w, 87.0, ml_x + ml_w + 3.0, 87.0)
    hd_i_x, hd_i_y, hd_i_w, hd_i_h = 57.0, 85.5, 5.0, 3.0
    rounded_box(ax, hd_i_x, hd_i_y, hd_i_w, hd_i_h,
                r"$\mathbf{h}_d^{item}$", fc="#9DC8F2", ec="#1B4C7F",
                fontsize=11, fontweight="bold")
    ax.text(hd_i_x + hd_i_w/2, hd_i_y - 0.4, "mean-pool + L2",
            ha="center", va="top", fontsize=7, style="italic")

    # Item h_raw target path (bottom of item section)
    ax.text(1.5, 81.2, "Raw text → GPT-4o-mini → profile",
            ha="left", va="top", fontsize=7.6, color="#555")
    rounded_box(ax, 1.5, 76.5, 14.5, 3.0,
                '"The toy attributes are xxx."',
                fc="#FFFAE0", ec="#8C7B22", fontsize=7.2, fontweight="bold")
    arrow(ax, 16.0, 78.0, 31.5, 78.0)
    rounded_box(ax, 31.5, 76.5, 10.0, 3.0,
                "MiniLM (frozen)", fc="#E8F1FA", ec="#2566A8",
                fontsize=8.0, fontweight="bold")
    ax.text(31.6 + 0.4, 76.5 + 3.0 - 0.4, "❄", fontsize=8, color="#2566A8")

    # h_raw_item
    arrow(ax, 41.5, 78.0, 50.0, 78.0)
    hri_x, hri_y, hri_w, hri_h = 50.0, 76.5, 5.5, 3.0
    rounded_box(ax, hri_x, hri_y, hri_w, hri_h,
                r"$\mathbf{h}_{raw}^{item}$",
                fc="#F2C2C2", ec="#7F1B1B", fontsize=10, fontweight="bold")

    # InfoNCE box for item
    info_i_x, info_i_y, info_i_w, info_i_h = 67.0, 80.5, 11.0, 7.0
    rounded_box(ax, info_i_x, info_i_y, info_i_w, info_i_h,
                "InfoNCE\n(in-batch)", fc="#F6E1F0",
                ec="#A8367A", fontsize=9, fontweight="bold")
    arrow(ax, hd_i_x + hd_i_w, hd_i_y + hd_i_h/2,
          info_i_x, info_i_y + info_i_h*0.7, style="->")
    arrow(ax, hri_x + hri_w, hri_y + hri_h/2,
          info_i_x, info_i_y + info_i_h*0.3, style="->")
    ax.text(info_i_x + info_i_w/2, info_i_y - 0.6,
            r"$L_{align}^{item}$",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#5A1F46")

    # ─────────── horizontal separator ───────────
    ax.plot([1.5, 98], [73.5, 73.5], color="#CCCCCC", lw=0.6, linestyle=(0, (4, 4)))

    # === USER BOTTOM TRUNK ===
    ax.text(1.5, 71.5, "USER", ha="left", va="center", fontsize=10,
            fontweight="bold", color="#7A2C2C")

    # User prompt
    rounded_box(ax, 1.5, 66.0, 14.5, 3.0,
                '"This user can be described by these aspects:"',
                fc="#FFE9C2", fontsize=7.4, fontweight="bold")
    ax.text(8.75, 65.2, "Prompt", ha="center", va="top",
            fontsize=7.4, style="italic")

    # User codeword boxes
    user_active = [True, False, True, True, False, True, True, True]
    cw_y2 = 66.1
    for i, (lab, act) in enumerate(zip(aspect_labels, user_active)):
        x = cw_x0 + i * (cw_w + gap)
        if act:
            rounded_box(ax, x, cw_y2, cw_w, cw_h, lab, fc="#FFD7D7",
                        ec="#A85B5B", fontsize=8.5, fontweight="bold")
        else:
            rounded_box(ax, x, cw_y2, cw_w, cw_h, lab, fc="#F4F4F4",
                        ec="#BBBBBB", fontsize=8.5, txt_color="#BBBBBB")
            ax.plot([x+0.2, x+cw_w-0.2], [cw_y2+0.3, cw_y2+cw_h-0.3],
                    color="#BB6666", lw=1.2)
    ax.text(cw_x0 + 4*(cw_w+gap) - gap/2, 65.2,
            "User codewords (active by depth_mask)",
            ha="center", va="top", fontsize=7.4, style="italic")

    # MiniLM user
    ml_x2, ml_y2, ml_w2, ml_h2 = 44.0, 64.0, 9.5, 5.0
    arrow(ax, 16.0, 67.5, ml_x2, 66.5)
    arrow(ax, cw_x0 + 8*(cw_w+gap)-gap, 67.5, ml_x2, 66.5)
    rounded_box(ax, ml_x2, ml_y2, ml_w2, ml_h2, "MiniLM (frozen)\ninputs-embeds",
                fc="#E8F1FA", ec="#2566A8", fontsize=8.5, fontweight="bold")
    ax.text(ml_x2 + 0.6, ml_y2 + ml_h2 - 0.5, "❄", fontsize=10, color="#2566A8")

    # h_d_user
    arrow(ax, ml_x2 + ml_w2, 66.5, ml_x2 + ml_w2 + 3.0, 66.5)
    hd_u_x, hd_u_y, hd_u_w, hd_u_h = 57.0, 65.0, 5.0, 3.0
    rounded_box(ax, hd_u_x, hd_u_y, hd_u_w, hd_u_h,
                r"$\mathbf{h}_d^{user}$", fc="#F2C2C2", ec="#7F1B1B",
                fontsize=11, fontweight="bold")

    # ★ u2i alignment target box (mean of interacted item h_raw)
    rounded_box(ax, 31.5, 58.0, 19.0, 3.5,
                "★ u2i target = mean$_{i \\in \\mathrm{history}(u)}$ $\\mathbf{h}_{raw}^{item}[i]$",
                fc="#E7F6E1", ec="#3A8A2C",
                fontsize=8.5, fontweight="bold", txt_color="#1B5E1B")
    ax.text(41.0, 56.7, "(injects collaborative signal — DAS-style)",
            ha="center", va="top", fontsize=7.0, style="italic", color="#1B5E1B")
    arrow(ax, 50.5, 59.7, 50.5, 65.6, style="-|>")

    # u2i target box → h_target_user
    htu_x, htu_y, htu_w, htu_h = 51.5, 60.0, 5.5, 3.0
    rounded_box(ax, htu_x, htu_y, htu_w, htu_h,
                r"$\mathbf{h}_{raw}^{u2i}$",
                fc="#A6D8A0", ec="#3A8A2C", fontsize=10, fontweight="bold")

    # InfoNCE for user
    info_u_x, info_u_y, info_u_w, info_u_h = 67.0, 60.5, 11.0, 7.0
    rounded_box(ax, info_u_x, info_u_y, info_u_w, info_u_h,
                "InfoNCE\n(in-batch)", fc="#F6E1F0",
                ec="#A8367A", fontsize=9, fontweight="bold")
    arrow(ax, hd_u_x + hd_u_w, hd_u_y + hd_u_h/2,
          info_u_x, info_u_y + info_u_h*0.7, style="->")
    arrow(ax, htu_x + htu_w, htu_y + htu_h/2,
          info_u_x, info_u_y + info_u_h*0.3, style="->")
    ax.text(info_u_x + info_u_w/2, info_u_y - 0.6,
            r"$L_{align}^{user}$",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#5A1F46")

    # ────────────────────────────────────────────────────────────────────
    # BRIDGE — identifier cache (y: 50–56)
    # ────────────────────────────────────────────────────────────────────
    rounded_box(ax, 25.0, 50.0, 50.0, 4.2,
                "identifier_cache.pkl   (user + item, textual word tokens, variable length L_i)",
                fc="#FFFAE0", ec="#8C7B22", fontsize=9, fontweight="bold")
    ax.text(50.0, 49.0,
            "['novel', 'child', 'imagine', <EOA>, 'magic', 'story', <EOA>, ...]   "
            "— each row variable, mean ~26 tokens",
            ha="center", va="top", fontsize=7.4, style="italic", color="#555")

    # Bridge arrows from alignment to cache
    arrow(ax, 60.0, 79.5, 50.0, 54.5, style="->", color="#888", lw=1)
    arrow(ax, 60.0, 60.5, 50.0, 54.5, style="->", color="#888", lw=1)
    ax.text(40.5, 56.5, "(saved after Stage 1 training)",
            ha="left", va="bottom", fontsize=7, style="italic", color="#888")

    # ────────────────────────────────────────────────────────────────────
    # (c) RECOMMENDATION — bottom half (y: 0–47)
    # ────────────────────────────────────────────────────────────────────
    ax.text(1.5, 45.5, "(c) Recommendation — RPG (item-only generative + user prefix)",
            ha="left", va="center", fontsize=13, fontweight="bold")

    # User input box (left)
    rounded_box(ax, 1.0, 35.0, 6.0, 4.0,
                "user $u$", fc="#FFEDED", ec="#A85B5B",
                fontsize=10, fontweight="bold")
    ax.text(4.0, 34.0, "(query)", ha="center", va="top",
            fontsize=7, style="italic")

    # User identifier (textual, multi-token) — but for prepend mode it's mean-pooled
    arrow(ax, 7.0, 37.0, 9.5, 37.0)
    rounded_box(ax, 9.5, 35.0, 13.5, 4.0,
                "lookup user_id2tokens\n+ mean-pool over active",
                fc="#FFD7D7", ec="#A85B5B",
                fontsize=8.0, fontweight="bold")
    # ★ tag
    ax.text(16.25, 39.4, "★ user_fuse = prepend",
            ha="center", va="bottom", fontsize=7.4,
            color="#A85B5B", fontweight="bold", style="italic")

    # User token output
    arrow(ax, 23.0, 37.0, 25.5, 37.0)
    rounded_box(ax, 25.5, 35.0, 4.0, 4.0,
                r"$\mathbf{u}_{vec}$", fc="#F2C2C2", ec="#7F1B1B",
                fontsize=10, fontweight="bold")

    # History items
    rounded_box(ax, 1.0, 22.0, 6.0, 4.0,
                "history\n$[i_1,...,i_S]$",
                fc="#FFE9C2", ec="#A88718", fontsize=8.5, fontweight="bold")
    arrow(ax, 7.0, 24.0, 9.5, 24.0)
    rounded_box(ax, 9.5, 22.0, 13.5, 4.0,
                "Item-level pool\nper history slot",
                fc="#E7F6E1", ec="#3A8A2C",
                fontsize=8.0, fontweight="bold")
    ax.text(16.25, 21.0,
            r"$\frac{\Sigma\, e_p \cdot m_p}{\Sigma\, m_p}$",
            ha="center", va="top", fontsize=8.5, color="#222")
    arrow(ax, 23.0, 24.0, 25.5, 24.0)
    rounded_box(ax, 25.5, 22.0, 4.0, 4.0,
                r"$\mathbf{x}_{1..S}$", fc="#CFE9FF", ec="#2566A8",
                fontsize=9, fontweight="bold")

    # GPT-2 backbone (combines u_vec + x_1..S)
    arrow(ax, 29.5, 35.0, 34.0, 31.0)  # u_vec → GPT-2 (top)
    arrow(ax, 29.5, 26.0, 34.0, 29.0)  # x → GPT-2 (bottom)
    rounded_box(ax, 34.0, 25.0, 13.0, 10.0,
                "GPT-2\n(2L, 4H, d=256\ncausal SA)",
                fc="#E8F1FA", ec="#2566A8",
                fontsize=9.5, fontweight="bold")
    ax.text(40.5, 24.0,
            r"input = $[\mathbf{u}_{vec} \;|\; \mathbf{x}_1, \dots, \mathbf{x}_S]$",
            ha="center", va="top", fontsize=7.4, style="italic", color="#555")

    # → h_last
    arrow(ax, 47.0, 30.0, 50.0, 30.0)
    rounded_box(ax, 50.0, 28.5, 5.0, 3.0,
                r"$\mathbf{h}_{last}$", fc="#9DC8F2", ec="#1B4C7F",
                fontsize=10, fontweight="bold")

    # N parallel heads
    arrow(ax, 55.0, 30.0, 58.0, 30.0)
    rounded_box(ax, 58.0, 24.0, 14.0, 12.0,
                "", fc="#FFF3E5", ec="#C97818")
    ax.text(65.0, 34.5, "N = K·L = 32",
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color="#7A4A0F")
    ax.text(65.0, 33.0, "parallel heads",
            ha="center", va="top", fontsize=8.5, color="#7A4A0F")
    # mini stack
    for i in range(6):
        ax.add_patch(Rectangle((59.0 + i*0.25, 25.0 + i*0.25),
                                10.5, 1.1, lw=0.6, ec="#A86420",
                                fc="#FFD89A", alpha=0.7))
    ax.text(65.0, 24.4, "★ no autoregressive",
            ha="center", va="top", fontsize=7.4, style="italic",
            color="#7A4A0F")

    # Logits / log-prob
    arrow(ax, 72.0, 30.0, 75.0, 30.0)
    rounded_box(ax, 75.0, 25.0, 9.0, 10.0,
                "logits\n→ log_p\n(B, P, V)",
                fc="#F6E1F0", ec="#A8367A",
                fontsize=8.5, fontweight="bold")

    # Ranking
    arrow(ax, 84.0, 30.0, 87.0, 30.0)
    rounded_box(ax, 87.0, 24.0, 12.0, 12.0,
                "", fc="#FFEDED", ec="#A85B5B")
    ax.text(93.0, 34.4, "Top-K Ranking",
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color="#7A2C2C")
    ax.text(93.0, 32.5,
            r"score$_i$ = $\frac{1}{|A_i|}$",
            ha="center", va="top", fontsize=7.6, color="#222")
    ax.text(93.0, 30.7,
            r"$\sum_{p \in A_i} \log p_{i,p}$",
            ha="center", va="top", fontsize=7.6, color="#222")
    ax.text(93.0, 28.5,
            "1. item₁₂\n2. item₃₄\n3. item₂₁",
            ha="center", va="top", fontsize=7.6, color="#2A2A2A",
            fontweight="bold")

    # ────────────────────────────────────────────────────────────────────
    # Variable-depth signal annotation
    # ────────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.025,
              "★ Variable-depth signal enters 5×: "
              "(1) Stage-1 reconstruction; (2) Stage-1 alignment (var-len descriptor); "
              "(3) Stage-2 item pool; (4) Stage-2 user pool; "
              "(5) Stage-2 per-position loss & ranking score $|A_i|$.",
              ha="center", fontsize=8.5, color="#444", style="italic")

    fig.text(0.5, 0.975,
              "InATTo — Alignment → Identifier Cache → Generative Recommendation",
              ha="center", fontsize=12.5, fontweight="bold")

    out = Path("/home/koohy/cikm/InATTo/figures") / "inatto_full_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
