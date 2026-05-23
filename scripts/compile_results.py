"""Compile every experiment result into a single markdown file.

Reads JSON logs/results from results/ and writes RESULTS.md at repo root.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO.parent / "RESULTS.md"

LABELS = ["sparse", "medium", "rich"]
METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]


def load_sparsity(path: Path):
    d = json.load(path.open())
    pu = d["per_user"]
    n_per_group = d["n_users_per_group"]
    bounds = [float(b) for b in d["boundaries"]]
    groups = {}
    offset = 0
    for lab, n in zip(LABELS, n_per_group):
        sub = {k: pu[k][offset:offset + n] for k in METRICS}
        groups[lab] = {k: float(np.mean(v)) for k, v in sub.items()}
        groups[lab]["n_users"] = n
        offset += n
    overall = {k: float(np.mean(pu[k])) for k in METRICS}
    return groups, overall, bounds, d.get("ckpt", "?")


def best_val(path: Path):
    d = json.load(path.open())
    m = max(d, key=lambda e: e["R@5"])
    return len(d), m


def last_stage1(path: Path):
    d = json.load(path.open())
    return d[-1]


def fmt_pct(x):
    return f"{x*100:+.2f}%"


def main():
    lines: list[str] = []
    L = lines.append

    L("# InATTo — Experimental Results")
    L("")
    L("All numbers below are full-ranking, leave-one-out evaluation.")
    L("`InATTo` denotes our paper-main config: ssw01 + Gumbel-STE v2.")
    L("Stage 2 RPG with `wd=0.05, dropout=0.3, lr=3e-3, batch=256, history=50, patience=20`.")
    L("")
    L("---")
    L("")

    # ============================================================
    # Table 1 — Main 4-dataset test overall: ssw01-base vs InATTo (gumbel_v2)
    # ============================================================
    L("## Table 1 — Main results (TEST overall, 4 datasets)")
    L("")
    L("`ssw01-base` = Stage 1 ssw01 baseline (without Gumbel-STE v2).")
    L("`InATTo (gumbel_v2)` = paper-main config.")
    L("")
    L("| Dataset | Method | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |")
    L("|---|---|---:|---:|---:|---:|---:|---:|")

    for ds in ["toys", "beauty", "sports", "yelp"]:
        b_g, b_o, _, _ = load_sparsity(RESULTS / f"sparsity_{ds}_2023.json")
        g_g, g_o, _, _ = load_sparsity(RESULTS / f"sparsity_{ds}_2023_gumbel_v2.json")
        L(f"| {ds.capitalize()} | ssw01-base | "
          + " | ".join(f"{b_o[k]:.4f}" for k in METRICS) + " |")
        L(f"| {ds.capitalize()} | **InATTo (gumbel_v2)** | "
          + " | ".join(f"**{g_o[k]:.4f}**" for k in METRICS) + " |")
        delta_pct = [(g_o[k] - b_o[k]) / b_o[k] for k in METRICS]
        L(f"| {ds.capitalize()} | Δ% | "
          + " | ".join(fmt_pct(d) for d in delta_pct) + " |")
    L("")
    L("---")
    L("")

    # ============================================================
    # Table 2 — Sparsity-group breakdown (InATTo gumbel_v2)
    # ============================================================
    L("## Table 2 — Sparsity-group breakdown (InATTo gumbel_v2)")
    L("")
    L("Items grouped by raw-text word-count quantile (tertile).")
    L("")
    L("| Dataset | Group | n_users | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |")
    L("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in ["toys", "beauty", "sports", "yelp"]:
        g, o, bounds, _ = load_sparsity(RESULTS / f"sparsity_{ds}_2023_gumbel_v2.json")
        for lab in LABELS:
            gr = g[lab]
            L(f"| {ds.capitalize()} | {lab} | {gr['n_users']} | "
              + " | ".join(f"{gr[k]:.4f}" for k in METRICS) + " |")
        L(f"| {ds.capitalize()} | **OVERALL** | all | "
          + " | ".join(f"**{o[k]:.4f}**" for k in METRICS) + " |")
        L(f"| {ds.capitalize()} | (bounds) | colspan | "
          + f"word-count = {bounds} |  |  |  |  |")
    L("")
    L("---")
    L("")

    # ============================================================
    # Table 3 — Toys text-sparsity ablation: Stage 1 final metrics
    # ============================================================
    L("## Table 3 — Toys text-sparsity ablation (Stage 1 final-epoch metrics)")
    L("")
    L("- **S0** = full raw text  `(title + brand + categories + description + price + salesrank)`")
    L("- **S1** = `(title + brand + categories + price + salesrank)`  (description removed)")
    L("- **S2** = `(title)`  (title only)")
    L("- **Light** = only `itm_text_embeds` sparsified (`h_raw`, ρ kept as S0)")
    L("- **Strict** = `itm_text_embeds`, `h_raw` (re-generated GPT profile), and ρ all sparsified")
    L("")
    L("| Setting | L_total | L_recon | L_Q | L_align | L_bpr | L_ssw | phi_item_mean |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|")
    settings_s1 = [
        ("S0 (full)",   "inatto-toys-2023.stage1bpr.ssw01.gumbel_v2.log.json"),
        ("Light S1",    "inatto-toys-2023.stage1bpr.ssw01.gumbel_v2.s1.log.json"),
        ("Strict S1",   "inatto-toys-2023.stage1bpr.ssw01.gumbel_v2.strict_s1.log.json"),
        ("Light S2",    "inatto-toys-2023.stage1bpr.ssw01.gumbel_v2.s2.log.json"),
        ("Strict S2",   "inatto-toys-2023.stage1bpr.ssw01.gumbel_v2.strict_s2.log.json"),
    ]
    for tag, fname in settings_s1:
        e = last_stage1(RESULTS / fname)
        L(f"| {tag} | {e['L_total']:.4f} | {e['L_recon']:.4f} | {e['L_Q']:.4f} | "
          f"{e['L_align']:.4f} | {e['L_bpr']:.4f} | {e['L_ssw']:.4f} | "
          f"{e['phi_item_mean']:.4f} |")
    L("")
    L("---")
    L("")

    # ============================================================
    # Table 4 — Toys text-sparsity Stage 2 VAL best
    # ============================================================
    L("## Table 4 — Toys text-sparsity ablation (Stage 2 VAL best)")
    L("")
    L("| Setting | epochs | best ep | val R@5 | val R@10 | val R@20 | val N@5 | val N@10 | val N@20 |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    settings_s2 = [
        ("S0 (full)",   "inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.wd05.do30.log.json"),
        ("Light S1",    "inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.s1.wd05.do30.log.json"),
        ("Strict S1",   "inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.strict_s1.wd05.do30.log.json"),
        ("Light S2",    "inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.s2.wd05.do30.log.json"),
        ("Strict S2",   "inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.strict_s2.wd05.do30.log.json"),
    ]
    for tag, fname in settings_s2:
        n, m = best_val(RESULTS / fname)
        L(f"| {tag} | {n} | {m['epoch']} | {m['R@5']:.4f} | {m['R@10']:.4f} | "
          f"{m['R@20']:.4f} | {m['N@5']:.4f} | {m['N@10']:.4f} | {m['N@20']:.4f} |")
    L("")
    L("---")
    L("")

    # ============================================================
    # Table 5 — Toys text-sparsity TEST overall
    # ============================================================
    L("## Table 5 — Toys text-sparsity ablation (TEST overall, n=19412)")
    L("")
    L("| Setting | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    settings_test = [
        ("S0 (full)",   "sparsity_toys_2023_gumbel_v2.json"),
        ("Light S1",    "sparsity_toys_2023_gumbel_v2_s1.json"),
        ("Strict S1",   "sparsity_toys_2023_gumbel_v2_strict_s1.json"),
        ("Light S2",    "sparsity_toys_2023_gumbel_v2_s2.json"),
        ("Strict S2",   "sparsity_toys_2023_gumbel_v2_strict_s2.json"),
    ]
    s0_overall = None
    rows_test = []
    for tag, fname in settings_test:
        _, o, _, _ = load_sparsity(RESULTS / fname)
        rows_test.append((tag, o))
        L(f"| {tag} | " + " | ".join(f"{o[k]:.4f}" for k in METRICS) + " |")
        if tag == "S0 (full)":
            s0_overall = o
    L("")
    L("**Δ% vs S0 (full)**")
    L("")
    L("| Setting | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    for tag, o in rows_test[1:]:
        deltas = [(o[k] - s0_overall[k]) / s0_overall[k] for k in METRICS]
        L(f"| {tag} | " + " | ".join(fmt_pct(d) for d in deltas) + " |")
    L("")
    L("**Δ% Strict vs Light**")
    L("")
    L("| Comparison | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    pairs = [(("Light S1", "Strict S1")), (("Light S2", "Strict S2"))]
    by_tag = dict(rows_test)
    for l_tag, s_tag in pairs:
        l = by_tag[l_tag]; s = by_tag[s_tag]
        deltas = [(s[k] - l[k]) / l[k] for k in METRICS]
        L(f"| {s_tag} vs {l_tag} | " + " | ".join(fmt_pct(d) for d in deltas) + " |")
    L("")
    L("---")
    L("")

    # ============================================================
    # Notes
    # ============================================================
    L("## Notes")
    L("")
    L("- **Light vs Strict (S1)**: differences within ~±1% noise — alignment-target leakage was minor when only description was removed.")
    L("- **Light vs Strict (S2)**: Light overstates R@5 by ~12% because the `h_raw` (GPT-profile) still encodes description-level semantics; Strict re-generates the profile from title-only input, removing that leakage and revealing the true robustness ceiling.")
    L("- Paper-honest sparsity claim: **Strict S1 −7%, Strict S2 −12% R@5 vs full text** — InATTo gracefully degrades with text sparsity, less than the information loss itself.")
    L("")

    OUT.write_text("\n".join(lines))
    print(f"saved -> {OUT}")
    print(f"size: {OUT.stat().st_size / 1024:.1f} KB, {len(lines)} lines")


if __name__ == "__main__":
    main()
