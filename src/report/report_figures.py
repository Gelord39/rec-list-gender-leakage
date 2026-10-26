"""Figures for the report (Overleaf ``figs/``), drawn at the size they are printed.

ACM sigconf: ``\\textwidth`` = 7.0 in (``figure*``), ``\\columnwidth`` = 3.33 in
(``figure``). Every figure is drawn at exactly that width with 7-8 pt text, so it
is included at ``width=\\textwidth`` / ``\\columnwidth`` without scaling.

Same data as ``slides_figures`` (every number read from ``runs/``); differences from
the 2026-10-05 slide figures:

- no headline titles inside the figures (the caption carries the claim);
- 03 threat-model diagram (new);
- 04 ladder: values printed only at T4, where the text discusses them;
- 05 classifiers: neutral legend, markers offset so overlapping classifiers stay visible;
- 06 popularity, 07 attacker x input, 12 age slices: single-column;
- 07: BPR list added; classifier drawn in a neutral colour (blue means LightGCN elsewhere);
- 08 shots: 0-128 shots incl. 64, 3 exemplar draws per point (scripts/hendrix_shot_fill.sh);
- 09 BPR vs LightGCN: many-shot rows are the mean of 3 exemplar draws (the slide used
  the single best draw, seed 2025); third panel = Romance intervention;
- 11 list change: paired 95% CIs, plus list overlap (Jaccard) by gender.

    uv run python -m src.report.report_figures [--out-dir docs/report_figs] [--only 08_shots ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.report.slides_figures import (
    BPR, C_BPR, C_GENDER, C_GLM, C_LGCN, C_LLAMA, C_NEUTRAL, C_QWEN, GRID, INK, INK2, LGCN, LLMREC, MUTED,
    SURFACE, _genres, _grid, _history, _j, _preds_auc, _romance_share, _shots_cell, plt,
)

TEXT_W, COL_W = 7.0, 3.33   # inches, acmart sigconf
C_CLF = "#52514e"           # trained classifier (neutral: blue/orange mean LightGCN/BPR)
C_WOMEN, C_MEN = "#b5317a", "#1f8a8a"   # magenta / teal: apart from the recommender and LLM colours
FS = 7                      # annotation text
RNG_SEED = 2024
N_BOOT = 2000

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "grid.linewidth": 0.4, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "axes.titlepad": 3, "axes.labelpad": 2, "pdf.fonttype": 42,
})


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout(pad=0.3)
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf", f"({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)")


def _dot(ax, xpos: float, v: float, lo: float, hi: float, color: str, label: str | None = None,
         text: bool = True, dx: float = 0.06) -> None:
    """Point estimate + CI whisker."""
    ax.plot([xpos, xpos], [lo, hi], color=color, lw=1.2, solid_capstyle="round", zorder=2)
    ax.scatter([xpos], [v], s=18, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3, label=label)
    if text:
        ax.text(xpos + dx, v, f"{v:.2f}", va="center", ha="left", fontsize=FS - 0.5, color=INK2)


def _chance(ax, label: bool = True) -> None:
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", ha="left", color=MUTED, fontsize=FS - 0.5)


def _boot_mean_ci(x: np.ndarray, n_boot: int = N_BOOT) -> list[float]:
    rng = np.random.default_rng(RNG_SEED)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _panel(ax, s: str) -> None:
    ax.set_title(s, loc="left", fontsize=8, fontweight="bold", color=INK)


# ------------------------------------------------------------------ 03 threat model
def fig_threat_model(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.25))
    ax.set_xlim(0, 11)
    ax.set_ylim(0.4, 3.95)
    ax.axis("off")

    def box(x, y, w, h, text, fc=SURFACE, fs=7.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=INK2, lw=0.8, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight=weight, zorder=3, linespacing=1.25)

    def arrow(x0, y0, x1, y1, color=INK2, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=0.9, ls=ls, zorder=1))

    def tier(x, y, code, what, labels, color):
        ax.text(x, y, code, ha="center", va="top", fontsize=8, fontweight="bold", color=color)
        ax.text(x, y - 0.32, what, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.2)
        ax.text(x, y - 0.95, labels, ha="center", va="top", fontsize=6.3, color=MUTED, style="italic")

    box(0.2, 2.45, 1.9, 1.2, "Interaction\nhistory\n(train items)")
    box(2.75, 2.25, 2.6, 1.6, "")
    ax.text(4.05, 3.6, "Recommender", ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold")
    box(2.95, 2.4, 1.05, 0.85, "user\nembedding u", fc="#f4f3ef", fs=6.5)
    box(4.12, 2.4, 1.05, 0.85, "item\nembeddings", fc="#f4f3ef", fs=6.5)
    box(6.0, 2.45, 1.9, 1.2, "Top-10 list\n(item IDs)")
    box(8.55, 2.45, 2.25, 1.2, "Public metadata\n(titles, genres)")
    arrow(2.1, 3.05, 2.75, 3.05)
    arrow(5.35, 3.05, 6.0, 3.05)
    arrow(7.9, 3.05, 8.55, 3.05, ls=(0, (3, 2)))
    ax.text(8.22, 3.22, "join", ha="center", fontsize=6.3, color=MUTED)

    y = 2.05
    tier(1.15, y, "TH", "full history", "3,367 (classifier) / 0 (LLM) labels", INK)
    tier(3.47, y, "T0", "user embedding", "3,367 labels", INK)
    tier(4.65, y, "T1", "list + item\nembeddings", "3,367 labels", INK)
    tier(6.95, y, "T2", "list item IDs", "3,367 labels", INK)
    tier(8.95, y, "T3", "list + genres /\ntitle text", "3,367 labels", INK)
    tier(10.25, y, "T4", "titles read\nby an LLM", "0-128 labels", C_GLM)

    ax.plot([0.2, 5.35], [0.78, 0.78], color=INK2, lw=0.9)
    ax.text(2.78, 0.57, "platform insider (data or model access)", ha="center", va="center", fontsize=6.8, color=INK2)
    ax.plot([6.0, 10.8], [0.78, 0.78], color=C_GLM, lw=0.9)
    ax.text(8.4, 0.57, "anyone who sees the list (e.g. a shared screen or a third-party plug-in)",
            ha="center", va="center", fontsize=6.8, color=C_GLM)
    _save(fig, out, "03_threat_model")


# ------------------------------------------------------------------ 04 ladder
def fig_ladder(out: Path) -> None:
    tiers = [("TH\nfull history", None, "3,367"), ("T0\nuser embedding", "true_user_embedding", "3,367"),
             ("T1\nreconstructed\nuser embedding", "inversion", "3,367"), ("T2\nlist item IDs\n(multi-hot)", "multi_hot", "3,367"),
             ("T3\nlist titles\n(MiniLM)", "title_text_embedding", "3,367"), ("T4\nLLM reads titles\n(GLM-4-9B)", "llm", "0 or 16")]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.9))
    x = np.arange(len(tiers))
    for k, (run, color, label, prefix) in enumerate(((BPR, C_BPR, "BPR", "bpr"), (LGCN, C_LGCN, "LightGCN", "lightgcn"))):
        vals, cis = [], []
        for _, inp, _ in tiers:
            if inp is None:
                v, ci = _history(run)
            elif inp == "llm":
                v, ci = _preds_auc(run / f"llm_aia/preds_zero_shot_{prefix}_glm4_9b_500.jsonl")
            else:
                v, ci = _grid(run, inp)
            vals.append(v)
            cis.append(ci)
        # T4: the 16-shot attacker (more labels) sits on the tier, the zero-shot one to its right,
        # so the line keeps falling as the attacker gets less information
        xs = x + (k - 0.5) * 0.1
        x0 = xs[-1] + 0.3
        m, sd = _shots_cell(run)
        ax.plot(list(xs[:-1]) + [xs[-1]], vals[:-1] + [m], color=color, lw=1.3, zorder=1, label=label)
        ax.plot(list(xs[:-1]) + [x0], vals, color=color, lw=0, marker="o", ms=4.5, mec=SURFACE, mew=0.8, zorder=3)
        ax.errorbar(list(xs[:-1]) + [x0], vals,
                    yerr=np.array([[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]),
                    fmt="none", ecolor=color, elinewidth=0.8, capsize=0, zorder=2)
        ax.errorbar([xs[-1]], [m], yerr=[[sd], [sd]], fmt="D", ms=4.5, mfc=SURFACE, mec=color, mew=1.1, ecolor=color,
                    elinewidth=0.8, zorder=3)
        ax.plot([xs[-1], x0], [m, vals[-1]], color=color, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(xs[-1] - 0.07, m, f"{m:.2f}", color=INK2, fontsize=FS, ha="right", va="center")
        ax.text(x0 + 0.07, vals[-1], f"{vals[-1]:.2f}", color=INK2, fontsize=FS, ha="left", va="center")
    h, lab = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=INK2, marker="o", ls="none", ms=4), Line2D([], [], color=INK2, marker="D", mfc=SURFACE, ls="none", ms=4)]
    lab += ["T4 zero-shot (0 labels)", "T4 16-shot (16 labels, mean ± sd of 3 draws)"]
    ax.legend(h, lab, loc="upper right", ncol=2)
    ax.set_xticks(x, [f"{t}\n[{n} labels]" for t, _, n in tiers])
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Gender AUC (95% CI)")
    ax.set_xlim(-0.4, len(tiers) - 0.1)
    _chance(ax)
    _save(fig, out, "04_ladder")


# ------------------------------------------------------------------ 05 classifiers
def fig_classifiers(out: Path) -> None:
    inputs = [("true_user_embedding", "T0 user embedding"), ("inversion", "T1 reconstructed embedding"),
              ("ld_mean_pool", "T1 mean-pool"), ("multi_hot", "T2 multi-hot"),
              ("tfidf_multi_hot", "T2 TF-IDF multi-hot"), ("title_text_embedding", "T3 MiniLM titles"),
              ("genre_histogram", "T3 genre histogram")]
    clfs = [("lr_balanced", "LR", "o"), ("svm_linear_balanced", "SVM linear", "s"),
            ("svm_rbf_balanced", "SVM RBF", "D"), ("mlp_balanced", "MLP", "^")]
    offs = np.linspace(-0.24, 0.24, len(clfs))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.9), sharey=True)
    for ax, (run, color, name) in zip(axes, ((LGCN, C_LGCN, "LightGCN"), (BPR, C_BPR, "BPR"))):
        for i, (inp, _) in enumerate(inputs):
            v = [_grid(run, inp, c)[0] for c, _, _ in clfs]
            ax.axhspan(i - 0.36, i + 0.36, color="#f4f3ef", zorder=0, lw=0)
            for (c, _, m), a, dy in zip(clfs, v, offs):
                lr = c == "lr_balanced"
                ax.scatter(a, i + dy, marker=m, s=16 if lr else 12, color=color if lr else SURFACE,
                           edgecolor=color, linewidth=0.8, zorder=3)
        ax.set_yticks(range(len(inputs)), [lab for _, lab in inputs])
        ax.set_ylim(len(inputs) - 0.5, -0.5)
        ax.set_xlim(0.6, 0.86)
        ax.set_xlabel("Gender AUC (class-balanced variants)")
        _panel(ax, name)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", which="both", length=0)
    h = [Line2D([], [], marker=m, ls="none", ms=4, mfc=INK2 if c == "lr_balanced" else SURFACE, mec=INK2, mew=0.8)
         for c, _, m in clfs]
    axes[1].legend(h, [lab for _, lab, _ in clfs], loc="upper left", bbox_to_anchor=(1.0, 1.0),
                   handletextpad=0.2, title="top to bottom\nin each band", title_fontsize=6.5)
    _save(fig, out, "05_classifiers")


# ------------------------------------------------------------------ 06 popularity (single column)
def fig_popularity(out: Path) -> None:
    arms = [("full", "Full\nlist"), ("tail_only", "Without\n10 most-\nrecommended"),
            ("random_ablation", "Same no.\nremoved\nat random"), ("head_only", "Only\n10 most-\nrecommended")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    w = 0.34
    for k, (run, color, name) in enumerate(((LGCN, C_LGCN, "LightGCN"), (BPR, C_BPR, "BPR"))):
        res = _j(run / "popularity_ablation_500/popularity_ablation.json")["results"]["multi_hot"]
        for i, (arm, _) in enumerate(arms):
            r = res[arm]["lr_balanced"]
            if arm == "random_ablation":
                v, lo, hi = r["auc"]["mean"], r["auc"]["mean"] - r["auc"]["sd"], r["auc"]["mean"] + r["auc"]["sd"]
            else:
                v, (lo, hi) = r["auc"], r["auc_ci95"]
            _dot(ax, i + (k - 0.5) * w, v, lo, hi, color, name if i == 0 else None, text=False)
    ax.set_xticks(range(len(arms)), [a for _, a in arms])
    ax.set_ylim(0.45, 0.8)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylabel("Gender AUC, multi-hot LR")
    _chance(ax, label=False)
    ax.legend(loc="upper right", ncol=2, handletextpad=0.1, columnspacing=0.8)
    _save(fig, out, "06_popularity")


# ------------------------------------------------------------------ 07 attacker x input (single column)
def fig_matrix(out: Path) -> None:
    groups = [("Watch\nhistory", _history(LGCN),
               _preds_auc(LGCN / "llm_aia/preds_history_zero_shot_lightgcn_glm4_9b_500_hist100.jsonl")),
              ("LightGCN\ntop-10 list", _grid(LGCN, "multi_hot"),
               _preds_auc(LGCN / "llm_aia/preds_staab_few_shot_lightgcn_glm4_9b_500.jsonl")),
              ("BPR\ntop-10 list", _grid(BPR, "multi_hot"),
               _preds_auc(BPR / "llm_aia/preds_staab_few_shot_bpr_glm4_9b_500.jsonl"))]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    w = 0.34
    for i, (_, clf, llm) in enumerate(groups):
        for k, (label, color, (v, ci)) in enumerate((("LR classifier (3,367 labels)", C_CLF, clf),
                                                     ("LLM, GLM-4-9B (0 or 2 labels)", C_GLM, llm))):
            _dot(ax, i + (k - 0.5) * w, v, ci[0], ci[1], color, label if i == 0 else None)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.35)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Gender AUC (95% CI)")
    _chance(ax, label=False)
    ax.legend(loc="upper right", handletextpad=0.1)
    _save(fig, out, "07_attacker_x_input")


# ------------------------------------------------------------------ 08 shots
def fig_shots(out: Path) -> None:
    models = (("glm4_9b", "GLM-4-9B", C_GLM), ("llama31_8b", "Llama-3.1-8B", C_LLAMA), ("qwen25_14b", "Qwen2.5-14B", C_QWEN))
    ticks = [0, 2, 8, 16, 32, 64, 128]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.55), sharey=True)
    for ax, (run, name) in zip(axes, ((LGCN, "LightGCN lists"), (BPR, "BPR lists"))):
        cells = _j(run / "shot_sweep/shot_sweep.json")["runs"]
        for slug, label, color in models:
            pts: dict[int, list[float]] = {}
            for r in cells:
                if r["model"] == slug and r["template"] == "plain" and r["auc"] is not None:
                    pts.setdefault(r["shots"], []).append(r["auc"])
            shots = sorted(pts)
            m = np.array([np.mean(pts[s]) for s in shots])
            sd = np.array([np.std(pts[s], ddof=1) if len(pts[s]) > 1 else 0 for s in shots])
            xpos = [np.log2(s + 1) for s in shots]
            ax.fill_between(xpos, m - sd, m + sd, color=color, alpha=0.15, lw=0)
            ax.plot(xpos, m, color=color, lw=1.2, marker="o", ms=3.5, mec=SURFACE, mew=0.6, label=label)
        lr = _grid(run, "multi_hot")[0]
        ax.axhline(lr, color=INK2, lw=0.8, ls=(0, (6, 3)))
        ax.text(0, lr + 0.005, f"LR on the same lists, 3,367 labels: {lr:.3f}", fontsize=FS, color=INK2)
        ax.set_xticks([np.log2(t + 1) for t in ticks], [str(t) for t in ticks])
        ax.set_xlim(-0.3, np.log2(129) + 0.3)
        ax.set_xlabel("In-context labelled exemplars (shots)")
        _panel(ax, name)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    axes[0].set_ylabel("Gender AUC\n(mean ± sd over 3 exemplar draws)")
    axes[0].set_ylim(0.45, 0.76)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.77))
    _save(fig, out, "08_shots")


# ------------------------------------------------------------------ 09 BPR vs LightGCN + Romance
def _sweep_mean(run: Path, model: str, shots: int) -> float:
    v = [r["auc"] for r in _j(run / "shot_sweep/shot_sweep.json")["runs"]
         if r["model"] == model and r["template"] == "plain" and r["shots"] == shots]
    if len(v) != 3:
        raise ValueError(f"{run} {model} {shots}-shot: expected 3 draws, found {len(v)}")
    return float(np.mean(v))


def fig_bpr_vs_lgcn_llm(out: Path) -> None:
    zero = [("llama32_500_lp", "llama3.2, 0-shot"), ("glm4_9b_500", "GLM-4-9B, 0-shot"),
            ("llama31_8b_500", "Llama-3.1-8B, 0-shot"), ("qwen25_14b_500", "Qwen2.5-14B, 0-shot")]
    many = [("glm4_9b", 8, "GLM-4-9B, 8-shot*"), ("qwen25_14b", 16, "Qwen2.5-14B, 16-shot*")]
    rows = []
    for tag, label in zero:
        rows.append((label, _preds_auc(LGCN / f"llm_aia/preds_zero_shot_lightgcn_{tag}.jsonl")[0],
                     _preds_auc(BPR / f"llm_aia/preds_zero_shot_bpr_{tag}.jsonl")[0]))
    for model, n, label in many:
        rows.append((label, _sweep_mean(LGCN, model, n), _sweep_mean(BPR, model, n)))

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(TEXT_W, 2.6), gridspec_kw={"width_ratios": [2.0, 0.85, 1.15]})
    for i, (label, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([a], [i], s=16, color=C_LGCN, zorder=3, edgecolor=SURFACE, linewidth=0.6, label="LightGCN lists" if i == 0 else None)
        ax.scatter([b], [i], s=16, color=C_BPR, zorder=3, edgecolor=SURFACE, linewidth=0.6, label="BPR lists" if i == 0 else None)
        ax.text(0.737, i, f"{b - a:+.2f}", va="center", ha="left", fontsize=FS - 0.5, color=INK2)
    lr_b, lr_l = _grid(BPR, "multi_hot")[0], _grid(LGCN, "multi_hot")[0]
    span = (-0.5, len(rows) - 0.5)  # keep the reference lines out of the legend row
    for v, c in ((lr_l, C_LGCN), (lr_b, C_BPR)):
        ax.vlines(v, *span, color=c, lw=0.7, ls=(0, (5, 3)))
    ax.vlines(0.5, *span, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.text(0.737, -0.85, "Δ", va="center", ha="left", fontsize=FS - 0.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlim(0.45, 0.765)
    ax.set_xlabel("Gender AUC of the LLM attacker")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(rows) - 0.5, -1.4)
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=0.7, ls=(0, (5, 3))))
    lab.append("LR, same lists")
    ax.legend(h, lab, loc="upper left", ncol=3, handletextpad=0.1, columnspacing=0.7)
    _panel(ax, "(a) LLM attacker on each recommender's lists")

    g = _genres()
    w = 0.36
    for k, (gender, shade) in enumerate((("female", INK2), ("male", "#c9c7bf"))):
        for i, run in enumerate((LGCN, BPR)):
            v = _romance_share(run / "lists.jsonl", g)[gender]
            ax2.bar(i + (k - 0.5) * w, v, width=w - 0.04, color=shade, zorder=2, label=gender if i == 0 else None)
            ax2.text(i + (k - 0.5) * w, v + 0.004, f"{v:.2f}", ha="center", fontsize=FS - 0.5, color=INK2)
    ax2.set_xticks([0, 1], ["LightGCN", "BPR"])
    ax2.set_ylabel("Romance share of list")
    ax2.set_ylim(0, 0.27)
    ax2.legend(loc="upper left", handlelength=1.0, handletextpad=0.3)
    _panel(ax2, "(b) Romance in the lists")

    arms = (("original", C_NEUTRAL, "o", "original"), ("romance_swap", C_GENDER, "o", "Romance swapped out"),
            ("control", MUTED, "s", "random movies swapped"))
    for k, run in enumerate((LGCN, BPR)):
        rep = _j(run / "romance_intervention/report_glm4_9b.json")
        for j, (arm, acol, mk, lab) in enumerate(arms):
            xpos = k + (j - 1) * 0.25
            if arm == "control":
                v = [rep[f"control_swap_s{s}"]["auc"] for s in (2024, 2025, 2026)]
                ax3.plot([xpos, xpos], [min(v), max(v)], color=acol, lw=1.2, zorder=2)
                y = float(np.mean(v))
            else:
                y = rep[arm]["auc"]
            ax3.scatter([xpos], [y], s=16, color=acol, edgecolor=SURFACE, linewidth=0.6, zorder=3, marker=mk,
                        label=lab if k == 0 else None)
        d = rep["romance_swap"]
        ci = d["delta_ci95"]
        ax3.text(k, 0.555 if k else 0.545, f"Δ {d['auc'] - rep['original']['auc']:+.3f}\n[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                 ha="center", va="top" if k else "bottom", fontsize=FS - 0.5, color=C_GENDER)
    ax3.set_xticks([0, 1], ["LightGCN", "BPR"])
    ax3.set_xlim(-0.55, 1.55)
    ax3.set_ylim(0.45, 0.76)
    ax3.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax3.set_ylabel("AUC, GLM-4-9B 0-shot")
    ax3.legend(loc="upper left", handletextpad=0.1)
    _panel(ax3, "(c) Removing Romance")
    _save(fig, out, "09_bpr_vs_lightgcn_llm")


# ------------------------------------------------------------------ 10 neutral -> gender
def fig_neutral_gender(out: Path) -> None:
    inputs = [("multi_hot", "T2\nmulti-hot"), ("genre_histogram", "T3 genre\nhistogram"),
              ("title_text_embedding", "T3 MiniLM\ntitles"), ("llm", "T4 LLM\n0-shot")]
    hist = _history()[0]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.6), sharey=True)
    for ax, (slug, name) in zip(axes, (("glm4_9b", "GLM-4-9B as recommender"), ("llama31_8b", "Llama-3.1-8B as recommender"))):
        w = 0.34
        for k, (variant, color) in enumerate((("neutral", C_NEUTRAL), ("gender", C_GENDER))):
            run = LLMREC / f"{variant}_req20_{slug}_unseen"
            for i, (inp, _) in enumerate(inputs):
                if inp == "llm":
                    v, ci = _preds_auc(run / f"llm_aia/preds_zero_shot_{slug}_500.jsonl")
                else:
                    v, ci = _grid(run, inp)
                _dot(ax, i + (k - 0.5) * w, v, ci[0], ci[1], color, f"{variant} prompt" if i == 0 else None)
        hr = [_j(LLMREC / f"{v}_req20_{slug}_unseen/llmrec_report.json")["rec_metrics"]["hit@10"]  # same lists as the AUCs
              for v in ("neutral", "gender")]
        ax.text(0.02, 0.03, f"HR@10: neutral {hr[0]:.3f}, gender {hr[1]:.3f}", transform=ax.transAxes,
                fontsize=FS, color=INK, va="bottom")
        ax.axhline(hist, color=INK2, lw=0.8, ls=(0, (6, 3)))
        ax.text(len(inputs) - 0.45, hist + 0.006, f"full history {hist:.2f}", fontsize=FS - 0.5, color=INK2, ha="right")
        # the recommender's own input: LR on the movies shown in its prompt (same for both prompts)
        prompt = _grid(LLMREC / f"neutral_req20_{slug}_unseen", "prompt_history_multi_hot")[0]
        ax.axhline(prompt, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.text(-0.45, prompt + 0.006, f"movies in the prompt {prompt:.2f}", fontsize=FS - 0.5, color=MUTED,
                ha="left", va="bottom", zorder=4, bbox=dict(fc=SURFACE, ec="none", pad=0.4, alpha=0.9))
        ax.set_xlim(-0.5, len(inputs) - 0.4)
        ax.set_xticks(range(len(inputs)), [lab for _, lab in inputs])
        _panel(ax, name)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    axes[0].set_ylim(0.45, 0.92)
    axes[0].set_ylabel("Gender AUC (95% CI)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, 0.885), ncol=2, handletextpad=0.1, columnspacing=0.8)
    _save(fig, out, "10_neutral_vs_gender")


# ------------------------------------------------------------------ 11 how the lists change
def _load_lists(path: Path) -> dict[str, dict]:
    return {r["user_id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def fig_romance_shift(out: Path) -> None:
    g = _genres()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.3), gridspec_kw={"width_ratios": [1.75, 1]})
    rows, jac = [], []
    for slug, name in (("glm4_9b", "GLM-4-9B"), ("llama31_8b", "Llama-3.1-8B")):
        a = _load_lists(LLMREC / f"neutral_req20_{slug}_unseen/lists.jsonl")
        b = _load_lists(LLMREC / f"gender_req20_{slug}_unseen/lists.jsonl")
        for gender in ("female", "male"):
            ra, rb, js = [], [], []
            for u, ua in a.items():
                ub = b[u]
                if ua["gender"] != gender or not ua["rec_item_ids"] or not ub["rec_item_ids"]:
                    continue
                ra.append(np.mean(["Romance" in g.get(i, set()) for i in ua["rec_item_ids"]]))
                rb.append(np.mean(["Romance" in g.get(i, set()) for i in ub["rec_item_ids"]]))
                sa, sb = set(ua["rec_item_ids"]), set(ub["rec_item_ids"])
                js.append(len(sa & sb) / len(sa | sb))
            ra, rb, js = map(np.array, (ra, rb, js))
            rows.append((f"{name}, {gender} (n={len(ra)})", ra.mean(), rb.mean(), _boot_mean_ci(rb - ra)))
            jac.append((name, gender, js.mean(), _boot_mean_ci(js)))
    for i, (_, a, b, ci) in enumerate(rows):
        ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=C_GENDER, lw=1.0,
                                                                 mutation_scale=7))
        ax.scatter([a], [i], s=16, color=C_NEUTRAL, zorder=3, label="neutral prompt" if i == 0 else None)
        ax.scatter([b], [i], s=16, color=C_GENDER, zorder=3, label="gender prompt" if i == 0 else None)
        ax.text(max(a, b) + 0.007, i, f"{b - a:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", va="center", fontsize=FS - 0.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.08, 0.37)
    ax.set_xlabel("Romance share of the list (change, paired 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", handletextpad=0.1)
    _panel(ax, "(a) Romance share")

    w = 0.36
    for k, (gender, shade) in enumerate((("female", INK2), ("male", "#c9c7bf"))):
        for i, name in enumerate(("GLM-4-9B", "Llama-3.1-8B")):
            _, _, m, ci = next(j for j in jac if j[0] == name and j[1] == gender)
            xpos = i + (k - 0.5) * w
            ax2.bar(xpos, m, width=w - 0.04, color=shade, zorder=2, label=gender if i == 0 else None)
            ax2.plot([xpos, xpos], ci, color=INK, lw=0.8, zorder=3)
            ax2.text(xpos, ci[1] + 0.012, f"{m:.2f}", ha="center", fontsize=FS - 0.5, color=INK2)
    ax2.set_xticks([0, 1], ["GLM-4-9B", "Llama-3.1-8B"])
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("Jaccard(neutral, gender list)")
    ax2.legend(loc="upper right", handlelength=1.0, handletextpad=0.3, ncol=2, columnspacing=0.8)
    _panel(ax2, "(b) List overlap (lower = rewritten more)")
    _save(fig, out, "11_romance_shift")


# ------------------------------------------------------------------ 12 age slices (single column)
def fig_age(out: Path) -> None:
    buckets = (("18-24", "18–24"), ("25-49", "25–49"), (">=50", "≥50"))
    rows = [("T0 user embedding", "T0 user embedding"), ("T2 multi-hot", "T2 multi-hot")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.3), sharey=True)
    n = {b: sum(1 for line in (LGCN / "lists.jsonl").open() if json.loads(line)["age"] == b) for b, _ in buckets}
    for ax, (row, title) in zip(axes, rows):
        for k, (run, color, name) in enumerate(((LGCN, C_LGCN, "LightGCN"), (BPR, C_BPR, "BPR"))):
            sl = _j(run / "age_slices/age_slices.json")["rows"][row]
            for i, (b, _) in enumerate(buckets):
                c = sl[b]
                _dot(ax, i + (k - 0.5) * 0.3, c["auc"], *c["auc_ci95"], color, name if i == 0 else None, text=False)
        ax.set_xticks(range(3), [f"{lab}\n({n[b]})" for b, lab in buckets])
        ax.set_xlim(-0.5, 2.5)
        _panel(ax, title)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    axes[0].set_ylabel("Gender AUC within age group")
    axes[0].set_ylim(0.45, 0.95)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 0.12), handletextpad=0.1)
    _save(fig, out, "12_age_slices")


# ------------------------------------------------------------------ 13 head films: watched vs recommended
TRAIN = Path("data/ml-1m-311/splits/train.csv")
SHORT_TITLE = {"1196": "Star Wars V", "260": "Star Wars IV", "1210": "Star Wars VI", "1270": "Back to the Future",
               "2028": "Saving Private Ryan", "2858": "American Beauty", "593": "Silence of the Lambs",
               "608": "Fargo", "1198": "Raiders of the Lost Ark", "527": "Schindler's List"}


def fig_head_masking(out: Path) -> None:
    """Head films: share of women / men who already watched each one vs share whose list contains it.

    Already-watched items are never recommended, so a film men have watched more often turns up more
    often in women's lists. Inset numbers: LR (balanced, trained on the 3,367-user pool) on a 10-d
    indicator of the head films, "in the list" vs "already watched" (the latter never sees the list).
    """
    import csv

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    watched: dict[str, set[str]] = {}
    with TRAIN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            watched.setdefault(r["user_id"], set()).add(r["item_id"])
    c_f, c_m = C_WOMEN, C_MEN
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.75), sharex=True, sharey=True)  # same 10 films, same order
    for ax, (run, name) in zip(axes, ((LGCN, "(a) LightGCN lists"), (BPR, "(b) BPR lists"))):
        head = _j(run / "popularity_ablation_500/popularity_ablation.json")["protocol"]["head_item_tokens"]
        te = [json.loads(line) for line in (run / "lists.jsonl").open(encoding="utf-8")]
        tr = [json.loads(line) for line in (run / "lists_train_pool.jsonl").open(encoding="utf-8")]

        def feats(rows, kind):
            return np.array([[(h in r["rec_item_ids"]) if kind == "list" else (h in watched.get(r["user_id"], ()))
                              for h in head] for r in rows], dtype=float)

        y_tr = np.array([r["gender"] == "male" for r in tr])
        y_te = np.array([r["gender"] == "male" for r in te])
        auc = {}
        for kind in ("list", "watched"):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(feats(tr, kind), y_tr)
            auc[kind] = roc_auc_score(y_te, clf.predict_proba(feats(te, kind))[:, 1])
        L, W = feats(te, "list"), feats(te, "watched")
        stats = [(h, W[~y_te, j].mean(), W[y_te, j].mean(), L[~y_te, j].mean(), L[y_te, j].mean())
                 for j, h in enumerate(head)]
        stats.sort(key=lambda s: s[2] - s[1], reverse=True)  # largest men-minus-women watched gap first
        for i, (h, wf, wm, lf, lm) in enumerate(stats):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
            if i == 0:  # name the two rows once, right of the top film's points
                ax.text(max(wf, wm) + 0.012, i - 0.2, "already watched", fontsize=FS - 0.5, color=MUTED,
                        style="italic", va="center")
                ax.text(max(lf, lm) + 0.012, i + 0.2, "in the top-10 list", fontsize=FS - 0.5, color=MUTED,
                        style="italic", va="center")
            for dy, (f_, m_, hollow) in ((-0.2, (wf, wm, True)), (0.2, (lf, lm, False))):
                y = i + dy
                ax.plot([f_, m_], [y, y], color=GRID, lw=1.6, zorder=1)
                for v, c in ((f_, c_f), (m_, c_m)):
                    ax.scatter([v], [y], s=15, zorder=3, linewidth=0.9,
                               facecolor=SURFACE if hollow else c, edgecolor=c)
        ax.set_yticks(range(len(stats)), [SHORT_TITLE.get(s[0], s[0]) for s in stats])
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_ylim(len(stats) - 0.5, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.15, 0.62)
        ax.set_xlabel("Share of test users")
        ax.text(0.99, 0.02, f"LR on the 10 movies, AUC\nin the list {auc['list']:.3f}\nalready watched {auc['watched']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=FS - 0.5, color=INK2, linespacing=1.3,
                bbox=dict(fc=SURFACE, ec=GRID, lw=0.5, boxstyle="round,pad=0.3"))
        _panel(ax, name)
    h = [Line2D([], [], marker="o", ls="none", ms=4, mfc=SURFACE, mec=INK2, mew=0.9),
         Line2D([], [], marker="o", ls="none", ms=4, mfc=INK2, mec=INK2),
         Line2D([], [], marker="s", ls="none", ms=4.5, mfc=c_f, mec=c_f),
         Line2D([], [], marker="s", ls="none", ms=4.5, mfc=c_m, mec=c_m)]
    fig.legend(h, ["already watched (train history)", "in the top-10 list", "women", "men"],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.savefig(out / "13_head_masking.png", dpi=300)
    fig.savefig(out / "13_head_masking.pdf")
    plt.close(fig)
    print("wrote", out / "13_head_masking.pdf")


FIGS = {
    "03_threat_model": fig_threat_model, "04_ladder": fig_ladder, "05_classifiers": fig_classifiers,
    "06_popularity": fig_popularity, "07_attacker_x_input": fig_matrix, "08_shots": fig_shots,
    "09_bpr_vs_lightgcn_llm": fig_bpr_vs_lgcn_llm, "10_neutral_vs_gender": fig_neutral_gender,
    "11_romance_shift": fig_romance_shift, "12_age_slices": fig_age, "13_head_masking": fig_head_masking,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("docs/report_figs"))
    p.add_argument("--only", nargs="*", choices=sorted(FIGS), help="figure names (default: all)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGS:
        FIGS[name](args.out_dir)


if __name__ == "__main__":
    main()
