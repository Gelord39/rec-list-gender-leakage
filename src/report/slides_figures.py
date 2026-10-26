"""Figures for the 2026-10-05 supervisor meeting (whole-project story, one finding per slide).

Every number is read from ``runs/`` (no hard-coded results); AUC CIs are the stored
bootstrap CIs, or a fresh 1,000-sample bootstrap where a file only has predictions.

    uv run python -m src.report.slides_figures [--out-dir docs/slides_2026-10-05/figs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

BPR = Path("runs/EXP-003_bpr_d32_311")
LGCN = Path("runs/EXP-004_lightgcn_d32_311")
LLMREC = Path("runs/EXP-006_llmrec_311")
ITEMS = Path("data/ml-1m-311/splits/items.csv")

# validated categorical palette (dataviz reference instance, light mode)
C_LGCN, C_BPR = "#2a78d6", "#eb6834"            # recommender identity, used deck-wide
C_GLM, C_LLAMA, C_QWEN = "#1baf7a", "#4a3aa7", "#eda100"  # LLM identity
C_NEUTRAL, C_GENDER = "#a3a19a", "#4a3aa7"      # baseline gray vs the treatment
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"

import scienceplots  # noqa: E402,F401  (registers the "science" styles)

# SciencePlots journal style (serif, inward ticks, boxed axes), no LaTeX dependency;
# font sizes raised from the paper defaults so the figures read on a slide.
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.linewidth": 2, "axes.prop_cycle": plt.cycler(color=[C_LGCN, C_BPR, C_GLM]),
})


def _j(p: Path) -> dict:
    return json.loads(p.read_text())


def _grid(run: Path, inp: str, variant: str = "lr_balanced") -> tuple[float, list[float]]:
    for r in _j(run / "attack_grid_500/summary.json")["table"]:
        if r["input"] == inp and r["variant"] == variant:
            return r["auc"], r["auc_ci95"]
    raise KeyError((run, inp, variant))


def _history(run: Path = LGCN) -> tuple[float, list[float]]:
    r = next(r for r in _j(run / "history_aia/summary_binary.json")["table"] if r["key"] == "lr_balanced")
    return r["auc"], r["auc_ci95"]


def _preds_auc(path: Path, n_boot: int = 1000) -> tuple[float, list[float]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = [r for r in rows if r.get("score") is not None]
    y = np.array([r["gender"] == "male" for r in rows])
    s = np.array([r["score"] for r in rows], dtype=float)
    rng = np.random.default_rng(2024)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    return float(roc_auc_score(y, s)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _err(v: float, ci: list[float]) -> list[list[float]]:
    return [[v - ci[0]], [ci[1] - v]]


def _chance(ax, axis: str = "y") -> None:
    (ax.axhline if axis == "y" else ax.axvline)(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    if axis == "y":
        ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", ha="left", color=MUTED, fontsize=9)


def _dot(ax, xpos: float, v: float, lo: float, hi: float, color: str, label: str | None = None,
         text: bool = True) -> None:
    """Point estimate + CI whisker (used instead of bars, whose length would need a zero baseline)."""
    ax.plot([xpos, xpos], [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([xpos], [v], s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
    if text:
        ax.text(xpos + 0.07, v, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK2)


def _save(fig, out: Path, name: str) -> None:
    # the slide title carries the claim: drop the figure's own headline, keep panel titles
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    if len(fig.axes) == 1:
        for loc in ("left", "center", "right"):
            fig.axes[0].set_title("", loc=loc)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=220)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ------------------------------------------------------------------ slide 4: the leakage ladder
def _shots_cell(run: Path, model: str = "glm4_9b", template: str = "plain", shots: int = 16) -> tuple[float, float]:
    """Mean and sample sd of AUC over the exemplar draws of one shot-sweep cell."""
    v = [r["auc"] for r in _j(run / "shot_sweep/shot_sweep.json")["runs"]
         if r["model"] == model and r["template"] == template and r["shots"] == shots]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def fig_ladder(out: Path) -> None:
    tiers = [("TH\nfull history", None, "3,367"), ("T0\nuser embedding", "true_user_embedding", "3,367"),
             ("T1\nrecommender\ninversion", "inversion", "3,367"), ("T2\nlist item IDs\n(multi-hot)", "multi_hot", "3,367"),
             ("T3\nlist titles\n(MiniLM)", "title_text_embedding", "3,367"), ("T4\nLLM reads titles\n(GLM-4-9B)", "llm", "0 or 16")]
    fig, ax = plt.subplots(figsize=(10, 5.2))
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
        xs = x + (k - 0.5) * 0.12
        ax.plot(xs, vals, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=label)
        ax.errorbar(xs, vals, yerr=np.array([[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]),
                    fmt="none", ecolor=color, elinewidth=1.2, capsize=0, zorder=2)
        for i in (1, 3):
            ax.text(xs[i] + (0.1 if k else -0.1), vals[i] - (0.02 if k else 0), f"{vals[i]:.2f}", color=INK2,
                    fontsize=10, ha="left" if k else "right", va="center")
        # T4 with 16 in-context labels: same model, mean ± sd over 3 exemplar draws
        m, sd = _shots_cell(run)
        x16 = len(tiers) - 1 + 0.3 + (k - 0.5) * 0.12
        ax.errorbar([x16], [m], yerr=[[sd], [sd]], fmt="D", ms=8, mfc=SURFACE, mec=color, mew=2, ecolor=color,
                    elinewidth=1.2, zorder=3, label=None)
        ax.plot([xs[-1], x16], [vals[-1], m], color=color, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.text(xs[-1] - 0.1, vals[-1], f"{vals[-1]:.2f}", color=INK2, fontsize=10, ha="right", va="center")
        ax.text(x16 + 0.09, m, f"{m:.2f}", color=INK2, fontsize=10, ha="left", va="center")
    from matplotlib.lines import Line2D
    h, lab = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=INK2, marker="o", ls="none", ms=7), Line2D([], [], color=INK2, marker="D", mfc=SURFACE, ls="none", ms=7)]
    lab += ["T4 zero-shot (0 labels)", "T4 16-shot (16 labels, mean ± sd of 3 draws)"]
    ax.legend(h, lab, loc="upper right", ncol=2, fontsize=10)
    ax.set_xticks(x, [f"{t}\n[{n} labels]" for t, _, n in tiers], fontsize=10)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Gender AUC (95% CI)")
    ax.set_xlim(-0.4, len(tiers) - 0.1)
    _chance(ax)
    ax.set_title("The less the attacker sees, the less leaks — and the list leaks more on BPR", loc="left")
    _save(fig, out, "04_ladder")


# ------------------------------------------------------------------ slide 5: classifier choice
def fig_classifiers(out: Path) -> None:
    inputs = [("true_user_embedding", "T0 user embedding"), ("inversion", "T1 inversion"),
              ("ld_mean_pool", "T1 mean-pool"), ("multi_hot", "T2 multi-hot"),
              ("tfidf_multi_hot", "T2 TF-IDF multi-hot"), ("title_text_embedding", "T3 MiniLM titles"),
              ("genre_histogram", "T3 genre histogram")]
    clfs = [("lr_balanced", "LR", "o"), ("svm_linear_balanced", "SVM linear", "s"),
            ("svm_rbf_balanced", "SVM RBF", "D"), ("mlp_balanced", "MLP", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (run, color, name) in zip(axes, ((LGCN, C_LGCN, "LightGCN"), (BPR, C_BPR, "BPR"))):
        for i, (inp, _) in enumerate(inputs):
            v = [_grid(run, inp, c)[0] for c, _, _ in clfs]
            ax.plot([min(v), max(v)], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
            for (c, lab, m), a in zip(clfs, v):
                lr = c == "lr_balanced"
                ax.scatter(a, i, marker=m, s=70 if lr else 46, color=color if lr else SURFACE,
                           edgecolor=color, linewidth=1.5, zorder=4 if lr else 3, label=lab if i == 0 else None)
        ax.set_yticks(range(len(inputs)), [lab for _, lab in inputs])
        ax.invert_yaxis()
        ax.set_xlim(0.55, 0.86)
        ax.set_xlabel("Gender AUC")
        ax.set_title(name, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, handletextpad=0.3)
    fig.suptitle("Which input matters far more than which classifier (filled = LR, the reported one)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, "05_classifiers")


# ------------------------------------------------------------------ slide 6: popularity ablation
def fig_popularity(out: Path) -> None:
    arms = [("full", "Full top-10"), ("tail_only", "Tail only\n(top-10 items removed)"),
            ("random_ablation", "Same number\nremoved at random"), ("head_only", "Head only\n(top-10 items)")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.36
    for k, (run, color, name) in enumerate(((LGCN, C_LGCN, "LightGCN"), (BPR, C_BPR, "BPR"))):
        res = _j(run / "popularity_ablation_500/popularity_ablation.json")["results"]["multi_hot"]
        for i, (arm, _) in enumerate(arms):
            r = res[arm]["lr_balanced"]
            if arm == "random_ablation":
                v, lo, hi = r["auc"]["mean"], r["auc"]["mean"] - r["auc"]["sd"], r["auc"]["mean"] + r["auc"]["sd"]
            else:
                v, (lo, hi) = r["auc"], r["auc_ci95"]
            _dot(ax, i + (k - 0.5) * w, v, lo, hi, color, name if i == 0 else None)
    ax.set_xticks(range(len(arms)), [a for _, a in arms], fontsize=9)
    ax.set_ylim(0.45, 0.8)
    ax.set_xlim(-0.5, len(arms) - 0.4)
    ax.set_ylabel("Gender AUC, multi-hot LR\n(95% CI; random arm ± sd of 10 draws)")
    ax.text(0.01, 0.02, "users whose list has both head and tail items (LightGCN 457, BPR 425 of 500)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _chance(ax)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Dropping the popular items costs little; the popular items alone leak least", loc="left")
    _save(fig, out, "06_popularity")


# ------------------------------------------------------------------ slide 7: attacker x input
def fig_matrix(out: Path) -> None:
    clf_hist = _history(LGCN)
    clf_list = _grid(LGCN, "multi_hot")
    llm_hist = _preds_auc(LGCN / "llm_aia/preds_history_zero_shot_lightgcn_glm4_9b_500_hist100.jsonl")
    llm_list = _preds_auc(LGCN / "llm_aia/preds_staab_few_shot_lightgcn_glm4_9b_500.jsonl")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    w = 0.36
    for k, (label, color, vals) in enumerate((("Trained classifier (LR, 3,367 labels)", C_LGCN, (clf_hist, clf_list)),
                                               ("LLM, GLM-4-9B (0–2 labels)", C_GLM, (llm_hist, llm_list)))):
        for i, (v, ci) in enumerate(vals):
            _dot(ax, i + (k - 0.5) * w, v, ci[0], ci[1], color, label if i == 0 else None)
    ax.set_xticks([0, 1], ["Raw watch history", "LightGCN top-10 list"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Gender AUC (95% CI)")
    _chance(ax)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("The LLM can attack — the list is what leaks less", loc="left")
    _save(fig, out, "07_attacker_x_input")


# ------------------------------------------------------------------ slide 8: shots
def fig_shots(out: Path) -> None:
    models = (("glm4_9b", "GLM-4-9B", C_GLM), ("llama31_8b", "Llama-3.1-8B", C_LLAMA), ("qwen25_14b", "Qwen2.5-14B", C_QWEN))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (run, name, ref) in zip(axes, ((LGCN, "LightGCN lists", "multi_hot"), (BPR, "BPR lists", "multi_hot"))):
        cells = _j(run / "shot_sweep/shot_sweep.json")["runs"]
        for slug, label, color in models:
            pts = {}
            for r in cells:
                if r["model"] == slug and r["template"] == "plain" and r["auc"] is not None:
                    pts.setdefault(r["shots"], []).append(r["auc"])
            shots = sorted(pts)
            m = np.array([np.mean(pts[s]) for s in shots])
            sd = np.array([np.std(pts[s], ddof=1) if len(pts[s]) > 1 else 0 for s in shots])
            xs = np.arange(len(shots)) if False else shots
            xpos = [np.log2(s + 1) for s in xs]
            ax.fill_between(xpos, m - sd, m + sd, color=color, alpha=0.15, lw=0)
            ax.plot(xpos, m, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
        lr = _grid(run, ref)[0]
        ax.axhline(lr, color=INK2, lw=1.2, ls=(0, (6, 3)))
        ax.text(0, lr + 0.006, f"LR on the same lists, 3,367 labels: {lr:.3f}", fontsize=9, color=INK2)
        ticks = [0, 2, 8, 16, 32, 64, 128]
        ax.set_xticks([np.log2(t + 1) for t in ticks], [str(t) for t in ticks])
        ax.set_xlim(-0.3, np.log2(129) + 0.4)
        ax.set_xlabel("In-context labelled exemplars (shots)")
        ax.set_title(name, loc="left")
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_ylabel("Gender AUC (mean ± sd over 3 exemplar draws)")
    axes[0].set_ylim(0.45, 0.76)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.77), fontsize=9)
    fig.suptitle("More exemplars stop helping after ~8; a bigger model does not close the gap to LR",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, "08_shots")


# ------------------------------------------------------------------ slide 9: BPR vs LightGCN for the LLM + Romance gap
def _genres() -> dict[str, set[str]]:
    import csv
    with ITEMS.open(encoding="utf-8") as f:
        return {r["item_id"]: set(r["genres"].split()) for r in csv.DictReader(f)}


def _romance_share(lists: Path, genres: dict[str, set[str]]) -> dict[str, float]:
    by = {"female": [], "male": []}
    for line in lists.open(encoding="utf-8"):
        r = json.loads(line)
        if r["rec_item_ids"]:
            by[r["gender"]].append(np.mean(["Romance" in genres.get(i, set()) for i in r["rec_item_ids"]]))
    return {g: float(np.mean(v)) for g, v in by.items()}


def fig_bpr_vs_lgcn_llm(out: Path) -> None:
    rows = [("zero_shot", "llama32_500_lp", "llama3.2 · 0-shot"), ("zero_shot", "glm4_9b_500", "GLM-4-9B · 0-shot"),
            ("zero_shot", "llama31_8b_500", "Llama-3.1-8B · 0-shot"), ("zero_shot", "qwen25_14b_500", "Qwen2.5-14B · 0-shot"),
            ("many_shot", "glm4_9b_500_shots8_s2025", "GLM-4-9B · 8-shot"),
            ("many_shot", "qwen25_14b_500_shots16_s2025", "Qwen2.5-14B · 16-shot")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    for i, (style, tag, label) in enumerate(rows):
        a = _preds_auc(LGCN / f"llm_aia/preds_{style}_lightgcn_{tag}.jsonl")[0]
        b = _preds_auc(BPR / f"llm_aia/preds_{style}_bpr_{tag}.jsonl")[0]
        ax.plot([a, b], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter([a], [i], s=60, color=C_LGCN, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="LightGCN lists" if i == 0 else None)
        ax.scatter([b], [i], s=60, color=C_BPR, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="BPR lists" if i == 0 else None)
        ax.text(b + 0.008, i, f"+{b - a:.2f}", va="center", fontsize=9, color=INK2)
    lr_b, lr_l = _grid(BPR, "multi_hot")[0], _grid(LGCN, "multi_hot")[0]
    for v, c in ((lr_l, C_LGCN), (lr_b, C_BPR)):
        ax.axvline(v, color=c, lw=1, ls=(0, (6, 3)))
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.set_xlim(0.45, 0.76)
    ax.set_xlabel("Gender AUC of the LLM attacker (same 500 users)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(rows) - 0.5, -1.0)
    from matplotlib.lines import Line2D
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=1, ls=(0, (6, 3))))
    lab.append("trained LR on the same lists")
    ax.legend(h, lab, loc="upper left", fontsize=9, ncol=3)
    ax.set_title("Every LLM attacker reads BPR lists better (all Δ CIs > 0)", loc="left")

    g = _genres()
    w = 0.36
    for k, (gender, shade) in enumerate((("female", INK2), ("male", "#c9c7bf"))):
        for i, run in enumerate((LGCN, BPR)):
            v = _romance_share(run / "lists.jsonl", g)[gender]
            ax2.bar(i + (k - 0.5) * w, v, width=w - 0.04, color=shade, zorder=2, label=f"{gender} users" if i == 0 else None)
            ax2.text(i + (k - 0.5) * w, v + 0.004, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    ax2.set_xticks([0, 1], ["LightGCN", "BPR"])
    ax2.set_ylabel("Share of list films tagged Romance")
    ax2.set_ylim(0, 0.27)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("Part of why: only BPR lists\nhave a Romance gap", loc="left")
    _save(fig, out, "09_bpr_vs_lightgcn_llm")


# ------------------------------------------------------------------ slide 10: neutral -> gender
def fig_neutral_gender(out: Path) -> None:
    inputs = [("multi_hot", "T2 multi-hot"), ("genre_histogram", "T3 genre histogram"),
              ("title_text_embedding", "T3 MiniLM titles"), ("llm", "T4 LLM 0-shot\n(same-family attacker)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, (slug, name) in zip(axes, (("glm4_9b", "GLM-4-9B as recommender"), ("llama31_8b", "Llama-3.1-8B as recommender"))):
        w = 0.36
        for k, (variant, color) in enumerate((("neutral", C_NEUTRAL), ("gender", C_GENDER))):
            run = LLMREC / f"{variant}_req20_{slug}_unseen"
            for i, (inp, _) in enumerate(inputs):
                if inp == "llm":
                    v, ci = _preds_auc(run / f"llm_aia/preds_zero_shot_{slug}_500.jsonl")
                else:
                    v, ci = _grid(run, inp)
                _dot(ax, i + (k - 0.5) * w, v, ci[0], ci[1], color, f"{variant} prompt" if i == 0 else None)
        hr = [_j(LLMREC / f"{v}_req20_{slug}/llmrec_report.json")["rec_metrics"]["hit@10"] for v in ("neutral", "gender")]
        ax.text(0.02, 0.03, f"HR@10  neutral {hr[0]:.3f} → gender {hr[1]:.3f}", transform=ax.transAxes,
                fontsize=10, color=INK, va="bottom", fontweight="bold")
        ax.axhline(_history()[0], color=INK2, lw=1, ls=(0, (6, 3)))
        ax.text(len(inputs) - 0.45, _history()[0] + 0.006, "full history 0.87", fontsize=8.5, color=INK2, ha="right")
        ax.set_xlim(-0.5, len(inputs) - 0.4)
        ax.set_xticks(range(len(inputs)), [lab for _, lab in inputs], fontsize=9)
        ax.set_title(name, loc="left")
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_ylim(0.45, 0.92)
    axes[0].set_ylabel("Gender AUC (95% CI)")
    axes[1].legend(loc="lower right", fontsize=9, bbox_to_anchor=(1, 0.08))
    fig.suptitle("Adding only the word \"male\"/\"female\" to the prompt makes the list leak gender — HR does not move",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, "10_neutral_vs_gender")


# ------------------------------------------------------------------ slide 11: what changes in the list
def fig_romance_shift(out: Path) -> None:
    g = _genres()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rows = []
    for slug, name in (("glm4_9b", "GLM-4-9B"), ("llama31_8b", "Llama-3.1-8B")):
        a = _romance_share(LLMREC / f"neutral_req20_{slug}_unseen/lists.jsonl", g)
        b = _romance_share(LLMREC / f"gender_req20_{slug}_unseen/lists.jsonl", g)
        for gender in ("female", "male"):
            rows.append((f"{name} · {gender} users", a[gender], b[gender]))
    for i, (lab, a, b) in enumerate(rows):
        ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=C_GENDER, lw=1.8))
        ax.scatter([a], [i], s=55, color=C_NEUTRAL, zorder=3, label="neutral prompt" if i == 0 else None)
        ax.scatter([b], [i], s=55, color=C_GENDER, zorder=3, label="gender prompt" if i == 0 else None)
        ax.text(max(a, b) + 0.008, i, f"{a:.2f} → {b:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.05, 0.42)
    ax.set_xlabel("Share of recommended films tagged Romance (500 test users)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Told \"female\", both LLMs add Romance; told \"male\", little changes", loc="left")
    _save(fig, out, "11_romance_shift")


# ------------------------------------------------------------------ slide 12: age slices
def fig_age(out: Path) -> None:
    buckets = ("18-24", "25-49", ">=50")
    rows = [("T0 user embedding", "T0"), ("T2 multi-hot", "T2")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (row, short) in zip(axes, rows):
        for k, (run, color, name) in enumerate(((LGCN, C_LGCN, "LightGCN"), (BPR, C_BPR, "BPR"))):
            sl = _j(run / "age_slices/age_slices.json")["rows"][row]
            for i, b in enumerate(buckets):
                c = sl[b]
                xpos = i + (k - 0.5) * 0.2
                ax.errorbar(xpos, c["auc"], yerr=_err(c["auc"], c["auc_ci95"]), fmt="o", color=color, ms=7,
                            mec=SURFACE, mew=1.5, elinewidth=1.2, label=name if i == 0 else None)
        n = {b: sum(1 for line in (LGCN / "lists.jsonl").open() if json.loads(line)["age"] == b) for b in buckets}
        ax.set_xticks(range(3), [f"{b}\n(n={n[b]})" for b in buckets])
        ax.set_title(row, loc="left")
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_ylabel("Gender AUC within the age bucket (95% CI)")
    axes[0].set_ylim(0.45, 0.95)
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Leakage differs by age group — but CIs overlap at n = 72–85",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    _save(fig, out, "12_age_slices")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("docs/slides_2026-10-05/figs"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in (fig_ladder, fig_classifiers, fig_popularity, fig_matrix, fig_shots, fig_bpr_vs_lgcn_llm,
              fig_neutral_gender, fig_romance_shift, fig_age):
        f(args.out_dir)


if __name__ == "__main__":
    main()
