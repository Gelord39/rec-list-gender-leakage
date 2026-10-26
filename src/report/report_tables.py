"""LaTeX tables for the report that are computed from ``runs/`` (printed to stdout and
written to ``docs/report_tables/``).

- ``concentration``: how concentrated each CF recommender's top-10 lists are, over all
  3,867 users (500 attack test users + 3,367 training pool): distinct movies recommended,
  catalogue coverage, and the share of list slots taken by the 10 / 50 most-recommended movies.
- ``effectiveness``: HR@10 / NDCG@10 of every recommender in the report, with the women - men
  gap and its 95% CI (from ``runs/effectiveness_by_gender/``).
- ``effectiveness_change``: neutral -> gender prompt change of HR@10 / NDCG@10 per gender,
  paired on the same users (appendix).
- ``nonanswers``: LLM attacker non-answers on the 500 test users, by model and prompt:
  refusals (no answer at all, e.g. "I can't provide ...") and abstentions (an answer slot
  that is neither male nor female, e.g. "Gender: Indeterminate"; these still get a
  log-probability score, so they count for AUC but not for BAcc).
- ``genre_gaps``: per-genre share of women's minus men's top-10 lists (500 test users) for
  BPR and LightGCN, and its absolute sum over the 18 genres (Section 5.3.3; text only, not
  a table in the report).

    uv run python -m src.report.report_tables
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

BPR = Path("runs/EXP-003_bpr_d32_311")
LGCN = Path("runs/EXP-004_lightgcn_d32_311")
N_ITEMS = 3094  # items after preprocessing (stats table)
OUT = Path("docs/report_tables")

REFUSAL = re.compile(r"^\s*(I can['’]t|I cannot|I can not|I won['’]t|I'm not able|I am not able|I'm unable|I am unable)",
                     re.IGNORECASE)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def concentration() -> str:
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Concentration of the CF recommenders' top-10 lists over all 3,867 users: number of distinct "
             r"movies recommended, share of the 3,094-movie catalogue, and share of all list slots filled by the 10 "
             r"and 50 most-recommended movies.}",
             r"\label{tab:concentration}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Recommender & Distinct movies & Coverage & Top-10 share & Top-50 share \\", r"\midrule"]
    for run, name in ((BPR, "BPR"), (LGCN, "LightGCN")):
        lists = [r["rec_item_ids"] for f in ("lists.jsonl", "lists_train_pool.jsonl") for r in _rows(run / f)]
        c = Counter(i for lst in lists for i in lst)
        slots = sum(c.values())
        top = [n for _, n in c.most_common()]
        lines.append(f"{name} & {len(c):,} & {len(c) / N_ITEMS:.1%} & {sum(top[:10]) / slots:.1%} & "
                     f"{sum(top[:50]) / slots:.1%} \\\\".replace("%", r"\%"))
        print(f"  {name}: {len(lists)} users, {slots} slots")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _count(path: Path) -> tuple[int, int, int]:
    rows = _rows(path)
    labelled = [r for r in rows if r.get("pred_gender") in ("male", "female")]
    other = [r for r in rows if r.get("pred_gender") not in ("male", "female")]
    refusals = sum(bool(REFUSAL.match(r.get("raw_response") or "")) for r in other)
    return len(rows), refusals, len(other) - refusals


def nonanswers() -> str:
    models = [("llama3.2 (3B)", "llama32_500_lp"), ("GLM-4-9B", "glm4_9b_500"),
              ("Llama-3.1-8B", "llama31_8b_500"), ("Qwen2.5-14B", "qwen25_14b_500")]
    styles = [("zero_shot", "zero-shot"), ("few_shot", "few-shot (2)"), ("staab", "Staab"),
              ("staab_few_shot", "Staab few-shot (2)")]
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Non-answers of the LLM attacker on the 500 test users, shown as refusals / abstentions. "
             r"A refusal gives no answer (e.g.\ ``I can't provide \dots''); an abstention answers neither male nor "
             r"female (e.g.\ \emph{Gender: Indeterminate}) but still receives a log-probability score, so it counts "
             r"towards AUC but not towards balanced accuracy.}",
             r"\label{tab:nonanswers}",
             r"\begin{tabular}{llcc}", r"\toprule",
             r"Model & Prompt & LightGCN lists & BPR lists \\", r"\midrule"]
    for mname, tag in models:
        first = True
        for style, sname in styles:
            cells = []
            for run, prefix in ((LGCN, "lightgcn"), (BPR, "bpr")):
                p = run / f"llm_aia/preds_{style}_{prefix}_{tag}.jsonl"
                if not p.exists():
                    cells.append(None)
                    continue
                n, ref, abst = _count(p)
                cells.append(f"{ref} / {abst}")
            if all(c is None for c in cells):
                continue
            lines.append(f"{mname if first else ''} & {sname} & " + " & ".join(c or "--" for c in cells) + r" \\")
            first = False
    lines.append(r"\midrule")
    for style, sname in (("history_zero_shot", "history, zero-shot"), ("history_staab", "history, Staab")):
        n, ref, abst = _count(LGCN / f"llm_aia/preds_{style}_lightgcn_glm4_9b_500_hist100.jsonl")
        lines.append(f"GLM-4-9B & {sname} & \\multicolumn{{2}}{{c}}{{{ref} / {abst} (watch history)}} \\\\")
    # many-shot runs: report the maximum over all draws and shot counts
    worst = (0, 0)
    for run in (LGCN, BPR):
        for p in (run / "llm_aia").glob("preds_*many_shot_*_500_shots*.jsonl"):
            _, ref, abst = _count(p)
            worst = max(worst, (ref, abst), key=sum)
    lines.append(f"all models & many-shot (8--128) & \\multicolumn{{2}}{{c}}{{at most {worst[0]} / {worst[1]} per run}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


EFF = Path("runs/effectiveness_by_gender/effectiveness_by_gender.json")
EFF_ROWS = [("BPR", "BPR", "CF"), ("LightGCN", "LightGCN", "CF"),
            ("GLM-4-9B LLMRec · neutral", "GLM-4-9B, neutral prompt", "LLM recommender"),
            ("GLM-4-9B LLMRec · gender", "GLM-4-9B, gender prompt", "LLM recommender"),
            ("Llama-3.1-8B LLMRec · neutral", "Llama-3.1-8B, neutral prompt", "LLM recommender"),
            ("Llama-3.1-8B LLMRec · gender", "Llama-3.1-8B, gender prompt", "LLM recommender")]


def _ci(v: float, ci: list[float]) -> str:
    return f"${v:+.3f}$ [${ci[0]:+.3f}$, ${ci[1]:+.3f}$]"


def effectiveness() -> str:
    d = json.loads(EFF.read_text())
    n_f = d["recommenders"]["BPR"]["n_female"]
    n_m = d["recommenders"]["BPR"]["n_male"]
    lines = [r"\begin{table*}[t]", r"\centering\small",
             rf"\caption{{Effectiveness of every recommender on the {n_f + n_m} users with test interactions "
             rf"({n_f} women, {n_m} men), and the gap between women and men (women $-$ men, 95\% bootstrap CI, "
             r"users resampled within each gender). LLM-recommender rows use the lists with already-watched "
             r"movies removed. No gap is significant.}",
             r"\label{tab:effectiveness}",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Recommender & Type & HR@10 & NDCG@10 & HR@10 gap & NDCG@10 gap \\", r"\midrule"]
    for key, name, kind in EFF_ROWS:
        r = d["recommenders"][key]
        lines.append(f"{name} & {kind} & {r['hr']['all']:.3f} & {r['ndcg']['all']:.3f} & "
                     f"{_ci(r['hr']['gap_f_minus_m'], r['hr']['gap_ci95'])} & "
                     f"{_ci(r['ndcg']['gap_f_minus_m'], r['ndcg']['gap_ci95'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def effectiveness_change() -> str:
    d = json.loads(EFF.read_text())["neutral_to_gender"]
    lines = [r"\begin{table*}[t]", r"\centering\small",
             r"\caption{Change in effectiveness from the neutral to the gender prompt, paired on the same users, "
             r"for women and men, and the difference between the two changes (women $-$ men); 95\% bootstrap "
             r"CIs. No change is significant.}",
             r"\label{tab:effectiveness-change}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"LLM recommender & Metric & Change, women & Change, men & Women $-$ men \\", r"\midrule"]
    for key, name in (("GLM-4-9B LLMRec", "GLM-4-9B"), ("Llama-3.1-8B LLMRec", "Llama-3.1-8B")):
        for m, mname in (("hr", "HR@10"), ("ndcg", "NDCG@10")):
            r = d[key][m]
            lines.append(f"{name if m == 'hr' else ''} & {mname} & {_ci(r['change_female'], r['change_female_ci95'])} & "
                         f"{_ci(r['change_male'], r['change_male_ci95'])} & {_ci(r['diff_f_minus_m'], r['diff_ci95'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def genre_gaps() -> str:
    from src.report.slides_figures import _genres
    g = _genres()
    names = sorted({x for s in g.values() for x in s})
    lines = ["Genre share of the top-10 list, women - men (percentage points), 500 test users"]
    for run, label in ((BPR, "BPR"), (LGCN, "LightGCN")):
        by = {"female": [], "male": []}
        for r in _rows(run / "lists.jsonl"):
            items = r["rec_item_ids"]
            by[r["gender"]].append([sum(n in g.get(i, set()) for i in items) / len(items) for n in names])
        gap = [100 * (sum(f[k] for f in by["female"]) / len(by["female"]) - sum(m[k] for m in by["male"]) / len(by["male"]))
               for k in range(len(names))]
        top = sorted(zip(gap, names), key=lambda t: -abs(t[0]))[:6]
        lines.append(f"{label}: sum |gap| = {sum(abs(x) for x in gap) / 100:.2f}; "
                     + ", ".join(f"{n} {x:+.1f}" for x, n in top))
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("concentration", concentration), ("nonanswers", nonanswers),
                     ("effectiveness", effectiveness), ("effectiveness_change", effectiveness_change),
                     ("genre_gaps", genre_gaps)):
        tex = fn()
        (OUT / f"{name}.tex").write_text(tex + "\n")
        print(f"--- {name}\n{tex}\n")


if __name__ == "__main__":
    main()
