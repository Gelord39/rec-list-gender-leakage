"""Diagnostics for the embedding attackers: MLP loss curves + learning curves.

Not a metric logged per production run — a one-off health check for whether
the list-derived attacker's near-chance BAcc (see CLAUDE.md, "List-derived
embedding attacker") is a training issue or a genuine lack of signal in the
input. Motivated by a real observation: every fold of the list-derived MLP
attack raised ``ConvergenceWarning: ... Maximum iterations (500) reached and
the optimization hasn't converged yet`` — so its near-chance BAcc could be
"no signal in mean-pooled list embeddings", or it could be "the MLP got cut
off before it finished learning". A loss curve tells these apart; a learning
curve extends the same question to LR/SVM, which don't expose a per-iteration
loss through sklearn's API (lbfgs and libsvm's QP solver don't report one).

Uses the SAME training pool as the fixed single train/test split protocol
(``src/attack/fixed_split_aia.py``, 2026-09-07): the users NOT in the fixed
test set (``--test-lists``), so what's diagnosed here matches what the
production fixed-split attacker actually trains on. Does not touch the held-
out test set at all — both plots are purely about the training pool's own
internal behavior (loss over iterations; a fresh internal CV over the
training pool for the learning curve), consistent with never letting the
test set influence anything other than final evaluation.

Produces three figures per run:
  1. MLP loss curve (real per-iteration training loss) — full-embedding vs
     list-derived embedding, side by side. Fit with a higher max_iter than
     production (default 2000 vs the production 500) so the plot shows
     whether/where it actually converges; a vertical line marks the
     production cutoff (500) for reference.
  2. Learning curve (BAcc vs training-set size, via
     ``sklearn.model_selection.learning_curve``) for MLP / LR / SVM, for both
     input types — 2x3 grid. Works uniformly across all three classifiers
     even though only MLP has an iteration-level loss curve.
  3. The same learning curve scored by threshold-independent ROC AUC. This
     distinguishes weak ranking signal from default-threshold collapse under
     class imbalance.

Example (run after the fixed-split pipeline's inputs exist):
  uv run python -m src.attack.attack_diagnostics \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-Aug-30-2026_23-00-46.pth \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --run-dir runs/EXP-004_lightgcn_d32_311 \\
    --test-lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \\
    --train-pool-lists runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \\
    --out-dir runs/EXP-004_lightgcn_d32_311/diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.attack.embedding_aia import _CLASSIFIER_BUILDERS, load_attack_table
from src.attack.fixed_split_aia import _read_user_ids, split_by_user_ids
from src.attack.list_embedding_aia import build_list_embeddings
from src.export.export_lists import _load_model
from src.train.train_recbole import get_user_item_embeddings


POOL_TITLES = {
    "full_embedding": "Full user embedding",
    "list_derived_mean": "List-derived, mean-pool",
    "list_derived_rank": "List-derived, rank-pool",
}


def _load_training_pool(
    config_file: Path,
    checkpoint: Path,
    model_name: str,
    users_csv: Path,
    run_dir: Path,
    test_lists_path: Path,
    train_pool_lists_path: Path,
    device: str | None,
    *,
    poolings: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {"full_embedding": (X, y_raw), "list_derived_<pooling>": (X, y_raw), ...}
    for the training pool only (mirrors
    ``fixed_split_aia.run_fixed_split_pipeline``'s train side, minus the
    held-out test set). ``poolings`` defaults to ``["mean"]``; pass
    ``["mean", "rank"]`` to compare both."""
    run_dir = Path(run_dir)
    poolings = poolings or ["mean"]
    test_user_ids = _read_user_ids(test_lists_path)

    X, y_raw, user_ids = load_attack_table(
        run_dir / "user_embeddings.npy",
        run_dir / "user_ids.npy",
        Path(users_csv),
        attribute="gender",
    )
    X_train, y_train, _, _, _, _ = split_by_user_ids(X, y_raw, user_ids, test_user_ids)

    _, dataset, model, _ = _load_model(config_file, checkpoint, device or "cpu", model_name=model_name)
    _, item_weight = get_user_item_embeddings(model)

    pool: dict[str, tuple[np.ndarray, np.ndarray]] = {"full_embedding": (X_train, y_train)}
    for pooling in poolings:
        X_train_list, y_train_list, _ = build_list_embeddings(
            train_pool_lists_path, item_weight, dataset, pooling=pooling
        )
        pool[f"list_derived_{pooling}"] = (X_train_list, y_train_list)
    return pool


def fit_mlp_loss_curve(
    X: np.ndarray, y_raw: np.ndarray, *, seed: int = 2024, max_iter: int = 2000
) -> list[float]:
    """Fit a fresh MLP (same architecture as the production attacker, just a
    higher max_iter so we can see where/if it actually converges) and return
    its per-iteration training loss."""
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    activation="relu",
                    solver="adam",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    return clf.named_steps["mlp"].loss_curve_


def plot_mlp_loss_curves(
    pool: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    seed: int = 2024,
    max_iter: int = 2000,
    production_max_iter: int = 500,
) -> None:
    keys = list(pool.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        X, y_raw = pool[key]
        loss_curve = fit_mlp_loss_curve(X, y_raw, seed=seed, max_iter=max_iter)
        ax.plot(range(1, len(loss_curve) + 1), loss_curve)
        ax.axvline(production_max_iter, color="red", linestyle="--", linewidth=1, label=f"production max_iter={production_max_iter}")
        ax.set_title(f"{POOL_TITLES.get(key, key)} (n={len(y_raw)})")
        ax.set_xlabel("iteration")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("training loss")
    fig.suptitle("MLP training loss — is it still learning when production stops?")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_learning_curves(
    pool: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    seed: int = 2024,
    cv: int = 5,
    train_sizes: np.ndarray | None = None,
    scoring: str = "balanced_accuracy",
    y_label: str = "balanced accuracy",
    figure_title: str = "Learning curves — does more training data help, or has it plateaued?",
) -> None:
    train_sizes = train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 8)
    input_keys = list(pool.keys())
    classifiers = sorted(_CLASSIFIER_BUILDERS)

    fig, axes = plt.subplots(len(input_keys), len(classifiers), figsize=(4.2 * len(classifiers), 4 * len(input_keys)), sharey=True)
    for row, input_key in enumerate(input_keys):
        X, y_raw = pool[input_key]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        for col, clf_name in enumerate(classifiers):
            ax = axes[row][col]
            build_estimator, _ = _CLASSIFIER_BUILDERS[clf_name]
            estimator = Pipeline(
                [("scaler", StandardScaler()), (clf_name, build_estimator(seed))]
            )
            sizes, train_scores, val_scores = learning_curve(
                estimator,
                X,
                y,
                train_sizes=train_sizes,
                cv=cv,
                scoring=scoring,
                shuffle=True,
                random_state=seed,
            )
            ax.plot(sizes, train_scores.mean(axis=1), label="train", marker="o")
            ax.plot(sizes, val_scores.mean(axis=1), label="cv (held-out fold)", marker="o")
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
            ax.set_title(f"{POOL_TITLES.get(input_key, input_key)}\n{clf_name}")
            ax.set_xlabel("training set size")
            if col == 0:
                ax.set_ylabel(y_label)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(figure_title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_diagnostics(
    config_file: Path,
    checkpoint: Path,
    model_name: str,
    users_csv: Path,
    run_dir: Path,
    test_lists_path: Path,
    train_pool_lists_path: Path,
    out_dir: Path,
    *,
    poolings: list[str] | None = None,
    plots: set[str] | None = None,
    seed: int = 2024,
    device: str | None = None,
) -> None:
    pool = _load_training_pool(
        config_file,
        checkpoint,
        model_name,
        users_csv,
        run_dir,
        test_lists_path,
        train_pool_lists_path,
        device,
        poolings=poolings,
    )
    out_dir = Path(out_dir)
    plots = plots or {"loss", "bacc", "auc"}
    if "loss" in plots:
        plot_mlp_loss_curves(pool, out_dir / "mlp_loss_curve.png", seed=seed)
    if "bacc" in plots:
        plot_learning_curves(pool, out_dir / "learning_curves.png", seed=seed)
    if "auc" in plots:
        plot_learning_curves(
            pool,
            out_dir / "learning_curves_auc.png",
            seed=seed,
            scoring="roc_auc",
            y_label="ROC AUC",
            figure_title=(
                "AUC learning curves — does ranking signal improve with more training data?"
            ),
        )


def main() -> None:
    p = argparse.ArgumentParser(description="MLP loss curves + learning curves for the embedding attackers")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True, help="RecBole model class name, e.g. BPR, LightGCN")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--users-csv", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--test-lists", type=Path, required=True)
    p.add_argument("--train-pool-lists", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--pooling",
        choices=["mean", "rank", "both"],
        default="both",
        help="which list-derived pooling(s) to include alongside the full embedding",
    )
    p.add_argument(
        "--plot",
        choices=["all", "loss", "bacc", "auc"],
        default="all",
        help="generate all diagnostics or only one plot (use 'auc' to avoid rerunning existing plots)",
    )
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    poolings = ["mean", "rank"] if args.pooling == "both" else [args.pooling]
    plots = {"loss", "bacc", "auc"} if args.plot == "all" else {args.plot}
    run_diagnostics(
        args.config,
        args.checkpoint,
        args.model,
        args.users_csv,
        args.run_dir,
        args.test_lists,
        args.train_pool_lists,
        args.out_dir,
        poolings=poolings,
        plots=plots,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
