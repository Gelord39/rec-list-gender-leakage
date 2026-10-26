"""Train a RecBole model and export user embeddings + metrics.

Example:
  uv run python -m src.train.train_recbole \\
    --model BPR --config configs/bpr_ml1m.yaml --exp-id EXP-001_bpr_d32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed

# RecBole checkpoints are local/trusted; PyTorch>=2.6 defaults weights_only=True.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

# scipy>=1.14 dropped dok_matrix._update (private alias RecBole's GCN models
# call internally, e.g. LightGCN.get_norm_adj_mat); public .update() is
# functionally identical, so alias it back.
import scipy.sparse as _sp

if not hasattr(_sp.dok_matrix, "_update"):
    _sp.dok_matrix._update = _sp.dok_matrix.update


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # RecBole is more reliable on CPU for some ops; try MPS first, caller can override.
        return "mps"
    return "cpu"


def get_user_item_embeddings(model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """Return (user_weight, item_weight) full tables, incl. the padding row 0.

    For GCN-family models (LightGCN, NGCF, NCL — detected via ``n_layers`` plus
    a no-arg ``forward()`` returning ``(user_all_embeddings, item_all_embeddings)``),
    ``model.user_embedding.weight`` is only the layer-0 input embedding *before*
    graph propagation, not what the model actually scores with. We call
    ``forward()`` to get the final propagated embedding instead (also giving us
    the equivalent propagated item embedding for free). BPR (and any other
    flat-embedding model) has no ``n_layers`` and falls back to the raw
    embedding tables. SGL is not covered by this heuristic (its ``forward``
    requires a graph argument) — needs its own case if/when it's added.
    """
    if hasattr(model, "n_layers"):
        model.eval()
        with torch.no_grad():
            user_all_embeddings, item_all_embeddings = model.forward()
        return (
            user_all_embeddings.detach().cpu().numpy(),
            item_all_embeddings.detach().cpu().numpy(),
        )
    if hasattr(model, "user_embedding") and hasattr(model, "item_embedding"):
        return (
            model.user_embedding.weight.detach().cpu().numpy(),
            model.item_embedding.weight.detach().cpu().numpy(),
        )
    raise AttributeError(
        f"{type(model).__name__} has no user_embedding/item_embedding and "
        "no n_layers; cannot export for AIA."
    )


def _export_user_embeddings(model: torch.nn.Module, dataset, out_dir: Path) -> dict[str, str]:
    """Save original user/item id tokens and their embedding rows (skip padding id 0).

    Item embeddings are exported alongside user embeddings so list-derived
    attacks (mean-pool the recommended items' own embeddings) can reuse the
    same trained representation instead of a separate encoder.
    """
    user_weight, item_weight = get_user_item_embeddings(model)
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    # RecBole internal ids: 0 = [PAD]; 1..num-1 are real users/items
    user_internal_ids = list(range(1, dataset.user_num))
    item_internal_ids = list(range(1, dataset.item_num))
    user_ids = np.array(
        [dataset.id2token(uid_field, i) for i in user_internal_ids], dtype=str
    )
    item_ids = np.array(
        [dataset.id2token(iid_field, i) for i in item_internal_ids], dtype=str
    )
    user_embeddings = user_weight[user_internal_ids]
    item_embeddings = item_weight[item_internal_ids]

    emb_path = out_dir / "user_embeddings.npy"
    ids_path = out_dir / "user_ids.npy"
    item_emb_path = out_dir / "item_embeddings.npy"
    item_ids_path = out_dir / "item_ids.npy"
    np.save(emb_path, user_embeddings)
    np.save(ids_path, user_ids)
    np.save(item_emb_path, item_embeddings)
    np.save(item_ids_path, item_ids)
    return {
        "user_embeddings": str(emb_path),
        "user_ids": str(ids_path),
        "item_embeddings": str(item_emb_path),
        "item_ids": str(item_ids_path),
        "embedding_dim": int(user_embeddings.shape[1]),
        "n_users": int(user_embeddings.shape[0]),
        "n_items": int(item_embeddings.shape[0]),
    }


def train_and_export(
    model_name: str,
    config_file: Path,
    exp_id: str,
    runs_root: Path,
    device: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_file = config_file if config_file.is_absolute() else root / config_file

    config = Config(
        model=model_name,
        config_file_list=[str(config_file)],
        config_dict=overrides or {},
    )

    # Make data_path absolute so cwd does not matter
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        config["data_path"] = str((root / data_path).resolve())

    resolved_device = _resolve_device(device)
    # RecBole + recent torch expect a torch.device (uses device.type in autocast)
    torch_device = torch.device(resolved_device)
    config["device"] = torch_device

    out_dir = runs_root if runs_root.is_absolute() else root / runs_root
    out_dir = out_dir / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(out_dir / "checkpoint")

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    logger.info("Device: %s | exp_id: %s", torch_device, exp_id)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    def _build_trainer(dev: torch.device):
        config["device"] = dev
        m = get_model(config["model"])(config, train_data.dataset).to(dev)
        t = get_trainer(config["MODEL_TYPE"], config["model"])(config, m)
        return m, t

    model, trainer = _build_trainer(torch_device)
    logger.info(model)

    try:
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=True
        )
    except (NotImplementedError, RuntimeError, AttributeError) as exc:
        if resolved_device == "mps":
            logger.warning("MPS failed (%s); retrying on CPU.", exc)
            resolved_device = "cpu"
            torch_device = torch.device("cpu")
            init_seed(config["seed"], config["reproducibility"])
            model, trainer = _build_trainer(torch_device)
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
        else:
            raise

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    emb_info = _export_user_embeddings(model, train_data.dataset, out_dir)

    metrics = {
        "exp_id": exp_id,
        "model": model_name,
        "config_file": str(config_file),
        "device": str(torch_device),
        "seed": config["seed"],
        "embedding_size": config["embedding_size"],
        "best_valid_score": float(best_valid_score)
        if not isinstance(best_valid_score, dict)
        else best_valid_score,
        "best_valid_result": dict(best_valid_result)
        if best_valid_result is not None
        else None,
        "test_result": dict(test_result) if test_result is not None else None,
        "embeddings": emb_info,
    }
    metrics_path = out_dir / "rec_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", metrics_path)
    logger.info("Test result: %s", test_result)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train RecBole model and export embeddings")
    parser.add_argument("--model", default="BPR")
    parser.add_argument("--config", type=Path, default=Path("configs/bpr_ml1m.yaml"))
    parser.add_argument("--exp-id", default="EXP-001_bpr_d32")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | mps | cpu (default: auto)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    metrics = train_and_export(
        model_name=args.model,
        config_file=args.config,
        exp_id=args.exp_id,
        runs_root=args.runs_root,
        device=args.device,
        overrides=overrides,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
