"""Train OneTrans-Lite on deterministic synthetic request data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from model import OneTransLite, OneTransLiteBatch


DEMO_DIR = Path(__file__).resolve().parent


class SyntheticRankingDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Small deterministic request data with learnable multi-task labels."""

    def __init__(
        self,
        *,
        size: int,
        list_len: int,
        history_len: int,
        query_dim: int,
        candidate_dim: int,
        history_dim: int,
        num_tasks: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.query = torch.randn(size, query_dim, generator=generator)
        self.candidates = torch.randn(
            size,
            list_len,
            candidate_dim,
            generator=generator,
        )
        self.history = torch.randn(
            size,
            history_len,
            history_dim,
            generator=generator,
        )
        lengths = torch.arange(size) % (history_len + 1)
        positions = torch.arange(history_len).unsqueeze(0)
        self.history_valid = positions < lengths.unsqueeze(1)
        self.history = self.history * self.history_valid.unsqueeze(-1)

        pooled_history = self.history.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        base_score = (
            self.candidates[..., 0]
            + 0.5 * self.candidates[..., 1]
            + 0.4 * self.query[:, None, 0]
            + 0.3 * pooled_history[:, None, 0]
        )
        task_offsets = torch.linspace(-0.4, 0.4, num_tasks)
        probabilities = torch.sigmoid(base_score.unsqueeze(-1) + task_offsets)
        self.labels = torch.bernoulli(probabilities, generator=generator)

    def __len__(self) -> int:
        return self.query.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.query[index],
            self.candidates[index],
            self.history[index],
            self.history_valid[index],
            self.labels[index],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEMO_DIR / "model.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEMO_DIR / "training.yaml",
    )
    return parser.parse_args()


def _read_section(path: Path, section: str) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw[section])


def main() -> None:
    args = _parse_args()
    model_config = _read_section(args.model_config, "model")
    training_config = _read_section(args.training_config, "training")
    data_config = dict(training_config["synthetic_data"])
    optimizer_config = dict(training_config["optimizer"])

    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    torch.set_num_threads(int(training_config["num_threads"]))
    dataset = SyntheticRankingDataset(
        size=int(data_config["size"]),
        list_len=int(data_config["list_len"]),
        history_len=int(data_config["history_len"]),
        query_dim=int(model_config["query_dim"]),
        candidate_dim=int(model_config["candidate_dim"]),
        history_dim=int(model_config["history_dim"]),
        num_tasks=int(model_config["num_tasks"]),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
    )
    model = OneTransLite(**model_config)
    optimizer_type = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[str(optimizer_config["name"])]
    optimizer = optimizer_type(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )

    losses: list[float] = []
    logits = torch.empty(0)
    labels = torch.empty(0)
    for step, tensors in enumerate(loader):
        query, candidates, history, history_valid, labels = tensors
        batch = OneTransLiteBatch(
            query=query,
            candidates=candidates,
            history=history,
            history_valid=history_valid,
            labels=labels,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = model.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step + 1 >= int(training_config["steps"]):
            break

    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    print(
        "onetrans_lite_torch_ok",
        f"optimizer={optimizer.__class__.__name__}",
        f"steps={len(losses)}",
        f"logits_shape={tuple(logits.shape)}",
        f"labels_shape={tuple(labels.shape)}",
        f"last_loss={losses[-1]:.6f}",
        f"gradient_norm={gradient_norm:.6f}",
    )


if __name__ == "__main__":
    main()
