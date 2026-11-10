"""Avaliação final no teste oficial do MNIST, uma única vez.

Pega as configurações finalistas (o melhor learning rate de cada otimizador no
batch de referência, segundo o estudo), treina com o mesmo orçamento do estudo
e mede a acurácia no conjunto de teste. O teste só é tocado aqui, no fim, para
não influenciar nenhuma decisão.

Para cada otimizador, treina com as seeds do estudo e reporta média e desvio da
acurácia de teste. Grava o resultado em research/test_results.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data import prepare
from src.network import MLP
from src.optim import SGD, Adam, Optimizer
from src.train import accuracy

TEST_RESULTS_PATH = Path("research/test_results.csv")

# Configurações finalistas: o melhor LR de cada otimizador no batch de
# referência (128), lido da tabela do estudo (research/summary.md).
FINALISTS = {
    "sgd": {"learning_rate": 0.4, "batch_size": 128},
    "momentum": {"learning_rate": 0.05, "batch_size": 128, "momentum": 0.9},
    "adam": {"learning_rate": 0.001, "batch_size": 128},
}


def make_optimizer(name: str, settings: dict[str, Any]) -> Optimizer:
    """Instancia o otimizador finalista a partir do nome e dos ajustes."""
    lr = float(settings["learning_rate"])
    if name == "sgd":
        return SGD(learning_rate=lr)
    if name == "momentum":
        return SGD(learning_rate=lr, momentum=float(settings["momentum"]))
    if name == "adam":
        return Adam(learning_rate=lr)
    raise ValueError(f"otimizador desconhecido: {name}")


def train_and_test(
    name: str,
    settings: dict[str, Any],
    data: dict[str, np.ndarray],
    layer_sizes: list[int],
    *,
    seed: int,
    epochs: int,
) -> float:
    """Treina um modelo e devolve a acurácia no teste (tocado uma vez)."""
    x_train, y_train = data["x_train"], data["y_train"]
    batch_size = int(settings["batch_size"])

    model = MLP(layer_sizes, seed=seed)
    pairs = model.parameters()
    params = [param for param, _ in pairs]
    grads = [grad for _, grad in pairs]
    optimizer = make_optimizer(name, settings)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        order = rng.permutation(x_train.shape[0])
        x_shuffled, y_shuffled = x_train[order], y_train[order]
        for begin in range(0, x_shuffled.shape[0], batch_size):
            stop = begin + batch_size
            model.forward(x_shuffled[begin:stop], y_shuffled[begin:stop])
            model.backward()
            optimizer.step(params, grads)

    return accuracy(model, data["x_test"], data["y_test"])


def main() -> None:
    config = load_config("configs/research.yaml")
    seeds = list(config["research"]["seeds"])
    epochs = int(config["research"]["epochs"])
    subset = int(config["data"]["train_subset"])
    hidden = list(config["model"]["hidden_sizes"])
    val_size = int(config["data"]["val_size"])

    data_full = prepare(val_size=val_size, seed=int(config["seed_base"]), include_test=True)
    data = {
        "x_train": data_full["x_train"][:subset],
        "y_train": data_full["y_train"][:subset],
        "x_test": data_full["x_test"],
        "y_test": data_full["y_test"],
    }
    layer_sizes = [data["x_train"].shape[1], *hidden, int(data["y_train"].max()) + 1]

    TEST_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, settings in FINALISTS.items():
        test_accs = [
            train_and_test(name, settings, data, layer_sizes, seed=int(seed), epochs=epochs)
            for seed in seeds
        ]
        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs))
        rows.append(
            {
                "optimizer": name,
                "learning_rate": settings["learning_rate"],
                "batch_size": settings["batch_size"],
                "test_acc_mean": round(mean_acc, 6),
                "test_acc_std": round(std_acc, 6),
                "seeds": len(seeds),
            }
        )
        print(f"{name:8s} teste: {mean_acc:.4f} ± {std_acc:.4f} (n={len(seeds)} seeds)")

    with TEST_RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresultados de teste em {TEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()
