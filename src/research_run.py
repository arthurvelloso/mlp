"""Roda a grade do estudo e grava uma linha por run num CSV.

Grade: otimizadores × learning rates × batches × seeds (ver configs/research.yaml
e research/PREREGISTRO.md). Cada run treina uma MLP por um número fixo de épocas
e registra acurácia final, épocas até o alvo, divergência e tempo.

A inicialização da rede e a ordem dos batches dependem só da seed do run, então
os três otimizadores partem da mesma inicialização em cada seed.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data import prepare
from src.network import MLP
from src.optim import SGD, Adam, Optimizer
from src.train import accuracy

RESEARCH_DIR = Path("research")
RESULTS_PATH = RESEARCH_DIR / "results.csv"


def make_optimizer(name: str, learning_rate: float, config: dict[str, Any]) -> Optimizer:
    """Cria o otimizador do run a partir do nome e do learning rate."""
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    if name == "momentum":
        momentum = float(config["research"]["optimizers"]["momentum"]["momentum"])
        return SGD(learning_rate=learning_rate, momentum=momentum)
    if name == "adam":
        return Adam(learning_rate=learning_rate)
    raise ValueError(f"otimizador desconhecido: {name}")


def run_once(
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    seed: int,
    data: dict[str, np.ndarray],
    layer_sizes: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Executa um run e devolve suas métricas."""
    epochs = int(config["research"]["epochs"])
    target = float(config["research"]["target_accuracy"])
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    model = MLP(layer_sizes, seed=seed)
    pairs = model.parameters()
    params = [param for param, _ in pairs]
    grads = [grad for _, grad in pairs]
    optimizer = make_optimizer(optimizer_name, learning_rate, config)
    rng = np.random.default_rng(seed)

    epochs_to_target: int | str = ""
    diverged = False
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(x_train.shape[0])
        x_shuffled, y_shuffled = x_train[order], y_train[order]
        for begin in range(0, x_shuffled.shape[0], batch_size):
            stop = begin + batch_size
            model.forward(x_shuffled[begin:stop], y_shuffled[begin:stop])
            model.backward()
            optimizer.step(params, grads)

        # Divergência: algum parâmetro virou NaN/inf.
        if any(not np.all(np.isfinite(param)) for param in params):
            diverged = True
            break

        val_acc = accuracy(model, x_val, y_val)
        if epochs_to_target == "" and val_acc >= target:
            epochs_to_target = epoch

    elapsed = time.perf_counter() - start
    final_val_acc = 0.0 if diverged else accuracy(model, x_val, y_val)

    return {
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "final_val_acc": round(final_val_acc, 6),
        "epochs_to_target": epochs_to_target,
        "diverged": int(diverged),
        "seconds": round(elapsed, 3),
    }


FIELDNAMES = [
    "optimizer",
    "learning_rate",
    "batch_size",
    "seed",
    "final_val_acc",
    "epochs_to_target",
    "diverged",
    "seconds",
]


def run_grid(config: dict[str, Any], path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Roda a grade inteira, gravando cada run no CSV assim que termina.

    A gravação incremental evita perder tudo se a execução for interrompida.
    """
    research = config["research"]
    seeds = list(research["seeds"])
    batch_sizes = list(research["batch_sizes"])
    val_size = int(config["data"]["val_size"])
    subset = int(config["data"]["train_subset"])
    hidden = list(config["model"]["hidden_sizes"])

    # Dados preparados uma vez; o subconjunto de treino é o mesmo para todos.
    data_full = prepare(val_size=val_size, seed=int(config["seed_base"]))
    data = {
        "x_train": data_full["x_train"][:subset],
        "y_train": data_full["y_train"][:subset],
        "x_val": data_full["x_val"],
        "y_val": data_full["y_val"],
    }
    layer_sizes = [data["x_train"].shape[1], *hidden, int(data["y_train"].max()) + 1]

    optimizers = research["optimizers"]
    results: list[dict[str, Any]] = []
    total = (
        sum(len(optimizers[name]["learning_rates"]) for name in optimizers)
        * len(batch_sizes)
        * len(seeds)
    )
    done = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        handle.flush()

        for optimizer_name in optimizers:
            for learning_rate in optimizers[optimizer_name]["learning_rates"]:
                for batch_size in batch_sizes:
                    for seed in seeds:
                        result = run_once(
                            optimizer_name,
                            float(learning_rate),
                            int(batch_size),
                            int(seed),
                            data,
                            layer_sizes,
                            config,
                        )
                        results.append(result)
                        writer.writerow(result)
                        handle.flush()  # grava já, para não perder em interrupção
                        done += 1
                        print(
                            f"[{done:3d}/{total}] {optimizer_name:8s} "
                            f"lr={learning_rate:<8g} bs={batch_size:<4d} seed={seed} "
                            f"-> acc={result['final_val_acc']:.4f} "
                            f"div={result['diverged']} ({result['seconds']:.1f}s)",
                            flush=True,
                        )
    return results


def main() -> None:
    config = load_config("configs/research.yaml")
    results = run_grid(config)
    print(f"\n{len(results)} runs salvos em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
