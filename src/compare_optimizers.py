"""Compara os quatro otimizadores na mesma rede, com a mesma inicialização.

Treina uma MLP no MNIST uma vez por otimizador, tudo com a mesma seed (mesma
inicialização de pesos e mesma ordem de batches) e o mesmo número de épocas.
No fim, plota as curvas de loss e de acurácia de validação no mesmo gráfico.

Não usa early stopping: o objetivo é comparar as trajetórias no mesmo horizonte.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from src.config import load_config
from src.data import prepare
from src.network import MLP
from src.optim import SGD, Adam, AdamW, Optimizer
from src.train import accuracy, mean_loss

matplotlib.use("Agg")  # backend sem janela; precisa vir antes do pyplot
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = Path("reports/figures")


def build_optimizers() -> dict[str, Optimizer]:
    """Os quatro otimizadores, cada um com um learning rate razoável.

    Learning rates diferentes de propósito: SGD precisa de passo maior que os
    adaptativos. A comparação é entre trajetórias bem ajustadas, não no mesmo lr.
    """
    return {
        "SGD": SGD(learning_rate=0.1),
        "SGD+momentum": SGD(learning_rate=0.1, momentum=0.9),
        "Adam": Adam(learning_rate=0.001),
        "AdamW": AdamW(learning_rate=0.001, weight_decay=0.01),
    }


def train_one(
    name: str,
    optimizer: Optimizer,
    data: dict[str, np.ndarray],
    layer_sizes: list[int],
    *,
    seed: int,
    batch_size: int,
    epochs: int,
) -> dict[str, list[float]]:
    """Treina uma MLP com um otimizador e devolve as curvas por época."""
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    # Mesma seed => mesma inicialização de pesos e mesma ordem de batches.
    model = MLP(layer_sizes, seed=seed)
    pairs = model.parameters()
    params = [param for param, _ in pairs]
    grads = [grad for _, grad in pairs]
    rng = np.random.default_rng(seed)

    history: dict[str, list[float]] = {"val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(x_train.shape[0])
        x_shuffled, y_shuffled = x_train[order], y_train[order]
        for start in range(0, x_shuffled.shape[0], batch_size):
            stop = start + batch_size
            model.forward(x_shuffled[start:stop], y_shuffled[start:stop])
            model.backward()
            optimizer.step(params, grads)

        val_loss = mean_loss(model, x_val, y_val, batch_size)
        val_acc = accuracy(model, x_val, y_val)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{name}] época {epoch:2d} | val loss {val_loss:.4f} acc {val_acc:.4f}")

    return history


def plot_comparison(
    histories: dict[str, dict[str, list[float]]], out_dir: Path = FIGURES_DIR
) -> Path:
    """Plota as curvas dos otimizadores lado a lado (loss e acurácia)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        ax_loss.plot(epochs, history["val_loss"], label=name)
        ax_acc.plot(epochs, history["val_acc"], label=name)

    ax_loss.set_xlabel("época")
    ax_loss.set_ylabel("loss de validação")
    ax_loss.set_title("Loss de validação por otimizador")
    ax_loss.legend()

    ax_acc.set_xlabel("época")
    ax_acc.set_ylabel("acurácia de validação")
    ax_acc.set_title("Acurácia de validação por otimizador")
    ax_acc.legend()

    fig.tight_layout()
    path = out_dir / "optimizers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    batch_size = int(config["train"]["batch_size"])
    epochs = int(config["train"]["epochs"])
    hidden = list(config["train"]["hidden_sizes"])
    val_size = int(config["data"]["val_size"])

    data = prepare(val_size=val_size, seed=seed)
    layer_sizes = [data["x_train"].shape[1], *hidden, int(data["y_train"].max()) + 1]

    histories = {
        name: train_one(
            name,
            optimizer,
            data,
            layer_sizes,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
        )
        for name, optimizer in build_optimizers().items()
    }

    path = plot_comparison(histories)
    print(f"\nfigura salva: {path}")


if __name__ == "__main__":
    main()
