"""Loop de treino da MLP no MNIST.

Para cada época:
1. embaralha o treino e passa em mini-batches (forward, backward, passo do SGD);
2. mede loss e acurácia em treino e validação;
3. grava uma linha no CSV;
4. guarda o melhor modelo pela validação (checkpoint) e aplica early stopping.

Tudo é controlado por ``configs/default.yaml`` e por uma seed fixa, então a
mesma config dá o mesmo resultado.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data import prepare
from src.network import MLP
from src.optim import SGD

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.csv"
CHECKPOINT_PATH = REPORTS_DIR / "best_model.npz"


def accuracy(model: MLP, inputs: np.ndarray, labels: np.ndarray) -> float:
    """Fração de acertos do modelo em ``inputs`` contra ``labels``."""
    return float(np.mean(model.predict(inputs) == labels))


def mean_loss(model: MLP, inputs: np.ndarray, labels: np.ndarray, batch_size: int) -> float:
    """Loss média em batches, para não carregar tudo de uma vez."""
    total = 0.0
    n = inputs.shape[0]
    for start in range(0, n, batch_size):
        stop = start + batch_size
        batch_x = inputs[start:stop]
        batch_y = labels[start:stop]
        total += model.forward(batch_x, batch_y) * len(batch_x)
    return total / n


def save_checkpoint(model: MLP, path: Path = CHECKPOINT_PATH) -> None:
    """Salva os pesos e vieses do modelo num .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for index, linear in enumerate(model.linears):
        arrays[f"weights_{index}"] = linear.weights
        arrays[f"bias_{index}"] = linear.bias
    np.savez(path, **arrays)


def train(config: dict[str, Any]) -> dict[str, float]:
    """Treina a MLP conforme a config e retorna as métricas finais.

    Returns:
        Dicionário com a melhor acurácia de validação e a época em que ocorreu.
    """
    seed = int(config["seed"])
    train_config = config["train"]
    hidden = list(train_config["hidden_sizes"])
    learning_rate = float(train_config["learning_rate"])
    batch_size = int(train_config["batch_size"])
    epochs = int(train_config["epochs"])
    patience = int(train_config.get("patience", epochs))  # early stopping
    val_size = int(config["data"]["val_size"])

    data = prepare(val_size=val_size, seed=seed)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    num_features = x_train.shape[1]
    num_classes = int(y_train.max()) + 1
    layer_sizes = [num_features, *hidden, num_classes]

    model = MLP(layer_sizes, seed=seed)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    rng = np.random.default_rng(seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, epochs + 1):
            # Embaralha o treino a cada época.
            order = rng.permutation(x_train.shape[0])
            x_shuffled, y_shuffled = x_train[order], y_train[order]

            for start in range(0, x_shuffled.shape[0], batch_size):
                stop = start + batch_size
                model.forward(x_shuffled[start:stop], y_shuffled[start:stop])
                model.backward()
                optimizer.step()

            train_loss = mean_loss(model, x_train, y_train, batch_size)
            train_acc = accuracy(model, x_train, y_train)
            val_loss = mean_loss(model, x_val, y_val, batch_size)
            val_acc = accuracy(model, x_val, y_val)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                ]
            )
            handle.flush()
            print(
                f"época {epoch:2d} | treino loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

            # Checkpoint do melhor modelo + early stopping, ambos pela validação.
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(model)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"early stopping na época {epoch} (sem melhora há {patience})")
                    break

    print(
        f"\nmelhor val acc = {best_val_accuracy:.4f} (época {best_epoch}); "
        f"métricas em {METRICS_PATH}, melhor modelo em {CHECKPOINT_PATH}"
    )
    return {"best_val_accuracy": best_val_accuracy, "best_epoch": float(best_epoch)}


def main() -> None:
    train(load_config())


if __name__ == "__main__":
    main()
