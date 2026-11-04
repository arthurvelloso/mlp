"""Gera as curvas de loss e acurácia a partir do CSV de métricas.

O ponto é gerar os gráficos por script (``make plots``), e não à mão num
notebook: assim qualquer pessoa reproduz as mesmas figuras a partir do CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela, funciona no CI e no terminal
import matplotlib.pyplot as plt  # noqa: E402 (precisa vir depois do use)

METRICS_PATH = Path("reports/metrics.csv")
FIGURES_DIR = Path("reports/figures")


def read_metrics(path: Path = METRICS_PATH) -> dict[str, list[float]]:
    """Lê o CSV de métricas e devolve uma lista de valores por coluna."""
    if not path.exists():
        raise FileNotFoundError(f"métricas não encontradas: {path}. Rode `make train` antes.")
    columns: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for field in reader.fieldnames or []:
            columns[field] = []
        for row in reader:
            for field, value in row.items():
                columns[field].append(float(value))
    return columns


def plot_curves(metrics: dict[str, list[float]], out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Salva duas figuras: curva de loss e curva de acurácia."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics["epoch"]
    saved: list[Path] = []

    # Loss.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, metrics["train_loss"], label="treino")
    ax.plot(epochs, metrics["val_loss"], label="validação")
    ax.set_xlabel("época")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title("Curva de loss")
    ax.legend()
    fig.tight_layout()
    loss_path = out_dir / "loss.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    saved.append(loss_path)

    # Acurácia.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, metrics["train_acc"], label="treino")
    ax.plot(epochs, metrics["val_acc"], label="validação")
    ax.set_xlabel("época")
    ax.set_ylabel("acurácia")
    ax.set_title("Curva de acurácia")
    ax.legend()
    fig.tight_layout()
    acc_path = out_dir / "accuracy.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    saved.append(acc_path)

    return saved


def main() -> None:
    saved = plot_curves(read_metrics())
    for path in saved:
        print(f"figura salva: {path}")


if __name__ == "__main__":
    main()
