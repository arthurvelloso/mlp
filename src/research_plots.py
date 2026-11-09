"""Gera os gráficos e a tabela-resumo do estudo a partir do CSV.

Figuras (por script, não à mão):
1. Acurácia de validação vs. learning rate, por otimizador e batch, com média
   entre seeds em traço grosso e seeds individuais em traço fino.
2. Heatmap LR × batch da acurácia média, um painel por otimizador.
3. LR ótimo × batch em escala log-log, com a inclinação ajustada (testa H1).

Tabela-resumo: para cada (otimizador, batch), o LR ótimo e a acurácia; e a
largura da faixa de LR que atinge o alvo (testa H2).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_PATH = Path("research/results.csv")
FIGURES_DIR = Path("research/figures")
SUMMARY_PATH = Path("research/summary.md")
TARGET_ACCURACY = 0.97
OPTIMIZERS = ["sgd", "momentum", "adam"]


def load_results(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Lê o CSV convertendo os campos numéricos."""
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode `python -m src.research_run`.")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "optimizer": row["optimizer"],
                    "learning_rate": float(row["learning_rate"]),
                    "batch_size": int(row["batch_size"]),
                    "seed": int(row["seed"]),
                    "final_val_acc": float(row["final_val_acc"]),
                    "diverged": int(row["diverged"]),
                }
            )
    return rows


def mean_over_seeds(rows: list[dict[str, Any]]) -> dict[tuple[str, int, float], dict[str, float]]:
    """Agrega por (otimizador, batch, lr): média e desvio da acurácia."""
    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["optimizer"], row["batch_size"], row["learning_rate"])
        grouped[key].append(row["final_val_acc"])
    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key, values in grouped.items()
    }


def _sorted_values(rows: list[dict[str, Any]], field: str) -> list:
    return sorted({row[field] for row in rows})


def plot_acc_vs_lr(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Acurácia vs. LR, média grossa e seeds finas, um painel por otimizador."""
    batches = _sorted_values(rows, "batch_size")
    agg = mean_over_seeds(rows)
    fig, axes = plt.subplots(1, len(OPTIMIZERS), figsize=(5 * len(OPTIMIZERS), 4.2), squeeze=False)

    for col, optimizer in enumerate(OPTIMIZERS):
        ax = axes[0][col]
        for batch in batches:
            lrs = sorted({row["learning_rate"] for row in rows if row["optimizer"] == optimizer})
            means = [agg[(optimizer, batch, lr)]["mean"] for lr in lrs]
            line = ax.plot(lrs, means, marker="o", linewidth=2.2, label=f"batch {batch}")[0]
            # Seeds individuais em traço fino, na mesma cor.
            for seed in _sorted_values(rows, "seed"):
                seed_acc = [
                    next(
                        r["final_val_acc"]
                        for r in rows
                        if r["optimizer"] == optimizer
                        and r["batch_size"] == batch
                        and r["learning_rate"] == lr
                        and r["seed"] == seed
                    )
                    for lr in lrs
                ]
                ax.plot(lrs, seed_acc, linewidth=0.5, alpha=0.4, color=line.get_color())
        ax.set_xscale("log", base=2)
        ax.set_xlabel("learning rate (log2)")
        ax.set_ylabel("acurácia de validação")
        ax.set_title(optimizer)
        ax.axhline(TARGET_ACCURACY, color="gray", linestyle="--", linewidth=0.8)
        ax.legend()

    fig.tight_layout()
    path = out_dir / "acc_vs_lr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmaps(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    """Heatmap LR × batch da acurácia média, um painel por otimizador."""
    batches = _sorted_values(rows, "batch_size")
    agg = mean_over_seeds(rows)
    fig, axes = plt.subplots(1, len(OPTIMIZERS), figsize=(5 * len(OPTIMIZERS), 4.2), squeeze=False)

    for col, optimizer in enumerate(OPTIMIZERS):
        ax = axes[0][col]
        lrs = sorted({row["learning_rate"] for row in rows if row["optimizer"] == optimizer})
        matrix = np.array(
            [[agg[(optimizer, batch, lr)]["mean"] for lr in lrs] for batch in batches]
        )
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{lr:g}" for lr in lrs], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(batches)))
        ax.set_yticklabels(batches)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("batch size")
        ax.set_title(optimizer)
        fig.colorbar(image, ax=ax, fraction=0.046)

    fig.tight_layout()
    path = out_dir / "heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def best_lr_per_batch(
    rows: list[dict[str, Any]],
) -> dict[str, list[tuple[int, float, float]]]:
    """Para cada otimizador, o LR de maior acurácia média em cada batch.

    Returns:
        ``{otimizador: [(batch, lr_otimo, acc), ...]}`` ordenado por batch.
    """
    batches = _sorted_values(rows, "batch_size")
    agg = mean_over_seeds(rows)
    best: dict[str, list[tuple[int, float, float]]] = {}
    for optimizer in OPTIMIZERS:
        lrs = sorted({row["learning_rate"] for row in rows if row["optimizer"] == optimizer})
        entries: list[tuple[int, float, float]] = []
        for batch in batches:
            lr_star = max(lrs, key=lambda lr: agg[(optimizer, batch, lr)]["mean"])
            entries.append((batch, lr_star, agg[(optimizer, batch, lr_star)]["mean"]))
        best[optimizer] = entries
    return best


def fit_slope(batches: list[int], lrs: list[float]) -> float:
    """Inclinação de log(LR ótimo) vs. log(batch) por mínimos quadrados."""
    log_b = np.log2(np.array(batches, dtype=float))
    log_lr = np.log2(np.array(lrs, dtype=float))
    slope, _ = np.polyfit(log_b, log_lr, 1)
    return float(slope)


def plot_optimal_lr(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, dict[str, float]]:
    """LR ótimo × batch em log-log, com a inclinação ajustada por otimizador."""
    best = best_lr_per_batch(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    slopes: dict[str, float] = {}

    for optimizer in OPTIMIZERS:
        batches = [entry[0] for entry in best[optimizer]]
        lrs = [entry[1] for entry in best[optimizer]]
        slope = fit_slope(batches, lrs)
        slopes[optimizer] = slope
        ax.plot(batches, lrs, marker="o", label=f"{optimizer} (inclinação {slope:.2f})")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("batch size (log2)")
    ax.set_ylabel("learning rate ótimo (log2)")
    ax.set_title("LR ótimo × batch size")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "optimal_lr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, slopes


def lr_range_at_target(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Quantas (LR, batch) médias atingem o alvo fixo, por otimizador."""
    agg = mean_over_seeds(rows)
    counts: dict[str, int] = dict.fromkeys(OPTIMIZERS, 0)
    for (optimizer, _batch, _lr), stats in agg.items():
        if stats["mean"] >= TARGET_ACCURACY:
            counts[optimizer] += 1
    return counts


def lr_range_near_best(rows: list[dict[str, Any]], margin: float = 0.01) -> dict[str, int]:
    """Quantos LRs ficam a ``margin`` do melhor do próprio otimizador (testa H2).

    Mede a "largura da faixa boa" de forma relativa, sem depender de um alvo
    absoluto que o orçamento pode não alcançar. Conta, por otimizador, as
    combinações (LR, batch) cuja acurácia média fica dentro de ``margin`` do
    melhor resultado daquele otimizador.
    """
    agg = mean_over_seeds(rows)
    best_by_opt: dict[str, float] = dict.fromkeys(OPTIMIZERS, 0.0)
    for (optimizer, _batch, _lr), stats in agg.items():
        best_by_opt[optimizer] = max(best_by_opt[optimizer], stats["mean"])

    counts: dict[str, int] = dict.fromkeys(OPTIMIZERS, 0)
    for (optimizer, _batch, _lr), stats in agg.items():
        if stats["mean"] >= best_by_opt[optimizer] - margin:
            counts[optimizer] += 1
    return counts


def write_summary(
    rows: list[dict[str, Any]], slopes: dict[str, float], path: Path = SUMMARY_PATH
) -> None:
    """Escreve a tabela-resumo em Markdown."""
    best = best_lr_per_batch(rows)
    target_counts = lr_range_at_target(rows)
    near_best_counts = lr_range_near_best(rows)
    diverged = sum(row["diverged"] for row in rows)

    lines = [
        "# Resumo do estudo",
        "",
        f"- Total de runs: {len(rows)}",
        f"- Runs que divergiram: {diverged}",
        "",
        "## LR ótimo por (otimizador, batch)",
        "",
        "| Otimizador | Batch | LR ótimo | Acurácia média |",
        "| --- | --- | --- | --- |",
    ]
    for optimizer in OPTIMIZERS:
        for batch, lr_star, acc in best[optimizer]:
            lines.append(f"| {optimizer} | {batch} | {lr_star:g} | {acc:.4f} |")

    lines += [
        "",
        "## H1: inclinação de log(LR ótimo) vs. log(batch)",
        "",
        "Esperado: ~1 para SGD/momentum (regra linear), ~0,5 para Adam (raiz).",
        "",
        "| Otimizador | Inclinação ajustada |",
        "| --- | --- |",
    ]
    for optimizer in OPTIMIZERS:
        lines.append(f"| {optimizer} | {slopes[optimizer]:.2f} |")

    lines += [
        "",
        "## H2: largura da faixa de bons learning rates",
        "",
        f"Alvo absoluto de {TARGET_ACCURACY:.0%}: número de combinações (LR, batch)",
        "cuja acurácia média o atinge. Neste orçamento curto, pode ser 0 para",
        "todos (o melhor resultado fica abaixo do alvo).",
        "",
        "| Otimizador | Combinações no alvo (de 24) |",
        "| --- | --- |",
    ]
    for optimizer in OPTIMIZERS:
        lines.append(f"| {optimizer} | {target_counts[optimizer]} |")

    lines += [
        "",
        "Medida relativa (independe do alvo): combinações (LR, batch) a no máximo",
        "1 ponto percentual do melhor resultado do próprio otimizador. Faixa mais",
        "larga = mais tolerante à escolha do learning rate.",
        "",
        "| Otimizador | Combinações perto do melhor (de 24) |",
        "| --- | --- |",
    ]
    for optimizer in OPTIMIZERS:
        lines.append(f"| {optimizer} | {near_best_counts[optimizer]} |")

    lines += [
        "",
        "> Conclusões escritas no diário. Qualquer resultado é válido, inclusive",
        "> 'a regra não se confirmou neste setup'.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_acc_vs_lr(rows, FIGURES_DIR)
    p2 = plot_heatmaps(rows, FIGURES_DIR)
    p3, slopes = plot_optimal_lr(rows, FIGURES_DIR)
    write_summary(rows, slopes)
    for path in (p1, p2, p3, SUMMARY_PATH):
        print(f"gerado: {path}")


if __name__ == "__main__":
    main()
