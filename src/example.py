"""Funções de exemplo, só para o template ter algo testável.

Apague este arquivo quando começar o projeto de verdade e coloque
o código real aqui em src/.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fração de previsões corretas.

    Args:
        y_true: rótulos verdadeiros, shape (n,).
        y_pred: rótulos previstos, shape (n,).

    Returns:
        Acurácia entre 0 e 1.

    Raises:
        ValueError: se os tamanhos não baterem ou os arrays forem vazios.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shapes diferentes: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("arrays vazios")
    return float(np.mean(y_true == y_pred))


def train_val_split(
    n: int, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Divide os índices 0..n-1 em treino e validação, de forma reprodutível.

    Args:
        n: número de exemplos.
        val_fraction: fração que vai para validação (entre 0 e 1).
        seed: semente do gerador aleatório, para o split ser sempre o mesmo.

    Returns:
        (índices de treino, índices de validação).
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction precisa estar entre 0 e 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return indices[n_val:], indices[:n_val]
