"""Ponto de entrada do `make reproduce`.

No projeto real, este arquivo carrega os dados, treina, avalia e imprime
(ou salva) o resultado principal. Aqui ele só mostra o esqueleto.
"""

from __future__ import annotations

import numpy as np

from src.example import accuracy, train_val_split


def main(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=100)
    y_pred = y_true.copy()
    y_pred[:10] = (y_pred[:10] + 1) % 3  # erra 10 de propósito

    train_idx, val_idx = train_val_split(len(y_true), val_fraction=0.2, seed=seed)
    print(f"treino: {len(train_idx)} exemplos | validação: {len(val_idx)} exemplos")
    print(f"acurácia (exemplo): {accuracy(y_true, y_pred):.3f}")


if __name__ == "__main__":
    main()
