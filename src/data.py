"""Carregamento e preparo do MNIST.

Fluxo:
1. Baixar o MNIST uma vez e guardar em ``data/mnist.npz`` (cache local).
2. Separar os 60 mil exemplos de treino em 50 mil de treino e 10 mil de
   validação, de forma estratificada (mesma proporção de classes) e com
   seed fixa.
3. Normalizar com a média e o desvio calculados **só no treino**, para não
   vazar informação da validação nem do teste.

Os 10 mil exemplos de teste oficiais ficam guardados e não são usados aqui;
eles só entram na avaliação final (dia 6).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

# Espelho do MNIST em .npz (mesmo formato usado pelo Keras).
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_DIR = Path("data")
MNIST_PATH = DATA_DIR / "mnist.npz"


def download_mnist(path: Path = MNIST_PATH) -> Path:
    """Baixa o MNIST uma vez e guarda em ``path``. Se já existe, não rebaixa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    print(f"baixando MNIST para {path} ...")
    urllib.request.urlretrieve(MNIST_URL, path)  # noqa: S310 (URL fixa e confiável)
    return path


def load_raw(path: Path = MNIST_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega o MNIST cru do .npz.

    Returns:
        ``(x_train, y_train, x_test, y_test)``. As imagens vêm como uint8 com
        shape ``(n, 28, 28)`` e os rótulos como inteiros ``(n,)``.
    """
    download_mnist(path)
    with np.load(path) as data:
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def stratified_split(
    labels: np.ndarray, val_size: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Divide índices em treino e validação mantendo a proporção das classes.

    Args:
        labels: rótulos de todos os exemplos, shape ``(n,)``.
        val_size: número total de exemplos que vão para validação.
        seed: semente para o embaralhamento, para o split ser reprodutível.

    Returns:
        ``(indices_treino, indices_validacao)``.
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    if not 0 < val_size < n:
        raise ValueError("val_size precisa estar entre 1 e n-1")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    val_fraction = val_size / n

    val_parts: list[np.ndarray] = []
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        n_val_cls = int(round(len(cls_indices) * val_fraction))
        val_parts.append(cls_indices[:n_val_cls])

    val_indices = np.concatenate(val_parts)
    mask = np.ones(n, dtype=bool)
    mask[val_indices] = False
    train_indices = np.where(mask)[0]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _flatten_and_scale(images: np.ndarray) -> np.ndarray:
    """Achata ``(n, 28, 28)`` em ``(n, 784)`` e coloca os pixels em [0, 1]."""
    return images.reshape(images.shape[0], -1).astype(np.float64) / 255.0


def prepare(
    val_size: int = 10_000, seed: int = 0, path: Path = MNIST_PATH
) -> dict[str, np.ndarray]:
    """Prepara treino e validação prontos para o modelo.

    As imagens são achatadas para 784, normalizadas com a média e o desvio do
    **treino**, e os rótulos ficam como inteiros.

    Returns:
        Dicionário com ``x_train``, ``y_train``, ``x_val``, ``y_val`` e também
        ``mean`` e ``std`` (guardados para normalizar o teste do mesmo jeito,
        no dia 6).
    """
    x_train_raw, y_train_raw, _, _ = load_raw(path)

    train_idx, val_idx = stratified_split(y_train_raw, val_size=val_size, seed=seed)

    x_train = _flatten_and_scale(x_train_raw[train_idx])
    x_val = _flatten_and_scale(x_train_raw[val_idx])
    y_train = y_train_raw[train_idx].astype(np.int64)
    y_val = y_train_raw[val_idx].astype(np.int64)

    # Normalização com média e desvio GLOBAIS (escalares), calculados só no
    # treino. Por pixel não funciona bem no MNIST: pixels de borda são quase
    # sempre 0, têm desvio minúsculo e explodiriam ao dividir. O desvio global
    # é dominado pelos pixels do centro e dá uma escala estável para todos.
    mean = float(x_train.mean())
    std = float(x_train.std())
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        # Guardados para normalizar o teste do mesmo jeito no dia 6.
        "mean": np.array(mean),
        "std": np.array(std),
    }
