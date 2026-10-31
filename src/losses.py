"""Funções de loss implementadas somente com NumPy."""

from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax e cross-entropy combinadas de forma numericamente estável.

    A classe recebe logits, não probabilidades. O ``forward`` retorna a loss
    média do lote e o ``backward`` retorna seu gradiente em relação aos logits:
    ``(probabilities - one_hot(targets)) / batch_size``.
    """

    def __init__(self) -> None:
        self.probabilities: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Calcula a cross-entropy média de um lote.

        Args:
            logits: escores sem normalização, com shape
                ``(batch_size, num_classes)``.
            targets: índices inteiros das classes corretas, com shape
                ``(batch_size,)``.

        Returns:
            Cross-entropy média do lote.
        """
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets)

        if logits.ndim != 2:
            raise ValueError("logits precisa ter shape (batch_size, num_classes)")

        batch_size, num_classes = logits.shape
        if batch_size == 0:
            raise ValueError("o lote não pode estar vazio")
        if num_classes < 2:
            raise ValueError("são necessárias pelo menos duas classes")
        if targets.ndim != 1 or targets.shape[0] != batch_size:
            raise ValueError(f"targets deveria ter shape ({batch_size},), recebeu {targets.shape}")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("targets precisa conter índices inteiros de classes")
        if np.any(targets < 0) or np.any(targets >= num_classes):
            raise ValueError(f"cada target precisa estar entre 0 e {num_classes - 1}")

        targets = targets.astype(np.intp, copy=True)

        # logsumexp(logits) = max(logits) + log(sum(exp(logits - max(logits)))).
        # Como subtraímos o logsumexp dos próprios logits, podemos trabalhar
        # inteiramente com os logits deslocados sem alterar as probabilidades.
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        log_normalizer = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        log_probabilities = shifted_logits - log_normalizer

        self.probabilities = np.exp(log_probabilities)
        self._targets = targets

        sample_losses = -log_probabilities[np.arange(batch_size), targets]
        return float(np.mean(sample_losses))

    def backward(self) -> np.ndarray:
        """Retorna o gradiente da loss média em relação aos logits."""
        if self.probabilities is None or self._targets is None:
            raise RuntimeError("forward precisa ser chamado antes de backward")

        batch_size = self.probabilities.shape[0]
        grad_logits = self.probabilities.copy()
        grad_logits[np.arange(batch_size), self._targets] -= 1.0
        grad_logits /= batch_size
        return grad_logits
