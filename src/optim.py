"""Otimizador. Por enquanto só SGD; outros entram em dias seguintes do plano."""

from __future__ import annotations

import numpy as np


class SGD:
    """Gradiente descendente estocástico simples.

    Atualiza cada parâmetro na direção oposta ao seu gradiente:
    ``parametro -= learning_rate * gradiente``.

    Args:
        parameters: lista de pares ``(parametro, gradiente)``, como a devolvida
            por ``MLP.parameters()``. Os arrays são atualizados no lugar.
        learning_rate: tamanho do passo.
    """

    def __init__(
        self, parameters: list[tuple[np.ndarray, np.ndarray]], learning_rate: float
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate precisa ser positivo")
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self) -> None:
        """Aplica um passo de atualização em todos os parâmetros."""
        for parameter, gradient in self.parameters:
            parameter -= self.learning_rate * gradient
