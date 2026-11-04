"""Camadas básicas de uma rede neural implementadas somente com NumPy."""

from __future__ import annotations

import numpy as np


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produto de matrizes ignorando flags de ponto flutuante espúrios.

    Em algumas combinações de NumPy e BLAS (por exemplo, o Accelerate da Apple
    em Python 3.14), ``@`` dispara avisos falsos de "divide by zero"/"overflow"
    mesmo com entradas pequenas e resultado correto. Isolamos o produto num
    ``errstate`` para não poluir a saída; erros numéricos reais ainda aparecem
    como valores não finitos, que o treino detectaria pela loss.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


class Linear:
    """Transformação afim ``y = x @ weights + bias``.

    Os pesos usam inicialização de He, adequada para redes com ReLU:
    ``weights ~ N(0, 2 / in_features)``.

    Args:
        in_features: número de atributos de entrada.
        out_features: número de unidades de saída.
        seed: semente opcional para tornar a inicialização reprodutível.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("in_features precisa ser um inteiro positivo")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError("out_features precisa ser um inteiro positivo")

        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.default_rng(seed)
        he_scale = np.sqrt(2.0 / in_features)
        self.weights = rng.standard_normal((in_features, out_features)) * he_scale
        self.bias = np.zeros(out_features, dtype=np.float64)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._input: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calcula a saída da camada e guarda a entrada para o backward.

        Args:
            inputs: lote com shape ``(batch_size, in_features)``.

        Returns:
            Saída com shape ``(batch_size, out_features)``.
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2:
            raise ValueError("inputs precisa ter shape (batch_size, in_features)")
        if inputs.shape[1] != self.in_features:
            raise ValueError(f"esperava {self.in_features} atributos, recebeu {inputs.shape[1]}")

        self._input = inputs.copy()
        return _matmul(self._input, self.weights) + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Calcula os gradientes dos parâmetros e da entrada.

        Args:
            grad_output: gradiente da loss em relação à saída da camada, com
                shape ``(batch_size, out_features)``.

        Returns:
            Gradiente da loss em relação à entrada, com
            shape ``(batch_size, in_features)``.
        """
        if self._input is None:
            raise RuntimeError("forward precisa ser chamado antes de backward")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        expected_shape = (self._input.shape[0], self.out_features)
        if grad_output.shape != expected_shape:
            raise ValueError(
                f"grad_output deveria ter shape {expected_shape}, recebeu {grad_output.shape}"
            )

        # A loss já calcula a média do lote. Não se divide pelo batch novamente aqui.
        self.grad_weights[...] = _matmul(self._input.T, grad_output)
        self.grad_bias[...] = np.sum(grad_output, axis=0)
        return _matmul(grad_output, self.weights.T)


class ReLU:
    """Ativação ReLU, ``max(0, x)``.

    Em ``x = 0``, esta implementação adota derivada igual a zero.
    """

    def __init__(self) -> None:
        self._positive_mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Aplica ReLU elemento a elemento e guarda sua máscara."""
        inputs = np.asarray(inputs)
        self._positive_mask = inputs > 0
        return np.maximum(inputs, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Bloqueia o gradiente onde a entrada do forward não era positiva."""
        if self._positive_mask is None:
            raise RuntimeError("forward precisa ser chamado antes de backward")

        grad_output = np.asarray(grad_output)
        if grad_output.shape != self._positive_mask.shape:
            raise ValueError(
                "grad_output precisa ter o mesmo shape da entrada usada no forward: "
                f"{self._positive_mask.shape}, recebeu {grad_output.shape}"
            )

        return grad_output * self._positive_mask
