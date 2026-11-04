"""Rede MLP: encadeia camadas e faz o backprop da rede inteira.

O forward passa a entrada por cada camada em ordem e guarda, em cada uma, o
que o backward vai precisar (isso já acontece dentro de ``Linear`` e ``ReLU``).
O backward percorre as camadas na ordem inversa, propagando o gradiente.
"""

from __future__ import annotations

import numpy as np

from src.layers import Linear, ReLU
from src.losses import SoftmaxCrossEntropy


class MLP:
    """Perceptron multicamadas com ativação ReLU e loss softmax + cross-entropy.

    A arquitetura é ``Linear → ReLU → Linear → ReLU → ... → Linear``, ou seja,
    uma ReLU entre cada par de camadas lineares, sem ativação após a última
    (a ``SoftmaxCrossEntropy`` cuida da saída).

    Args:
        layer_sizes: tamanhos das camadas, da entrada à saída. Por exemplo,
            ``(784, 128, 10)`` cria duas camadas lineares (784→128 e 128→10)
            com uma ReLU no meio.
        seed: semente opcional para a inicialização dos pesos ser reprodutível.
    """

    def __init__(
        self,
        layer_sizes: list[int] | tuple[int, ...],
        *,
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes precisa ter pelo menos entrada e saída")

        rng = np.random.default_rng(seed)
        self.linears: list[Linear] = []
        self.layers: list[Linear | ReLU] = []

        for index in range(len(layer_sizes) - 1):
            # Seed derivada por camada: reprodutível e diferente em cada uma.
            layer_seed = int(rng.integers(0, 2**31 - 1))
            linear = Linear(layer_sizes[index], layer_sizes[index + 1], seed=layer_seed)
            self.linears.append(linear)
            self.layers.append(linear)
            # ReLU entre as lineares, mas não depois da última.
            if index < len(layer_sizes) - 2:
                self.layers.append(ReLU())

        self.loss_fn = SoftmaxCrossEntropy()

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        """Passa o lote pelas camadas e retorna os logits (sem a loss).

        Usado para prever e medir acurácia. Também guarda, em cada camada, o
        que o backward precisa, então pode ser combinado com ``forward``.
        """
        activations = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def forward(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Passa o lote pela rede e retorna a loss média.

        Cada camada guarda internamente o que seu backward precisa, então não
        é necessário acumular ativações aqui.
        """
        return self.loss_fn.forward(self.logits(inputs), targets)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Retorna a classe prevista (argmax dos logits) para cada exemplo."""
        return np.argmax(self.logits(inputs), axis=1)

    def backward(self) -> None:
        """Backprop da rede inteira, preenchendo os gradientes de cada Linear.

        Começa no gradiente da loss em relação aos logits e propaga para trás,
        camada por camada, na ordem inversa do forward.
        """
        gradient = self.loss_fn.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Lista os parâmetros treináveis e seus gradientes.

        Returns:
            Lista de pares ``(parametro, gradiente)``, onde ``parametro`` e
            ``gradiente`` são o mesmo array que a camada usa internamente
            (views, não cópias). Útil para o gradient check e para o treino.
        """
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for linear in self.linears:
            pairs.append((linear.weights, linear.grad_weights))
            pairs.append((linear.bias, linear.grad_bias))
        return pairs
