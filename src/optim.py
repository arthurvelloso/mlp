"""Otimizadores implementados do zero, com interface comum.

Todos seguem o mesmo contrato:

    optimizer.step(params, grads)

onde ``params`` e ``grads`` são listas paralelas de arrays NumPy (um gradiente
por parâmetro, na mesma ordem a cada chamada). Os parâmetros são atualizados no
lugar. Otimizadores com estado (momentum, Adam, AdamW) guardam um buffer por
parâmetro, indexado pela posição na lista.

As fórmulas seguem as do ``torch.optim`` para que os resultados batam passo a
passo (ver ``tests/test_optim.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Params = list[np.ndarray]
Grads = list[np.ndarray]


class Optimizer(ABC):
    """Interface comum: um método ``step(params, grads)``."""

    @abstractmethod
    def step(self, params: Params, grads: Grads) -> None:
        """Atualiza ``params`` no lugar usando ``grads``."""
        raise NotImplementedError


class SGD(Optimizer):
    """Gradiente descendente estocástico, com momentum opcional.

    Sem momentum (``momentum=0``): ``p -= lr * g``.

    Com momentum, segue a formulação do PyTorch:
        ``b <- momentum * b + g``
        ``p <- p - lr * b``
    O primeiro buffer é inicializado com o próprio gradiente, como no PyTorch.
    """

    def __init__(self, learning_rate: float, momentum: float = 0.0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate precisa ser positivo")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum precisa estar em [0, 1)")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._buffers: list[np.ndarray] | None = None

    def step(self, params: Params, grads: Grads) -> None:
        if self.momentum == 0.0:
            for param, grad in zip(params, grads, strict=True):
                param -= self.learning_rate * grad
            return

        if self._buffers is None:
            # PyTorch inicia o buffer com o gradiente no primeiro passo.
            self._buffers = [grad.copy() for grad in grads]
            for param, buffer in zip(params, self._buffers, strict=True):
                param -= self.learning_rate * buffer
            return

        for param, grad, buffer in zip(params, grads, self._buffers, strict=True):
            buffer *= self.momentum
            buffer += grad
            param -= self.learning_rate * buffer


class Adam(Optimizer):
    """Adam com correção de viés.

        ``m <- beta1 * m + (1 - beta1) * g``
        ``v <- beta2 * v + (1 - beta2) * g^2``
        ``m_hat = m / (1 - beta1^t)``
        ``v_hat = v / (1 - beta2^t)``
        ``p <- p - lr * m_hat / (sqrt(v_hat) + eps)``

    ``m`` e ``v`` começam em zero; a correção de viés compensa esse início.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate precisa ser positivo")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 e beta2 precisam estar em [0, 1)")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self._t = 0

    def _ensure_state(self, params: Params) -> None:
        if self._m is None:
            self._m = [np.zeros_like(param) for param in params]
            self._v = [np.zeros_like(param) for param in params]

    def step(self, params: Params, grads: Grads) -> None:
        self._ensure_state(params)
        self._t += 1
        assert self._m is not None and self._v is not None

        bias_correction1 = 1.0 - self.beta1**self._t
        bias_correction2 = 1.0 - self.beta2**self._t

        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            m = self._m[index]
            v = self._v[index]
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (grad * grad)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW: Adam com weight decay desacoplado.

    Diferente do Adam com L2, o decaimento não entra no gradiente nem nos
    momentos; é aplicado direto ao parâmetro, como no PyTorch:
        ``p <- p - lr * weight_decay * p``   (decaimento desacoplado)
        depois o passo normal de Adam.

    É o otimizador padrão no treino de LLMs.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate precisa ser positivo")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 e beta2 precisam estar em [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay não pode ser negativo")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self._t = 0

    def _ensure_state(self, params: Params) -> None:
        if self._m is None:
            self._m = [np.zeros_like(param) for param in params]
            self._v = [np.zeros_like(param) for param in params]

    def step(self, params: Params, grads: Grads) -> None:
        self._ensure_state(params)
        self._t += 1
        assert self._m is not None and self._v is not None

        bias_correction1 = 1.0 - self.beta1**self._t
        bias_correction2 = 1.0 - self.beta2**self._t

        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            # Weight decay desacoplado: aplicado ao parâmetro, fora do gradiente.
            param -= self.learning_rate * self.weight_decay * param

            m = self._m[index]
            v = self._v[index]
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (grad * grad)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
