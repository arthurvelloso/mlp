"""Gradient check como teste automático: roda no CI a cada push.

Garante que o backprop da rede inteira confere com o gradiente numérico,
numa rede pequena e com batch 4, para ser rápido.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gradcheck import check_mlp, numerical_gradient, relative_error
from src.network import MLP

LIMITE = 1e-7


def test_gradcheck_rede_pequena_batch_4():
    errors = check_mlp(layer_sizes=(6, 8, 5, 3), batch_size=4, seed=0)
    # Confere que cada parâmetro passou, com mensagem clara em caso de falha.
    for name, error in errors.items():
        assert error < LIMITE, f"{name}: erro relativo {error:.3e} acima de {LIMITE:.0e}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradcheck_varias_seeds(seed):
    errors = check_mlp(layer_sizes=(5, 7, 3), batch_size=4, seed=seed)
    assert max(errors.values()) < LIMITE


def test_relative_error_iguais_eh_zero():
    a = np.array([1.0, -2.0, 3.0])
    assert relative_error(a, a) == 0.0


def test_relative_error_ambos_zero_nao_estoura():
    zeros = np.zeros(3)
    assert relative_error(zeros, zeros) == 0.0


def test_numerical_gradient_funcao_quadratica():
    # Para f(x) = sum(x^2), o gradiente é 2x. Confere num ponto conhecido.
    x = np.array([1.0, -2.0, 3.0])

    def loss_at() -> float:
        return float(np.sum(x**2))

    gradient = numerical_gradient(loss_at, x)
    assert np.allclose(gradient, 2 * np.array([1.0, -2.0, 3.0]), atol=1e-6)


def test_backward_sem_forward_levanta_erro():
    network = MLP((4, 3), seed=0)
    with pytest.raises(RuntimeError):
        network.backward()
