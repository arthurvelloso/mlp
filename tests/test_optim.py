"""Confere os otimizadores do zero contra o ``torch.optim`` equivalente.

Para cada otimizador, roda vários passos num problema pequeno e compara os
parâmetros, passo a passo, com a versão do PyTorch. Se baterem dentro da
tolerância, a implementação está correta.

O PyTorch é dependência só de teste (ver requirements-dev.txt). Se não estiver
instalado, estes testes são pulados.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optim import SGD, Adam, AdamW

torch = pytest.importorskip("torch", reason="PyTorch é dependência só de teste")

TOLERANCE = 1e-6
NUM_STEPS = 25


def _run_ours(optimizer, param_init: np.ndarray, grad_fn) -> np.ndarray:
    """Roda NUM_STEPS da nossa implementação e devolve o parâmetro final."""
    param = param_init.copy()
    for _ in range(NUM_STEPS):
        grad = grad_fn(param)
        optimizer.step([param], [grad])
    return param


def _run_torch(make_optimizer, param_init: np.ndarray, grad_fn) -> np.ndarray:
    """Roda NUM_STEPS com o torch.optim e devolve o parâmetro final."""
    param = torch.tensor(param_init, dtype=torch.float64, requires_grad=True)
    optimizer = make_optimizer([param])
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        grad = grad_fn(param.detach().numpy())
        param.grad = torch.tensor(grad, dtype=torch.float64)
        optimizer.step()
    return param.detach().numpy()


def _quadratic_grad(target: np.ndarray):
    """Gradiente de f(p) = 0.5 * ||p - target||^2, que é (p - target)."""

    def grad_fn(param: np.ndarray) -> np.ndarray:
        return param - target

    return grad_fn


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    param_init = rng.normal(size=(4, 3))
    target = rng.normal(size=(4, 3))
    return param_init, _quadratic_grad(target)


def test_sgd(problem):
    param_init, grad_fn = problem
    ours = _run_ours(SGD(learning_rate=0.1), param_init, grad_fn)
    theirs = _run_torch(lambda p: torch.optim.SGD(p, lr=0.1), param_init, grad_fn)
    assert np.allclose(ours, theirs, atol=TOLERANCE)


def test_sgd_momentum(problem):
    param_init, grad_fn = problem
    ours = _run_ours(SGD(learning_rate=0.1, momentum=0.9), param_init, grad_fn)
    theirs = _run_torch(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), param_init, grad_fn)
    assert np.allclose(ours, theirs, atol=TOLERANCE)


def test_adam(problem):
    param_init, grad_fn = problem
    ours = _run_ours(Adam(learning_rate=0.01), param_init, grad_fn)
    theirs = _run_torch(lambda p: torch.optim.Adam(p, lr=0.01), param_init, grad_fn)
    assert np.allclose(ours, theirs, atol=TOLERANCE)


def test_adamw(problem):
    param_init, grad_fn = problem
    ours = _run_ours(AdamW(learning_rate=0.01, weight_decay=0.1), param_init, grad_fn)
    theirs = _run_torch(
        lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1), param_init, grad_fn
    )
    assert np.allclose(ours, theirs, atol=TOLERANCE)
