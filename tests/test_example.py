"""Testes das funções de exemplo.

Servem de modelo: um teste do caso normal, um dos limites e um dos erros.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.example import accuracy, train_val_split


def test_accuracy_tudo_certo():
    y = np.array([0, 1, 2, 1])
    assert accuracy(y, y) == 1.0


def test_accuracy_metade():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 0.5


def test_accuracy_shapes_diferentes_levanta_erro():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0, 1, 0]))


def test_accuracy_vazio_levanta_erro():
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_split_tamanhos():
    train, val = train_val_split(100, val_fraction=0.2, seed=0)
    assert len(train) == 80
    assert len(val) == 20


def test_split_sem_sobreposicao():
    train, val = train_val_split(100, val_fraction=0.3, seed=0)
    assert set(train).isdisjoint(set(val))
    assert sorted([*train, *val]) == list(range(100))


def test_split_reprodutivel():
    a = train_val_split(50, seed=42)
    b = train_val_split(50, seed=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_split_fracao_invalida():
    with pytest.raises(ValueError):
        train_val_split(100, val_fraction=1.5)
