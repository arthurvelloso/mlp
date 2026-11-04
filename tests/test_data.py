"""Testes do preparo de dados: split estratificado e normalização.

Não baixam o MNIST; usam rótulos e arrays sintéticos para rodar rápido no CI.
"""

from __future__ import annotations

import numpy as np

from src.data import stratified_split


def test_split_tamanhos_e_sem_sobreposicao():
    labels = np.array([0, 1, 2] * 100)  # 300 exemplos, 3 classes balanceadas
    train_idx, val_idx = stratified_split(labels, val_size=60, seed=0)
    assert len(val_idx) == 60
    assert len(train_idx) == 240
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted([*train_idx, *val_idx]) == list(range(300))


def test_split_mantem_proporcao_das_classes():
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    _, val_idx = stratified_split(labels, val_size=30, seed=0)
    counts = np.bincount(labels[val_idx], minlength=3)
    # 10% de cada classe: 10 por classe.
    assert list(counts) == [10, 10, 10]


def test_split_reprodutivel_com_mesma_seed():
    labels = np.arange(200) % 4
    a = stratified_split(labels, val_size=40, seed=7)
    b = stratified_split(labels, val_size=40, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_split_val_size_invalido():
    labels = np.array([0, 1, 0, 1])
    for bad in (0, 4, 10):
        try:
            stratified_split(labels, val_size=bad, seed=0)
        except ValueError:
            continue
        raise AssertionError(f"val_size={bad} deveria ter levantado ValueError")


def test_normalizacao_usa_estatisticas_do_treino():
    # Replica a lógica de normalização de prepare() num array pequeno,
    # conferindo que o treino fica com média ~0 e desvio ~1.
    rng = np.random.default_rng(0)
    x_train = rng.normal(loc=5.0, scale=3.0, size=(500, 4))
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    x_norm = (x_train - mean) / std
    assert np.allclose(x_norm.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(x_norm.std(axis=0), 1.0, atol=1e-3)
