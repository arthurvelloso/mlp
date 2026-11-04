"""Leitura da configuração de treino a partir de um arquivo YAML.

Manter os hiperparâmetros num arquivo (e não espalhados pelo código) deixa os
experimentos reprodutíveis: muda o YAML, commita, e dá para saber exatamente
qual configuração gerou cada resultado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Carrega o YAML de configuração como dicionário."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config não encontrada: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("a config precisa ser um mapeamento (chave: valor)")
    return config
