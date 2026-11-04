# MLP do zero em NumPy

Uma rede neural (Multilayer Perceptron) implementada do zero, só com NumPy,
sem PyTorch nem TensorFlow. O objetivo é entender o que acontece por baixo dos
frameworks: o forward, o backprop e o treino, com os gradientes conferidos por
diferenças finitas.

> **Status:** em construção. Camadas, backprop, gradient check e treino no MNIST
> estão prontos. Otimizadores adicionais e a avaliação no teste vêm depois.

## Resultado

> MLP 784-256-128-10 em NumPy com SGD atinge **97,94% de acurácia na validação**
> (10 mil exemplos, seed 0), com gradientes conferidos por diferenças finitas
> (erro relativo máximo ~7e-10).

![Curva de acurácia](reports/figures/accuracy.png)
![Curva de loss](reports/figures/loss.png)

As figuras são geradas por `make plots`. Elas ficam fora do git; rode
`make reproduce` para recriá-las.

| Etapa | Status |
| --- | --- |
| Camadas `Linear`, `ReLU`, `SoftmaxCrossEntropy` | ✅ implementadas |
| Gradient check (diferenças finitas) | ✅ validado, também como teste no CI |
| Treino no MNIST (mini-batches, CSV, early stopping, checkpoint) | ✅ |
| Curvas de loss e acurácia por script | ✅ |
| Avaliação no teste oficial | ⬜ dia 6 |

## O que já tem

Em `src/`:

- **`layers.py`**
  - `Linear` — transformação afim `y = x @ W + b`, com inicialização de He.
  - `ReLU` — ativação `max(0, x)`.
- **`losses.py`**
  - `SoftmaxCrossEntropy` — softmax e cross-entropy combinadas de forma
    numericamente estável (log-sum-exp). O gradiente na saída é
    `probabilidades − one_hot`.

Cada camada tem `forward` e `backward`, do jeito que um framework faz por dentro.
Exemplo de uma rede `Linear → ReLU → Linear → Softmax`:

```python
import numpy as np

from src.layers import Linear, ReLU
from src.losses import SoftmaxCrossEntropy

x = np.random.default_rng(0).normal(size=(32, 784))
targets = np.random.default_rng(1).integers(0, 10, size=32)

linear_1 = Linear(784, 128, seed=0)
relu = ReLU()
linear_2 = Linear(128, 10, seed=1)
loss_fn = SoftmaxCrossEntropy()

# Forward
hidden = relu.forward(linear_1.forward(x))
logits = linear_2.forward(hidden)
loss = loss_fn.forward(logits, targets)

# Backward
gradient = loss_fn.backward()
gradient = linear_2.backward(gradient)
gradient = relu.backward(gradient)
gradient = linear_1.backward(gradient)
```

## Como reproduzir

Testado em Python 3.11+.

```bash
# 1. criar e ativar o ambiente
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate

# 2. instalar as dependências (versões fixas)
pip install -r requirements.txt

# 3. rodar os testes
make test
```

Comandos do `Makefile`:

| Comando | O que faz | Disponível |
| --- | --- | --- |
| `make test` | roda os testes | ✅ |
| `make lint` | verifica o estilo (ruff) | ✅ |
| `make format` | formata o código (ruff) | ✅ |
| `make gradcheck` | confere os gradientes por diferenças finitas | ✅ |
| `make train` | treina no MNIST e grava `reports/metrics.csv` | ✅ |
| `make plots` | gera as curvas a partir do CSV | ✅ |
| `make reproduce` | gradient check + treino + curvas | ✅ |

O primeiro `make train` baixa o MNIST para `data/mnist.npz` (uma vez só).
Os hiperparâmetros ficam em `configs/default.yaml`. Com a mesma config e a mesma
seed, o CSV de métricas sai idêntico a cada execução.

## Decisões

Por que o projeto é do jeito que é. Decisões que não são óbvias pelo código.

- **NumPy puro:** sem framework de deep learning, de propósito. A ideia é
  implementar forward e backward na mão.
- **Softmax + cross-entropy numa classe só:** junto é numericamente estável e o
  gradiente fica simples (`probabilidades − one_hot`); separado, o cálculo pode
  estourar com logits grandes.
- **Inicialização de He na `Linear`:** adequada para redes com ReLU.
- **Split:** 50 mil treino e 10 mil validação, estratificado e com seed fixa. Os
  10 mil de teste oficiais não são usados até a avaliação final.
- **Normalização global:** média e desvio escalares, calculados só no treino.
  Por pixel não funciona no MNIST: pixels de borda quase sempre valem 0, e
  dividir pelo desvio minúsculo deles gerou entradas de até ~100 e uma loss de
  validação na casa de centenas.
- **Early stopping e checkpoint pela acurácia de validação.**

## Limitações

O que o projeto **não** garante. Seja honesto aqui.

- Uma seed e um split só; a variação entre seeds ainda não foi medida.
- Há overfitting leve: a loss de treino vai a ~0 e a de validação sobe um pouco
  depois da época ~8. O checkpoint usa a acurácia, não a loss.
- A validação serviu para escolher a época. O número real de generalização
  virá do teste oficial.
- Foco didático: não é otimizado para velocidade nem pensado para produção.

## Estrutura

```
.
├── src/
│   ├── layers.py   # Linear, ReLU
│   ├── losses.py   # SoftmaxCrossEntropy
│   ├── network.py  # MLP: forward e backprop da rede inteira
│   ├── gradcheck.py# gradient check por diferenças finitas
│   ├── data.py     # MNIST, split estratificado, normalização
│   ├── optim.py    # SGD
│   ├── train.py    # loop de treino, CSV, early stopping, checkpoint
│   └── plots.py    # curvas a partir do CSV
├── tests/          # testes automatizados (pytest)
├── notebooks/      # exploração; o código "de verdade" mora em src/
├── configs/        # hiperparâmetros e caminhos, versionados
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Licença

Este projeto usa a licença MIT. Veja [LICENSE](LICENSE).
