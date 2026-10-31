# MLP do zero em NumPy

Uma rede neural (Multilayer Perceptron) implementada do zero, só com NumPy,
sem PyTorch nem TensorFlow. O objetivo é entender o que acontece por baixo dos
frameworks: o forward, o backprop e o treino, com os gradientes conferidos por
diferenças finitas.

> **Status:** em construção. As camadas (`Linear`, `ReLU`, `SoftmaxCrossEntropy`)
> já estão implementadas e validadas. O treino no MNIST é o próximo passo.

## Resultado

O número que importa fica aqui, logo no topo, assim que o treino estiver pronto.

> _Em breve:_ MLP em NumPy atingindo X% de acurácia no MNIST (validação), com
> gradientes conferidos por diferenças finitas (erro relativo < 1e-7).

| Etapa | Status |
| --- | --- |
| Camadas `Linear`, `ReLU`, `SoftmaxCrossEntropy` | ✅ implementadas |
| Gradient check (diferenças finitas) | ✅ validado (erro ~1e-10) |
| Treino no MNIST | ⬜ a fazer |
| Curvas de loss e acurácia | ⬜ a fazer |

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
| `make gradcheck` | confere os gradientes por diferenças finitas | ⬜ em breve |
| `make train` | treina a MLP no MNIST | ⬜ em breve |
| `make reproduce` | gradient check + treino | ⬜ em breve |

Os alvos `gradcheck`, `train` e `reproduce` dependem de scripts ainda não
escritos (`src/gradcheck.py`, `src/train.py`).

## Decisões

Por que o projeto é do jeito que é. Decisões que não são óbvias pelo código.

- **NumPy puro:** sem framework de deep learning, de propósito. A ideia é
  implementar forward e backward na mão.
- **Softmax + cross-entropy numa classe só:** junto é numericamente estável e o
  gradiente fica simples (`probabilidades − one_hot`); separado, o cálculo pode
  estourar com logits grandes.
- **Inicialização de He na `Linear`:** adequada para redes com ReLU.

## Limitações

O que o projeto **não** garante. Seja honesto aqui.

- O treino e as métricas ainda não existem; por enquanto só há as camadas.
- Foco didático: não é otimizado para velocidade nem pensado para produção.

## Estrutura

```
.
├── src/
│   ├── layers.py   # Linear, ReLU
│   └── losses.py   # SoftmaxCrossEntropy
├── tests/          # testes automatizados (pytest)
├── notebooks/      # exploração; o código "de verdade" mora em src/
├── configs/        # hiperparâmetros e caminhos, versionados
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Licença

Este projeto usa a licença MIT. Veja [LICENSE](LICENSE).
