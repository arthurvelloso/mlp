# Nome do projeto

Uma frase dizendo o que o projeto faz e para quem.

> Template de repositório para projetos de ML. Copie, renomeie e preencha as seções abaixo.
> Apague esta citação quando o projeto for real.

## Resultado

O número que importa, logo no topo. Exemplo:

> MLP em NumPy atinge **97,8% de acurácia** no MNIST (validação), com gradientes
> conferidos por diferenças finitas (erro relativo < 1e-7).

Se tiver, coloque aqui a figura principal:

<!-- ![Curva de loss](reports/figures/loss.png) -->

| Modelo | Métrica | Valor |
| --- | --- | --- |
| Baseline | acurácia | — |
| Modelo final | acurácia | — |

## Como reproduzir

Testado em Python 3.11+.

```bash
# 1. criar e ativar o ambiente
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate

# 2. instalar as dependências (versões fixas)
pip install -r requirements.txt

# 3. rodar os testes (deve passar tudo)
make test

# 4. reproduzir o resultado principal
make reproduce
```

`make reproduce` roda com uma seed fixa, então rodar de novo dá o mesmo número.

## Decisões

Por que o projeto é do jeito que é. Decisões que não são óbvias pelo código.

- **Métrica:** por que macro-F1 e não acurácia (classes desbalanceadas), por exemplo.
- **Split:** como treino, validação e teste foram separados e por quê.
- **Modelo:** por que este e não outro.

## Limitações

O que o projeto **não** garante. Seja honesto aqui.

- Um único dataset; não há garantia de que generaliza para outros.
- Poucas seeds; a variação pode ser maior do que a medida.
- O que ficou de fora por falta de tempo.

## Estrutura

```
.
├── src/            # código do projeto (importável)
├── tests/          # testes automatizados (pytest)
├── notebooks/      # exploração; o código "de verdade" mora em src/
├── configs/        # hiperparâmetros e caminhos, versionados
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Licença

Este projeto usa a licença MIT. Veja [LICENSE](LICENSE).
