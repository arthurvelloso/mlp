# Resumo do estudo

- Total de runs: 216
- Runs que divergiram: 41

## LR ótimo por (otimizador, batch)

| Otimizador | Batch | LR ótimo | Acurácia média |
| --- | --- | --- | --- |
| sgd | 32 | 0.2 | 0.9640 |
| sgd | 128 | 0.4 | 0.9579 |
| sgd | 512 | 0.4 | 0.9532 |
| momentum | 32 | 0.025 | 0.9644 |
| momentum | 128 | 0.05 | 0.9608 |
| momentum | 512 | 0.1 | 0.9558 |
| adam | 32 | 0.00025 | 0.9582 |
| adam | 128 | 0.001 | 0.9607 |
| adam | 512 | 0.002 | 0.9575 |

## H1: inclinação de log(LR ótimo) vs. log(batch)

Esperado: ~1 para SGD/momentum (regra linear), ~0,5 para Adam (raiz).

| Otimizador | Inclinação ajustada |
| --- | --- |
| sgd | 0.25 |
| momentum | 0.50 |
| adam | 0.75 |

## H2: largura da faixa de bons learning rates

Alvo absoluto de 97%: número de combinações (LR, batch)
cuja acurácia média o atinge. Neste orçamento curto, pode ser 0 para
todos (o melhor resultado fica abaixo do alvo).

| Otimizador | Combinações no alvo (de 24) |
| --- | --- |
| sgd | 0 |
| momentum | 0 |
| adam | 0 |

Medida relativa (independe do alvo): combinações (LR, batch) a no máximo
1 ponto percentual do melhor resultado do próprio otimizador. Faixa mais
larga = mais tolerante à escolha do learning rate.

| Otimizador | Combinações perto do melhor (de 24) |
| --- | --- |
| sgd | 6 |
| momentum | 8 |
| adam | 14 |

> Conclusões escritas no diário. Qualquer resultado é válido, inclusive
> 'a regra não se confirmou neste setup'.
