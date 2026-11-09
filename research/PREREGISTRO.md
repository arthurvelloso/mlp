# Pré-registro: learning rate ótimo vs. batch size

> Este documento é escrito e commitado **antes** de rodar os experimentos. O
> commit datado funciona como pré-registro: fixa a pergunta, as hipóteses, a
> grade e as métricas antes de ver qualquer resultado, para não ajustar a
> conclusão aos dados depois.

## Pergunta

O learning rate ótimo cresce com o batch size de forma **linear** no SGD, como
sugerem Goyal et al. (2017), e pela **raiz quadrada** no Adam, como sugerem
Malladi et al. (2022)? E qual otimizador converge numa faixa maior de learning
rates?

## Hipóteses (fixadas antes de rodar)

- **H1:** no SGD (com e sem momentum), a inclinação de `log(LR ótimo)` contra
  `log(batch)` fica perto de **1**; no Adam, perto de **0,5**.
- **H2:** o Adam alcança 97% de acurácia de validação numa faixa de learning
  rates mais larga que o SGD.

## Grade

- **Otimizadores:** SGD, SGD+momentum (0,9), Adam.
- **Learning rates:** 8 valores espaçados por fator 2, por otimizador.
  - SGD e SGD+momentum: `{0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6}`
  - Adam: `{6.25e-5, 1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3}`
  - As faixas são diferentes de propósito: SGD e Adam operam em escalas de LR
    distintas. O que importa é cada uma cobrir o ponto ótimo do seu otimizador.
- **Batches:** `{32, 128, 512}`.
- **Seeds:** `{0, 1, 2}`.
- **Total:** 3 × 8 × 3 × 3 = **216 runs**.
- **Orçamento:** número fixo de épocas por run (ver `configs/research.yaml`),
  o mesmo para todos. Subconjunto do MNIST para caber no tempo medido no dia 3.

## Métricas (por run)

- Acurácia final de validação.
- Épocas até atingir 97% de validação (vazio se nunca atingir).
- Se o run divergiu (loss virou NaN/inf ou acurácia ficou no nível do acaso).
- Tempo de execução.

Agregações: média e desvio entre seeds; "LR ótimo" por (otimizador, batch) é o
LR com maior acurácia média de validação.

## O que conta como "pronto"

- Um CSV com uma linha por run (otimizador, lr, batch, seed, métricas, tempo).
- 3 ou 4 figuras geradas por script.
- Uma tabela-resumo e as primeiras conclusões no diário.

Qualquer resultado serve, inclusive "a regra não se confirmou neste setup".

## Limitações já conhecidas

- Um dataset (MNIST), uma arquitetura, grade de LR discreta e orçamento curto.
- "LR ótimo" é o melhor da grade, não um ótimo contínuo; a resolução é o fator 2.
- 3 seeds dão uma ideia da variação, não um intervalo de confiança estreito.

## Referências

- Goyal et al. (2017), "Accurate, Large Minibatch SGD". arXiv:1706.02677.
- Malladi et al. (2022), "On the SDEs and Scaling Rules for Adaptive Gradient
  Algorithms". arXiv:2205.10287.
- "The Best Optimizer Depends on Batch Size" (arXiv, out/2026) — leitura
  relacionada, posterior ao desenho deste experimento.
