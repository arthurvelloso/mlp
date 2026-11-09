# Diário do estudo (dia 5)

## O que foi feito

- Pré-registro (`PREREGISTRO.md`) escrito e commitado antes de rodar.
- Grade de 216 runs: 3 otimizadores × 8 LRs × 3 batches × 3 seeds, 15 épocas
  cada, treino de 10 mil exemplos do MNIST. Rodou em ~10 min.
- CSV com uma linha por run (`results.csv`), gravado incrementalmente.
- 3 figuras por script e tabela-resumo (`summary.md`).

## Resultados

### H1: inclinação de log(LR ótimo) vs. log(batch)

Esperado: ~1 (SGD/momentum, regra linear) e ~0,5 (Adam, raiz quadrada).
Medido: **SGD 0,25 · momentum 0,50 · Adam 0,75**.

**H1 não se confirmou neste setup.** A ordem até fica ao contrário do esperado:
o Adam teve a inclinação mais alta, não a mais baixa. Possíveis razões, a
investigar:

- A grade de LR é grossa (fator 2): o "LR ótimo" só pode cair em 8 valores, e
  a inclinação é sensível a isso. Com só 3 batches, dois pontos quase iguais já
  mudam bastante a reta.
- As regras de escala valem perto do ótimo e com treino longo; 15 épocas num
  subconjunto pode estar longe desse regime.
- Para SGD, o ótimo de batch 128 e 512 caiu no mesmo LR (0,4), achatando a reta.

### H2: tolerância ao learning rate

Com o alvo absoluto de 97%, **nenhuma** combinação atingiu (o melhor foi
~96,4%), então essa métrica não discrimina neste orçamento. Pela medida
relativa (dentro de 1 ponto do melhor de cada otimizador):

- **Adam: 14** combinações · **momentum: 8** · **SGD: 6**.

**H2 se confirmou na direção esperada:** o Adam tolera uma faixa bem mais larga
de learning rates. As figuras deixam claro: SGD e momentum têm um platô curto e
despencam para ~0 quando o LR passa do ponto (41 runs divergiram, todos deles);
o Adam é uma curva suave em sino e não divergiu em nenhum LR da grade.

## Conclusão (honesta)

- A regra de escala linear vs. raiz (H1) **não apareceu** aqui. O resultado é
  inconclusivo sobre as teorias: o setup é curto e a grade é grossa demais para
  estimar inclinações com confiança. Não dá para dizer que as teorias estão
  erradas; dá para dizer que este experimento não as detecta.
- A robustez do Adam ao learning rate (H2) **apareceu com clareza** e é
  consistente com a prática.

## Próximos passos (se virar algo maior)

- Grade de LR mais fina (fator √2) e mais batches, para estimar a inclinação.
- Orçamento maior (mais épocas, treino completo) e alvo coerente com ele.
- 5 seeds nas finalistas; repetir no Fashion-MNIST.
- Ler "The Best Optimizer Depends on Batch Size" (arXiv, out/2026): argumenta
  que a melhor escolha de otimizador muda com o batch, o que conversa com H2.
