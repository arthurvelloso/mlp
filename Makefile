# Atalhos para as tarefas do dia a dia.
# Rode `make` ou `make help` para ver os comandos.

.PHONY: help install test lint format gradcheck train plots compare research reproduce clean

help:  ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala as dependências (produção + desenvolvimento)
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:  ## Roda os testes
	pytest

lint:  ## Verifica o estilo do código (não altera arquivos)
	ruff check .
	ruff format --check .

format:  ## Formata o código (altera arquivos)
	ruff format .
	ruff check --fix .

gradcheck:  ## Confere os gradientes por diferenças finitas
	python -m src.gradcheck

train:  ## Treina a MLP no MNIST
	python -m src.train

plots:  ## Gera as curvas de loss e acurácia a partir do CSV
	python -m src.plots

compare:  ## Compara os quatro otimizadores na mesma rede
	python -m src.compare_optimizers

research:  ## Roda a grade do estudo e gera gráficos + tabela
	python -m src.research_run
	python -m src.research_plots

reproduce:  ## Reproduz o resultado principal (gradient check + treino + curvas)
	$(MAKE) gradcheck
	$(MAKE) train
	$(MAKE) plots

clean:  ## Remove arquivos temporários
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__
