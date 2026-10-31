# Atalhos para as tarefas do dia a dia.
# Rode `make` ou `make help` para ver os comandos.

.PHONY: help install test lint format gradcheck train reproduce clean

help:  ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala as dependências
	pip install -r requirements.txt

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

reproduce:  ## Reproduz o resultado principal (gradient check + treino)
	$(MAKE) gradcheck
	$(MAKE) train

clean:  ## Remove arquivos temporários
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__
