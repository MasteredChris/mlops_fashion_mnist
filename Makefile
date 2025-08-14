# Variabili
PYTHON := python
PACKAGE := mlops-fmnist
OUTPUT_DIR := outputs
DATA_DIR := data
SHELL := /bin/bash


# Ambiente virtuale (default .venv)
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

# Comandi principali
.PHONY: help venv install eda train eval test lint build docker-build docker-run clean

help:
	@echo "Comandi disponibili:"
	@echo "  make venv           - crea ambiente virtuale"
	@echo "  make install        - installa dipendenze"
	@echo "  make eda            - esegue analisi esplorativa"
	@echo "  make train          - addestra il modello"
	@echo "  make eval           - valuta il modello"
	@echo "  make test           - esegue unit test"
	@echo "  make lint           - esegue flake8"
	@echo "  make build          - build pacchetto wheel"
	@echo "  make docker-build   - build immagine Docker"
	@echo "  make docker-run     - run training in Docker"
	@echo "  make clean          - pulizia file temporanei"

venv:
	$(PYTHON)3 -m venv $(VENV)
	@echo "Attiva con: source $(VENV)/bin/activate"

install:
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt && pip install -e . && pip install pytest flake8 build

eda:
	$(ACTIVATE) && $(PYTHON) scripts/eda.py

train:
	$(ACTIVATE) && $(PYTHON) -m src.mlops_fmnist.train --epochs 5 --output_dir $(OUTPUT_DIR)

eval:
	$(ACTIVATE) && $(PYTHON) -m src.mlops_fmnist.evaluate --checkpoint $(OUTPUT_DIR)/model.pth

test:
	$(ACTIVATE) && pytest -q

lint:
	$(ACTIVATE) && flake8 src tests --max-line-length=100

build:
	$(ACTIVATE) && python -m build

docker-build:
	sudo docker build -t $(PACKAGE) .

docker-run:
	sudo docker run --rm -v $(PWD)/$(OUTPUT_DIR):/app/outputs -v $(PWD)/$(DATA_DIR):/app/data $(PACKAGE)

clean:
	rm -rf $(VENV) dist build *.egg-info __pycache__ */__pycache__ $(OUTPUT_DIR) $(DATA_DIR) .pytest_cache
