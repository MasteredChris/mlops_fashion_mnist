# MLOps Fashion-MNIST

Piccolo progetto AI: CNN per classificare Fashion-MNIST con pipeline MLOps (tests, packaging, Docker, CI).

## Setup

```bash
make venv
source .venv/bin/activate
make install
```

## Avviamento e test

```bash
make eda
make train
make eval
make test
make lint
make build
make docker-build
make docker-run
```

## Reset e pulizia

```bash
make clean
```
