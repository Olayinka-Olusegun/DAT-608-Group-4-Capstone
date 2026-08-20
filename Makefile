PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install reference ingest backfill features train score brief alert pipeline figures test api app status clean

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

reference:
	$(PYTHON) -m pau_risk.cli reference

ingest:
	$(PYTHON) -m pau_risk.cli ingest --days 45

backfill:
	$(PYTHON) -m pau_risk.cli ingest --since 2010-01-01 --until 2024-12-31

features:
	$(PYTHON) -m pau_risk.cli features

train:
	$(PYTHON) -m pau_risk.cli train

score:
	$(PYTHON) -m pau_risk.cli score

brief:
	$(PYTHON) -m pau_risk.cli brief

alert:
	$(PYTHON) -m pau_risk.cli alert

pipeline:
	$(PYTHON) -m pau_risk.cli pipeline

tune:
	$(PYTHON) scripts/tune_model.py

figures:
	$(PYTHON) scripts/make_figures.py

test:
	$(PYTHON) -m pytest tests -q

api:
	$(PYTHON) -m pau_risk.cli serve --port 8000

app:
	cd app && Rscript -e 'shiny::runApp(".", port = 7788, launch.browser = FALSE)'

status:
	$(PYTHON) -m pau_risk.cli status

clean:
	rm -rf data/processed/*.parquet data/artifacts/model data/streams/*
