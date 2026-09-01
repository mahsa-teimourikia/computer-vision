VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP = $(PYTHON) -m pip

.PHONY: help setup test notebooks notebook-check diagrams links pages

help:
	@echo "make setup          Create the environment and install contributor tools"
	@echo "make test           Run deterministic tests"
	@echo "make notebooks      Start JupyterLab in the curriculum directory"
	@echo "make notebook-check Execute all credential-free notebooks"
	@echo "make diagrams       Validate and render course diagrams"
	@echo "make links          Validate the learning-product structure and local links"
	@echo "make pages          Preview the static Learning Hub"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[contributor]'
	$(PYTHON) -m ipykernel install --user --name computer-vision-field-guide --display-name "Computer Vision Field Guide"

test:
	PYTHONPATH=. $(PYTHON) -m pytest -q

notebooks:
	PYTHONPATH=. $(PYTHON) -m jupyterlab curriculum

notebook-check:
	PYTHONPATH=. $(PYTHON) scripts/execute_notebooks.py --timeout 300

diagrams:
	$(PYTHON) scripts/render_course_diagrams.py curriculum/beginner/*/assets/specs/*.json

links:
	$(PYTHON) scripts/validate_structure.py

pages:
	python3 -m http.server 8000 --directory hub
