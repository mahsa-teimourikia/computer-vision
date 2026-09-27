VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP = $(PYTHON) -m pip
PYTHON_BOOTSTRAP ?= python3.13
KERNEL ?= computer-vision-field-guide

.DEFAULT_GOAL := help
.PHONY: help setup setup-learner setup-contributor setup-course test quiz-check notebooks notebook-check diagrams links pages check

help:
	@echo "make setup                         Install the complete contributor/CI environment"
	@echo "make setup-learner                 Install the common learner environment"
	@echo "make setup-course COURSE=<path>    Install one course from its tested constraints"
	@echo "make test                          Run deterministic tests"
	@echo "make quiz-check                    Validate all Hub checkpoints"
	@echo "make notebooks [COURSE=<path>]     Start JupyterLab for all or one course"
	@echo "make notebook-check                Execute all credential-free notebooks"
	@echo "make diagrams                      Validate and render course diagrams"
	@echo "make links                         Validate structure, links, Hub, and setup metadata"
	@echo "make pages                         Preview the static Learning Hub"
	@echo "make check                         Run the complete local validation suite"

$(PYTHON):
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)

setup: setup-contributor

setup-learner: $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[learner]'
	$(PYTHON) -m ipykernel install --user --name $(KERNEL) --display-name "Computer Vision Field Guide"

setup-contributor: $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[contributor]'
	$(PYTHON) -m ipykernel install --user --name $(KERNEL) --display-name "Computer Vision Field Guide"

setup-course: $(PYTHON)
	@test -n "$(COURSE)" || (echo "Set COURSE, for example COURSE=curriculum/beginner/01-modern-computer-vision-foundations" && exit 2)
	@test -f "$(COURSE)/requirements.txt" || (echo "No course requirements found at $(COURSE)/requirements.txt" && exit 2)
	$(PIP) install --upgrade pip
	$(PIP) install -r "$(COURSE)/requirements.txt"
	$(PYTHON) -m ipykernel install --user --name $(KERNEL) --display-name "Computer Vision Field Guide"

test:
	PYTHONPATH=. $(PYTHON) -m pytest -q

quiz-check:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/test_structure.py -k "hub or quiz"

notebooks:
	PYTHONPATH=. $(PYTHON) -m jupyterlab $(if $(COURSE),$(COURSE),curriculum)

notebook-check:
	PYTHONPATH=. $(PYTHON) scripts/execute_notebooks.py --timeout 300 --kernel-name $(KERNEL)

diagrams:
	$(PYTHON) scripts/render_course_diagrams.py curriculum/*/*/assets/specs/*.json

links:
	$(PYTHON) scripts/validate_structure.py

pages:
	$(PYTHON) -m http.server 8000 --directory hub

check: links diagrams test quiz-check notebook-check
