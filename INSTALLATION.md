# Installation and environment guide

The Computer Vision Field Guide has three supported setup modes. Choose the smallest environment that fits your work.

## Requirements

- Python **3.13** is recommended and used by CI. Python 3.12 is supported.
- Git is required for a local checkout.
- `make` is recommended on macOS and Linux. Windows learners can use the equivalent Python commands below or the development container.
- The default notebooks are CPU-safe and credential-free. Optional foundation models, provider APIs, robotics stacks, GPU runtimes, and deployment systems are not installed automatically.

The repository `.python-version`, development container, package metadata, and CI all target the same Python 3.13 line.

## Option 1 — use the Learning Hub without installing anything

Open the [Computer Vision Learning Hub](https://mahsa-teimourikia.github.io/computer-vision/). Each available lesson provides:

- a technical chapter;
- a direct notebook link;
- a focused checkpoint with answer feedback; and
- local progress tracking stored only in your browser.

Running the practical labs still requires a local or hosted Jupyter environment.

## Option 2 — install one course

This is the smallest reproducible setup. Every completed course owns a `requirements.txt` and a `constraints-tested.txt` file.

```bash
git clone https://github.com/mahsa-teimourikia/computer-vision.git
cd computer-vision
make setup-course COURSE=curriculum/beginner/01-modern-computer-vision-foundations
make notebooks COURSE=curriculum/beginner/01-modern-computer-vision-foundations
```

Replace `COURSE` with any available course directory. The command creates `.venv`, installs the course against its tested constraints, and registers the `Computer Vision Field Guide` Jupyter kernel.

Without `make`:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r curriculum/beginner/01-modern-computer-vision-foundations/requirements.txt
.venv/bin/python -m ipykernel install --user --name computer-vision-field-guide --display-name "Computer Vision Field Guide"
.venv/bin/python -m jupyterlab curriculum/beginner/01-modern-computer-vision-foundations
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## Option 3 — install the complete repository environment

Use this mode to run every notebook and repository check. It includes the common learner libraries, contributor tooling, and the Course 07 FAISS dependency used by the complete notebook sweep.

```bash
git clone https://github.com/mahsa-teimourikia/computer-vision.git
cd computer-vision
make setup
make check
```

`make setup-learner` installs the common notebook stack without contributor tooling or FAISS. `make setup` installs the complete contributor/CI environment.

## Development container

Open the repository in a tool that supports `devcontainer.json`, then choose **Reopen in Container**. The container uses Python 3.13 and installs the contributor environment automatically.

## Verified package set

The repository reviewed these package releases on **September 27, 2026** and validates them together on Python 3.13 before marking the constraints tested.

| Purpose | Package versions |
| --- | --- |
| arrays and tables | [NumPy 2.5.3](https://pypi.org/project/numpy/), [pandas 3.0.6](https://pypi.org/project/pandas/) |
| scientific and classical ML | [SciPy 1.18.1](https://pypi.org/project/scipy/), [scikit-learn 1.9.1](https://pypi.org/project/scikit-learn/), [NetworkX 3.7](https://pypi.org/project/networkx/) |
| visualization and images | [Matplotlib 3.11.2](https://pypi.org/project/matplotlib/), [Pillow 12.3.0](https://pypi.org/project/pillow/) |
| deep learning and vision | [PyTorch 2.14.0](https://pypi.org/project/torch/), [torchvision 0.29.0](https://pypi.org/project/torchvision/) |
| model SDK | [Transformers 5.17.0](https://pypi.org/project/transformers/) |
| notebooks | [JupyterLab 4.6.4](https://pypi.org/project/jupyterlab/), [ipykernel 7.3.0](https://pypi.org/project/ipykernel/) |
| course-specific retrieval | [FAISS CPU 1.15.1](https://pypi.org/project/faiss-cpu/) |
| validation | [pytest 9.1.1](https://pypi.org/project/pytest/), [nbclient 0.11.0](https://pypi.org/project/nbclient/), [nbformat 5.11.1](https://pypi.org/project/nbformat/) |

Latest does not automatically mean compatible. `pyproject.toml` defines the repository environment; each course constraint file defines the smaller environment actually exercised for that lesson. Update either only with a clean notebook and test run.

## Common commands

| Command | Purpose |
| --- | --- |
| `make help` | list supported workflows |
| `make notebooks` | open all curriculum notebooks in JupyterLab |
| `make notebooks COURSE=<path>` | open one course directory |
| `make test` | run deterministic tests |
| `make quiz-check` | validate every Hub checkpoint |
| `make links` | validate topic structure, local links, Hub coverage, and setup metadata |
| `make diagrams` | regenerate and validate deterministic course diagrams |
| `make notebook-check` | execute all credential-free notebooks in isolated copies |
| `make pages` | preview the Hub at `http://localhost:8000` |
| `make check` | run the complete local validation suite |

## Optional integrations

Optional model downloads and production systems remain isolated from the default path. Before enabling one:

1. pin an immutable code and checkpoint revision;
2. review the code, model, dataset, and transitive licenses;
3. verify preprocessing and processor identity;
4. test the required Python, accelerator, and runtime combination separately;
5. define data retention, credential, and network policy; and
6. rerun capability-specific evaluation on the target source and hardware.

The relevant course README and `constraints-tested.txt` list reviewed optional systems and their boundaries.

## Troubleshooting

### The project refuses to install

Check the interpreter first:

```bash
python3.13 --version
```

The tested project range is Python 3.12–3.13. Remove and recreate `.venv` if it was built with an older interpreter.

### Jupyter cannot find the kernel

Run:

```bash
.venv/bin/python -m ipykernel install --user --name computer-vision-field-guide --display-name "Computer Vision Field Guide"
```

Then select **Computer Vision Field Guide** in JupyterLab.

### FAISS fails to install

FAISS is required only for the Course 07 native approximate-search extension and the complete contributor validation environment. Use `make setup-learner` or a course-specific installation when you do not need that extension.

### A heavyweight optional import is unavailable

The notebooks label optional imports and keep their default path runnable without them. Follow the pinned adapter instructions in that course rather than adding an unbounded package to the shared environment.

### A notebook works only from one directory

Launch it through `make notebooks` or set JupyterLab to the course folder. Repository validation executes each notebook with its own course directory as the working directory.
