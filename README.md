# Computer Vision Field Guide

> A notebook-first path from image fundamentals to dependable, production-ready computer vision systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Learning materials](https://github.com/mahsa-teimourikia/computer-vision/actions/workflows/validate-learning.yml/badge.svg)](https://github.com/mahsa-teimourikia/computer-vision/actions/workflows/validate-learning.yml)

Computer vision turns images and video into measurable decisions. This repository is organized as a learning product: each lesson combines a technical chapter, a credential-free notebook, reusable Python code, experiments, evaluation, failure analysis, and a focused checkpoint.

## Start here

Open the [Computer Vision Learning Hub](https://mahsa-teimourikia.github.io/computer-vision/) for the guided **Learn → Lab → Checkpoint** experience, or browse the [curriculum index](curriculum/README.md) directly on GitHub.

The first complete vertical slice is [Image Fundamentals](curriculum/beginner/01-image-fundamentals/README.md). It teaches arrays, channels, dtype, normalization, thresholding, and quantitative comparison without requiring a GPU, model download, or API key.

## Curriculum roadmap

| Level | Focus | Status |
| --- | --- | --- |
| Beginner | Pixels, color, filtering, features, and first classifiers | Image fundamentals available |
| Intermediate | Detection, segmentation, augmentation, transfer learning, and evaluation | Planned |
| Advanced | Video, multimodal systems, robustness, optimization, and deployment | Planned |
| Enterprise | Governance, privacy, monitoring, human oversight, and fleet operations | Planned |

See [ROADMAP.md](ROADMAP.md) for the ordered topic plan and acceptance criteria.

## Repository structure

```text
curriculum/
├── beginner/
│   └── 01-image-fundamentals/
│       ├── README.md
│       ├── image_fundamentals.ipynb
│       ├── lab.py
│       └── assets/
├── intermediate/
├── advanced/
├── enterprise/
└── shared/
assets/              # shared brand and global diagrams
data/                # small, redistributable datasets and fixtures
hub/                 # GitHub Pages learning experience
quiz/                # course-wide knowledge check as the curriculum grows
scripts/             # validation and notebook execution
tests/               # deterministic tests for labs and structure
```

Each topic owns its README, one primary notebook, reusable `lab.py`, and local assets. Shared code belongs in `curriculum/shared/` only after at least two lessons use it.

## Run locally

```bash
make setup
make test
make notebook-check
```

Use `make notebooks` to launch JupyterLab and `make pages` to preview the Hub at `http://localhost:8000`.

## Learning contract

Every completed lesson should let a learner:

- explain the underlying mechanism and trade-offs;
- implement the core technique in a deterministic notebook;
- inspect intermediate representations and metrics;
- inject a realistic failure and evaluate a mitigation; and
- describe the production, safety, privacy, and monitoring implications.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a lesson. Contributions should deepen the connected learning path rather than add disconnected links or demos.

Built as an open learning project by [One+i](https://oneplusi.io).
