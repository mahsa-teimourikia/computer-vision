# Computer Vision Field Guide

> A notebook-first path from image fundamentals to dependable, production-ready computer vision systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Learning materials](https://github.com/mahsa-teimourikia/computer-vision/actions/workflows/validate-learning.yml/badge.svg)](https://github.com/mahsa-teimourikia/computer-vision/actions/workflows/validate-learning.yml)

Computer vision turns images and video into measurable decisions. This repository is organized as a learning product: each lesson combines a technical chapter, a self-contained credential-free notebook, experiments, evaluation, failure analysis, and a focused checkpoint.

## Start here

Open the [Computer Vision Learning Hub](https://mahsa-teimourikia.github.io/computer-vision/) for the guided **Learn → Lab → Checkpoint** experience, or browse the [curriculum index](curriculum/README.md) directly on GitHub.

The first complete vertical slice is [Image Fundamentals](curriculum/beginner/01-image-fundamentals/README.md). It teaches arrays, channels, dtype, normalization, thresholding, and quantitative comparison without requiring a GPU, model download, or API key.

The research-grounded [curriculum architecture](docs/CURRICULUM_ARCHITECTURE.md) defines the complete beginner, intermediate, advanced, enterprise, and capstone journey before additional course content is generated.

## Curriculum roadmap

The long-form pathway connects six capability domains:

```text
Foundations
    ↓
Vision Foundation Models
    ↓
Multimodal Vision
    ↓
Spatial Intelligence
    ↓
Embodied Intelligence
    ↓
Enterprise CV
```

| Domain | Focus |
| --- | --- | --- |
| Foundations | Modern architectures; detection, segmentation, and tracking; vision transformers; self-supervised learning |
| Vision Foundation Models | Promptable and open-vocabulary vision, foundation segmentation, and vision embeddings |
| Multimodal Vision | Vision-language models, multimodal reasoning and RAG, video LLMs, and visual agents |
| Spatial Intelligence | Depth and geometry, 3D reconstruction, Gaussian splatting, generative 3D, dynamic scenes, and spatial reasoning |
| Embodied Intelligence | Vision-language-action, robot learning, egocentric vision, world models, simulation, and vision-based planning |
| Enterprise CV | Synthetic data, edge vision, evaluation, observability, security, robustness, privacy, and governance |

See [ROADMAP.md](ROADMAP.md) for the complete topic map, 2026 state-of-the-art radar, level mapping, and acceptance criteria. Use the [tooling review](TOOLING.md) to select learning, data, spatial, embodied, deployment, and operations tools deliberately.

## Repository structure

```text
curriculum/
├── beginner/
│   └── 01-image-fundamentals/
│       ├── README.md
│       ├── image_fundamentals.ipynb
│       ├── requirements.txt
│       └── assets/
├── intermediate/
├── advanced/
├── enterprise/
├── capstones/
└── shared/
assets/              # shared brand and global diagrams
data/                # small, redistributable datasets and fixtures
docs/                # curriculum architecture and design references
hub/                 # GitHub Pages learning experience
quiz/                # course-wide knowledge check as the curriculum grows
scripts/             # validation and notebook execution
tests/               # deterministic tests for labs and structure
```

Each topic owns its README, one self-contained primary notebook, a focused `requirements.txt`, and local assets. All teaching code stays in the notebook so the learning sequence can run from top to bottom without hidden local modules. Shared data fixtures belong in `curriculum/shared/` only after at least two lessons genuinely use them.

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
