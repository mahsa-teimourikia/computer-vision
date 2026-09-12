# Computer Vision & Multimodal AI Field Guide

> A notebook-first path from pixels and learned representations to dependable multimodal, spatial, embodied, and enterprise vision systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Learning materials](https://github.com/mahsa-teimourikia/computer-vision/actions/workflows/validate-learning.yml/badge.svg)](https://github.com/mahsa-teimourikia/computer-vision/actions/workflows/validate-learning.yml)

Computer vision turns images and video into measurable decisions. This repository is organized as a learning product: each lesson combines a technical chapter, a self-contained credential-free notebook, experiments, evaluation, failure analysis, and a focused checkpoint.

## Start here

Open the [Computer Vision Learning Hub](https://mahsa-teimourikia.github.io/computer-vision/) for the guided **Learn → Lab → Checkpoint** experience, or browse the [curriculum index](curriculum/README.md) directly on GitHub.

The nine complete Beginner courses lead into six Intermediate courses and the first Advanced course:

1. [Modern Computer Vision Foundations](curriculum/beginner/01-modern-computer-vision-foundations/README.md) moves from image contracts and convolution to scratch CNNs, real pretrained encoders, embeddings, source shift, failure analysis, and enterprise decision policy.
2. [Modern CNN Architectures & Efficient Vision](curriculum/beginner/02-modern-cnn-architectures-efficient-vision/README.md) explains residual and efficient blocks, then compares five official pretrained backbones through controlled probes, profiling, resolution, robustness, Pareto fronts, and deployment contracts.
3. [Vision Transformers](curriculum/beginner/03-vision-transformers/README.md) moves from patches, positions, and manual attention to a minimal ViT, a controlled CNN/ViT/Swin benchmark, resolution interpolation, attention-distance diagnostics, and architecture selection.
4. [Self-Supervised Visual Representation Learning](curriculum/beginner/04-self-supervised-visual-representation-learning/README.md) moves from manual NT-Xent and augmentation contracts through matched contrastive/teacher–student/reconstruction experiments, explicit collapse diagnostics, repeated label-efficiency evaluation, global-versus-patch testing, and governed foundation-feature decisions.
5. [Object Detection](curriculum/beginner/05-object-detection/README.md) moves from box contracts, IoU, matching, AP, anchors, and feature pyramids through a trained dense detector, NMS, Hungarian assignment, YOLO/DETR design choices, open-vocabulary extensions, and source-aware deployment evidence.
6. [Segmentation & Promptable Segmentation](curriculum/beginner/06-segmentation-promptable-segmentation/README.md) moves from mask contracts, U-Net, losses, overlap, and boundaries through source-sliced semantic evaluation, prompt sensitivity, detector-box error propagation, human review, and a governed optional SAM 3.1 adapter.
7. [Visual Embeddings, Metric Learning & Retrieval](curriculum/beginner/07-visual-embeddings-metric-learning-retrieval/README.md) turns explicit similarity contracts into normalized embeddings, metric-learning objectives, simulated mining adjudication, exact and approximate search, retrieval failure analysis, and versioned enterprise evidence.
8. [Tracking, Keypoints & Pose](curriculum/beginner/08-tracking-keypoints-pose/README.md) builds timestamp-aware association, lifecycle, motion, appearance, and Byte-style recovery before connecting persistent identity to landmark accuracy, pose geometry, temporal stability, and failure propagation.
9. [Vision Foundation Models & Open-Vocabulary Vision](curriculum/beginner/09-vision-foundation-models-open-vocabulary/README.md) synthesizes the track through reusable visual features, image–text alignment, prompt and vocabulary evaluation, patch correspondence, language-conditioned localization, detector→segmenter composition, and governed adaptation.
10. [Vision-Language Models: From Visual Features to Multimodal Reasoning](curriculum/intermediate/01-vision-language-models/README.md) turns visual features into language-model context through projection, resampling, fusion, and causal generation, then tests grounding, hallucination, evidence, structured output, and systems contracts.
11. [Multimodal Reasoning & Verification: From Visual Evidence to Checked Conclusions](curriculum/intermediate/02-multimodal-reasoning-verification/README.md) decomposes bounded questions into evidence-bound facts, deterministic tool calls, checked claims, counterfactual and contradiction tests, uncertainty-aware decisions, and governed evidence artifacts.
12. [Document Intelligence: From Pixels and Layout to Structured Evidence](curriculum/intermediate/03-document-intelligence/README.md) routes mixed documents through page, OCR, layout, reading-order, table, form, normalization, provenance, verification, source-shift, and review contracts without flattening away evidence.
13. [Multimodal Retrieval & RAG: From Evidence Indexing to Grounded Multimodal Answers](curriculum/intermediate/04-multimodal-retrieval-rag/README.md) turns pages, cells, figures, images, and regions into authorized multimodal indexes, then measures hybrid retrieval, evidence sufficiency, claim-level citations, freshness, and retrieval-versus-generation failures.
14. [Video-Language Understanding: From Temporal Evidence to Grounded Video Reasoning](curriculum/intermediate/05-video-language-understanding/README.md) turns timestamped streams into sampled observations, temporal representations, event intervals, multi-event relations, long-video retrieval, verified citations, and bounded streaming decisions.
15. [Visual Agents: From Multimodal Evidence to Bounded Tool-Using Systems](curriculum/intermediate/06-visual-agents/README.md) turns image, document, and video evidence into typed state, permissioned tool proposals, validated execution, verified facts, bounded recovery, explicit stop decisions, and safe escalation.
16. [3D Vision & Spatial Intelligence: From Camera Geometry to Metric Scene Understanding](curriculum/advanced/01-3d-vision-spatial-intelligence/README.md) turns calibrated 2D observations into frame-aware projection, robust correspondence, stereo depth, triangulation, pose, reconstruction, uncertainty, and source-held-out metric spatial evidence.

All sixteen CPU-friendly notebooks use common PyTorch, torchvision, NumPy, pandas, Matplotlib, Pillow, SciPy, and scikit-learn APIs and keep all teaching code inside the notebook. Course 07 adds FAISS through its course-local requirements rather than imposing the native dependency on other learners; Courses 08–09 keep heavyweight official trackers, pose models, foundation checkpoints, and remote code optional. Intermediate 01 exposes the common Hugging Face multimodal processor/chat-template/generation path behind an explicit opt-in flag and immutable model revisions. Intermediate 02 uses standard Python, NumPy, pandas, Pillow, and Matplotlib to keep its DAG, tool, verification, and evidence contracts fully observable without an agent framework. Intermediate 03 keeps its synthetic OCR/layout pipeline transparent and places Tesseract, Table Transformer, and PaddleOCR-VL behind revision-pinned, disabled-by-default adapters. Intermediate 04 implements BM25, semantic/visual/multi-vector proxies, RRF, reranking, evidence assembly, ACL checks, and citation verification directly; FAISS, BGE-M3, SigLIP 2, BGE reranking, and ColQwen remain course-local, pinned, and disabled by default. Intermediate 05 implements timestamp-safe sampling, temporal retrieval, interval tools, evidence completeness, citation verification, counterfactuals, and streaming state directly. Intermediate 06 uses standard Python, NumPy, pandas, Pillow, and Matplotlib to keep typed state, tool contracts, authorization, validation, budgets, traces, failure attribution, and held-out evaluation observable; Qwen3-VL, LangGraph, the OpenAI Agents SDK, and MCP remain disabled optional mappings rather than hidden execution dependencies. Advanced 01 uses NumPy, SciPy, pandas, and Matplotlib to expose projection, DLT, RANSAC, stereo uncertainty, pose optimisation, point refinement, reconstruction metrics, and held-out spatial decisions; Depth Anything V2, UniDepth, COLMAP, Open3D, VGGT, and VGGT-Ω remain revision-pinned, disabled optional integrations.

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
| --- | --- |
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
│   ├── 01-modern-computer-vision-foundations/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # validated SVGs + coordinate specs
│   ├── 02-modern-cnn-architectures-efficient-vision/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 03-vision-transformers/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 04-self-supervised-visual-representation-learning/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 05-object-detection/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 06-segmentation-promptable-segmentation/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 07-visual-embeddings-metric-learning-retrieval/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 08-tracking-keypoints-pose/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   └── 09-vision-foundation-models-open-vocabulary/
│       ├── README.md
│       ├── lab.ipynb
│       ├── requirements.txt
│       └── assets/            # deterministic SVGs + coordinate specs
├── intermediate/
│   ├── 01-vision-language-models/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 02-multimodal-reasoning-verification/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 03-document-intelligence/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 04-multimodal-retrieval-rag/
│   │   ├── README.md
│   │   ├── lab.ipynb
│   │   ├── requirements.txt
│   │   └── assets/            # deterministic SVGs + coordinate specs
│   ├── 05-video-language-understanding/
│       ├── README.md
│       ├── lab.ipynb
│       ├── requirements.txt
│       └── assets/            # deterministic SVGs + coordinate specs
│   └── 06-visual-agents/
│       ├── README.md
│       ├── lab.ipynb
│       ├── requirements.txt
│       └── assets/            # deterministic SVGs + coordinate specs
├── advanced/
│   └── 01-3d-vision-spatial-intelligence/
│       ├── README.md
│       ├── lab.ipynb
│       ├── requirements.txt
│       └── assets/            # deterministic SVGs + coordinate specs
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
