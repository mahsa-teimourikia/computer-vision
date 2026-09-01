# Curriculum

The Computer Vision Field Guide is organized by increasing system complexity. Begin with observable image mechanics, then move into learned representations, task-specific architectures, evaluation, robustness, deployment, and governance.

## Strategic pathway

The lesson levels below are the delivery structure; the connected subject pathway is:

1. **Foundations** — modern vision architectures; detection, segmentation, and tracking; vision transformers; self-supervised learning.
2. **Vision Foundation Models** — promptable vision, open-vocabulary vision, foundation segmentation, and vision embeddings.
3. **Multimodal Vision** — vision-language models, multimodal reasoning, video LLMs, multimodal RAG, and visual agents.
4. **Spatial Intelligence** — depth and geometry, 3D reconstruction, Gaussian splatting, generative 3D, 4D dynamic scenes, and spatial reasoning.
5. **Embodied Intelligence** — vision-language-action, robot learning, egocentric vision, world models, simulation, and vision-based planning.
6. **Enterprise CV** — synthetic data, edge vision, evaluation, observability, security and robustness, privacy, and governance.

Topics move into higher lesson levels as their theory, system complexity, operational risk, or governance burden increases.

The [2026 state-of-the-art radar](../ROADMAP.md#2026-state-of-the-art-radar) distinguishes established practice, emerging practice, and research-frontier topics. The [tooling review](../TOOLING.md) defines the comparison required in each lesson.

## Beginner

- [01 · Image fundamentals](beginner/01-image-fundamentals/README.md) — inspect pixels, channels, dtype and range; build and evaluate a deterministic threshold segmenter.
- Filtering and edges *(planned)*
- Features and matching *(planned)*
- First image classifier *(planned)*

## Intermediate

The [intermediate track](intermediate/README.md) will cover convolutional networks, detection, segmentation, data quality, transfer learning, and evaluation.

## Advanced

The [advanced track](advanced/README.md) will cover video, vision transformers, multimodal systems, robustness, efficient inference, and production operations.

## Enterprise

The [enterprise track](enterprise/README.md) will cover responsible use, governance, assurance, and fleet operations.

## Capstones

The [capstone track](capstones/README.md) integrates completed prerequisites into five production-style multimodal, video, spatial, inspection, and embodied systems.

## How to study a lesson

1. Read the lesson chapter and define the system boundary.
2. Run the notebook from top to bottom without changing it.
3. Change one meaningful variable and inspect the metric and intermediate output.
4. Trigger the documented failure, apply the mitigation, and explain the trade-off.
5. Complete the Hub checkpoint and extend one exercise.
