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

- [01 · Modern Computer Vision Foundations](beginner/01-modern-computer-vision-foundations/README.md) — move from image contracts and convolution to scratch CNNs, real pretrained encoders, embeddings, source shift, failure analysis, and an enterprise review policy.
- [02 · Modern CNN Architectures & Efficient Vision](beginner/02-modern-cnn-architectures-efficient-vision/README.md) — understand residual, mobile, efficient, and modernized ConvNet designs; compare five encoders; and select against measured deployment constraints.
- [03 · Vision Transformers](beginner/03-vision-transformers/README.md) — turn images into tokens, build attention and a minimal ViT, compare CNN/ViT/Swin representations, test resolution shift, and review transformer systems evidence.
- [04 · Self-Supervised Visual Representation Learning](beginner/04-self-supervised-visual-representation-learning/README.md) — learn without manual pretraining labels through contrastive, teacher–student, and masked objectives; diagnose collapse; and evaluate label efficiency, retrieval, global features, and patch features.
- [05 · Object Detection](beginner/05-object-detection/README.md) — move from box contracts, matching, and AP to a tiny dense detector, NMS, Hungarian assignment, YOLO/DETR trade-offs, open-vocabulary extensions, and governed deployment evidence.
- [06 · Segmentation & Promptable Segmentation](beginner/06-segmentation-promptable-segmentation/README.md) — move from mask contracts, U-Net, overlap, and boundaries to source-sliced evaluation, interactive prompts, detector-box error propagation, SAM 3.1 governance, and human review.
- [07 · Visual Embeddings, Metric Learning & Retrieval](beginner/07-visual-embeddings-metric-learning-retrieval/README.md) — define similarity, learn normalized spaces, compare full-image and region retrieval, review hard negatives, verify exact search, tune HNSW, and govern embedding versions.
- [08 · Tracking, Keypoints & Pose](beginner/08-tracking-keypoints-pose/README.md) — maintain temporal identity through association and lifecycle policies, estimate structured landmarks, measure tracking and pose separately, and attribute failure propagation across the combined system.
- [09 · Vision Foundation Models & Open-Vocabulary Vision](beginner/09-vision-foundation-models-open-vocabulary/README.md) — synthesize the Beginner track through reusable global and patch features, image–text alignment, prompt/vocabulary evaluation, grounding, detector→segmenter composition, adaptation, and provenance.

## Intermediate

- [01 · Vision-Language Models: From Visual Features to Multimodal Reasoning](intermediate/01-vision-language-models/README.md) — move from image–text alignment to visual tokens, connectors, multimodal fusion, grounded generation, evidence ablation, capability-specific evaluation, and governed VLM inference.
- [02 · Multimodal Reasoning & Verification: From Visual Evidence to Checked Conclusions](intermediate/02-multimodal-reasoning-verification/README.md) — decompose bounded visual questions into source-bound evidence, deterministic tools, checked claims, counterfactuals, contradiction tests, uncertainty-aware decisions, and governed artifacts.
- [03 · Document Intelligence: From Pixels and Layout to Structured Evidence](intermediate/03-document-intelligence/README.md) — preserve page, region, reading-order, table-cell, normalization, and document lineage while extracting and verifying mixed enterprise documents.
- [04 · Multimodal Retrieval & RAG: From Evidence Indexing to Grounded Multimodal Answers](intermediate/04-multimodal-retrieval-rag/README.md) — index authorized pages, cells, figures, images, and regions; combine lexical, semantic, visual, structured, and multi-vector retrieval; assemble sufficient evidence; and verify claim-level citations.
- [05 · Video-Language Understanding: From Temporal Evidence to Grounded Video Reasoning](intermediate/05-video-language-understanding/README.md) — preserve frame and timestamp identity, compare sampling and representation policies, retrieve and ground events, verify multi-event claims, and evaluate long-video and streaming behavior.
- [06 · Visual Agents: From Multimodal Evidence to Bounded Tool-Using Systems](intermediate/06-visual-agents/README.md) — turn image, document, and video evidence into typed state; propose and authorize narrow tools; validate results; verify claims; enforce budgets and stop conditions; and evaluate safe recovery and escalation.

The [intermediate track](intermediate/README.md) culminates in bounded visual agency.

## Advanced

- [01 · 3D Vision & Spatial Intelligence: From Camera Geometry to Metric Scene Understanding](advanced/01-3d-vision-spatial-intelligence/README.md) — move from coordinate frames, pinhole projection, calibration, epipolar geometry, stereo, and triangulation into pose, point-cloud reconstruction, representation choice, uncertainty, and source-held-out metric decisions.
- [02 · Neural Rendering & 3D Scene Representations: From NeRFs to 3D Gaussian Splatting](advanced/02-neural-rendering-3d-scene-representations/README.md) — move from camera rays and volume rendering into held-out view synthesis, independent geometry evaluation, Gaussian projection and splatting, density-control lineage, systems trade-offs, and governed scene evidence.

The [advanced track](advanced/README.md) continues next into dynamic scenes, world models, adaptation, and embodied perception.

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
