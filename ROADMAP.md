# Curriculum roadmap

The curriculum progresses through six connected capability domains. Lessons remain filed by difficulty level so learners can take a practical beginner → intermediate → advanced → enterprise route; this domain map keeps the larger technical story visible.

## Strategic learning pathway

```text
FOUNDATIONS
│
├── Modern Vision Architectures
├── Detection / Segmentation / Tracking
├── Vision Transformers
└── Self-Supervised Learning
          ↓
VISION FOUNDATION MODELS
│
├── Promptable Vision
├── Open-Vocabulary Vision
├── Foundation Segmentation
└── Vision Embeddings
          ↓
MULTIMODAL VISION
│
├── Vision-Language Models
├── Multimodal Reasoning
├── Video LLMs
├── Multimodal RAG
└── Visual Agents
          ↓
SPATIAL INTELLIGENCE
│
├── Depth / Geometry
├── 3D Reconstruction
├── Gaussian Splatting
├── Generative 3D
├── 4D Dynamic Scenes
└── Spatial Reasoning
          ↓
EMBODIED INTELLIGENCE
│
├── Vision-Language-Action
├── Robot Learning
├── Egocentric Vision
├── World Models
├── Simulation
└── Vision-Based Planning
          ↓
ENTERPRISE CV
│
├── Synthetic Data
├── Edge Vision
├── Evaluation
├── Observability
├── Security / Robustness
├── Privacy
└── Governance
```

### Domain-to-level mapping

| Domain | Likely lesson levels | Progression goal |
| --- | --- | --- |
| Foundations | Beginner → Advanced | Move from observable image mechanics to modern architectures and representation learning |
| Vision Foundation Models | Intermediate → Advanced | Adapt general-purpose visual representations through prompts, open vocabularies, masks, and embeddings |
| Multimodal Vision | Intermediate → Advanced | Ground language and agent behavior in images and video, with explicit evaluation and tool boundaries |
| Spatial Intelligence | Intermediate → Advanced | Recover and reason over geometry, scenes, time, and generated 3D representations |
| Embodied Intelligence | Advanced → Enterprise | Connect perception to action in simulated and physical environments with safety controls |
| Enterprise CV | Intermediate → Enterprise | Evaluate, deploy, monitor, secure, govern, and responsibly operate the full stack |

## 2026 state-of-the-art radar

This radar was reviewed on **2026-08-31**. It is a curriculum watchlist, not a claim that every new release is production-ready. Each future lesson must recheck the source, reproduce relevant results on an appropriate baseline, document licensing and compute, and distinguish research evidence from vendor demonstrations.

### Established and rapidly consolidating

- **Self-supervised universal visual backbones:** [DINOv3](https://ai.meta.com/research/publications/dinov3/) scales self-supervised image learning and targets transferable dense features across classification, detection, depth, segmentation, and tracking. Lessons should compare frozen features, lightweight adapters, and task-specific fine-tuning rather than assume one backbone dominates every domain.
- **Concept-prompted segmentation and tracking:** [SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) extends promptable segmentation toward text/exemplar concepts and instance identities across images and video. Teach the evolution from closed-set masks → visual prompts → video memory → open-vocabulary concept prompts, with false-positive, temporal-consistency, and domain-slice evaluation.
- **Long-context image/video-language modeling:** the [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631) is a current primary reference for interleaved multimodal context, video understanding, and visual reasoning. Treat benchmark claims as model-specific evidence and test OCR, localization, temporal ordering, calibration, and hallucination separately.
- **Open robot foundation policies:** [NVIDIA Isaac GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) and [Gemini Robotics](https://deepmind.google/models/gemini-robotics/) make VLA policies, embodied reasoning, long-horizon planning, and multiple embodiments central study areas. Physical deployment remains high-risk and must be separated from safe simulation labs.

### Emerging practice

- **World foundation models for physical AI:** [NVIDIA Cosmos](https://research.nvidia.com/labs/cosmos-lab/) treats video generation, physical reasoning, simulation, and policy data as a connected platform. Lessons should compare predictive fidelity, controllability, action relevance, uncertainty, and simulation-to-real transfer—not only visual realism.
- **Spatial foundation models:** object-centric spatial reasoning benchmarks such as [Spatial Reasoning in Foundation Models](https://arxiv.org/abs/2509.21922) expose gaps that ordinary recognition scores miss. Curriculum work should connect 2D grounding to depth, camera geometry, reconstruction, 3D/4D representations, navigation, and action.
- **Gaussian and hybrid scene representations:** the original [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) work and [4D Gaussian Splatting](https://arxiv.org/abs/2310.08528) motivate explicit coverage of real-time rendering, dynamic scenes, reconstruction quality, memory, editing, and geometric limitations alongside NeRF-style methods.
- **Agentic visual systems:** visual agents increasingly combine VLM perception with search, code, GUI, spatial, or robot tools. Lessons must use typed tool contracts, narrow permissions, result validation, budgets, stop conditions, and observable traces; model-generated reasoning is not authorization.
- **On-device multimodal and VLA inference:** compact models, quantization, native-resolution inputs, and hardware-specific runtimes are moving more perception and control toward the edge. Evaluate privacy gains against thermal limits, memory, energy, cold start, degraded modes, and update safety.

### Research frontier

- **Streaming/codec-native video models:** work such as [Mage-VL](https://arxiv.org/abs/2607.24904) explores avoiding redundant frame decoding and improving continual multimodal perception. It is a new research direction; teach reproducibility, latency methodology, causal/streaming constraints, and comparisons with sampled-frame baselines.
- **Reinforcement learning for visual reasoning:** projects such as [Open Vision Reasoner](https://arxiv.org/abs/2507.05255) study multimodal reasoning trained with verifiable rewards. Treat benchmark improvements cautiously and evaluate localization, faithfulness, reward hacking, compute, and out-of-distribution behavior.
- **Generative 3D/4D and action-conditioned scene simulation:** rapid progress connects diffusion/flow models, neural rendering, video, geometry, and world models. Lessons should distinguish compelling generation from accurate, controllable, temporally consistent, physically useful simulation.
- **Multi-robot and whole-body embodied intelligence:** current systems explore high-level embodied reasoning, collaboration, and whole-body control. Course coverage should emphasize hierarchical control, latency, recovery, human interruption, formal safety envelopes, and real-world evidence limits.

### How a lesson may claim “state of the art”

1. Date the review and link the primary paper, official technical report, model card, code, and dataset where available.
2. Name the exact task, benchmark version, split, metric, input resolution, test-time policy, hardware, precision, and comparison class.
3. Separate author-reported results from locally reproduced results.
4. Review training/evaluation data overlap, contamination, license, access, demographic/geographic coverage, and domain transfer.
5. Include a simpler baseline and at least one credible alternative architecture or tool.
6. Report failures, uncertainty, latency, memory, energy/cost, and operational constraints alongside quality.
7. Label the item established, emerging, or research frontier; do not convert a leaderboard position into a universal recommendation.

## Delivery plan by lesson level

## Beginner

1. **Modern Computer Vision Foundations** — task contracts, image tensors, convolution, receptive fields, CNNs, transfer learning, pretrained encoders, embeddings, leakage, shortcut learning, metrics, shift testing, and enterprise decision policy. *(available)*
2. **Modern CNN Architectures & Efficient Vision** — residual, ConvNeXt, efficient and hybrid designs; profiling, adaptation depth, and target-hardware trade-offs. *(available)*
3. **Vision Transformers** — patches, positions, attention, hierarchy, CNN/ViT trade-offs, and resolution shift. *(available)*
4. **Self-Supervised Visual Representation Learning** — contrastive, masked, and teacher-student learning; probing, retrieval, and collapse prevention. *(available)*
5. **Object Detection** — localisation, matching, DETR/YOLO-style systems, small-object slices, error diagnosis, and the transition from closed-set to open-set detection. *(available)*
6. **Segmentation & Promptable Segmentation** — semantic, instance, and panoptic outputs; mask contracts, boundaries, prompt sensitivity, detector-to-mask error propagation, human review, and governed foundation-model comparison. *(available)*
7. **Visual Embeddings, Metric Learning & Retrieval** — Siamese models, contrastive/triplet objectives, FAISS-style similarity search, re-identification, hard-example discovery, and retrieval evaluation. *(available)*
8. **Tracking, Keypoints & Pose** — temporal identity, occlusion, association, landmarks, pose estimation, and video evaluation.
9. **Vision Foundation Models & Open-Vocabulary Vision** — promptable and open-vocabulary perception, concept segmentation, reusable embeddings, and governed adaptation.

## Intermediate

1. **Convolutional networks** — receptive fields, feature hierarchies, architecture choices, and transfer learning.
2. **Object detection** — localization, IoU, non-maximum suppression, mAP, and error diagnosis.
3. **Semantic and instance segmentation** — masks, loss functions, class imbalance, and boundary metrics.
4. **Data quality and augmentation** — leakage, label noise, invariance, synthetic data, and dataset shift.
5. **Evaluation and explainability** — calibration, slices, attribution limits, robustness, and release gates.

## Advanced

1. **Video understanding and tracking** — temporal sampling, motion, identity association, and drift.
2. **Vision transformers** — tokenization, attention, scaling behavior, and hybrid architectures.
3. **Self-supervised visual learning** — pretext tasks, contrastive objectives, masked modeling, transfer, and collapse prevention.
4. **Vision foundation models** — promptable and open-vocabulary vision, foundation segmentation, embeddings, adaptation, and evaluation.
5. **Multimodal vision-language systems** — grounding, multimodal reasoning and RAG, video LLMs, visual agents, and evaluation.
6. **Spatial intelligence** — depth, geometry, reconstruction, Gaussian splatting, generative 3D, 4D scenes, and spatial reasoning.
7. **Embodied intelligence** — vision-language-action, robot learning, egocentric vision, world models, simulation, and planning.
8. **Robustness and domain adaptation** — corruptions, out-of-distribution inputs, adaptation, and uncertainty.
9. **Efficient inference** — profiling, batching, quantization, pruning, compilation, and edge constraints.
10. **Production operations** — serving, observability, drift detection, rollback, and incident response.

## Enterprise

1. **Synthetic data systems** — generation, simulation, provenance, realism gaps, contamination, and validation.
2. **Edge vision platforms** — device constraints, fleet identity, secure updates, offline operation, and hardware lifecycle.
3. **Evaluation and observability** — quality slices, calibration, drift, traces, alerts, release gates, and incident response.
4. **Security and robustness** — adversarial inputs, physical attacks, supply chain, model extraction, and recovery.
5. **Responsible vision systems** — privacy, biometric risk, consent, accessibility, and human oversight.
6. **Governance and assurance** — lineage, documentation, auditability, vendor review, and regulatory controls.

## Definition of done for a lesson

A lesson is complete only when its README, self-contained notebook, local assets, checkpoint, tests, links, and Hub registry entry all pass automated validation. The notebook must execute without credentials and include its teaching code directly, a baseline, at least two experiments, evaluation, a failure injection, mitigation, and production guidance.
