# Computer vision tooling review

> Reviewed: 2026-09-09. Recheck releases, hardware support, model licenses, and project health before standardizing a production stack.

Tools are selected per lesson, not imposed as one universal framework. Every tooling decision should compare maintenance, portability, observability, licensing, reproducibility, hardware fit, exportability, and operational cost.

## Default learning stack

| Layer | Default | Why it is the teaching default | When to choose something else |
| --- | --- | --- | --- |
| Arrays and baselines | [NumPy](https://numpy.org/doc/stable/) + [Pillow](https://pillow.readthedocs.io/) | Makes dtype, range, channels, and transforms visible | Use OpenCV for optimized classical pipelines and camera/video I/O |
| Training | [PyTorch](https://pytorch.org/docs/stable/) + [torchvision](https://docs.pytorch.org/vision/stable/) | Broad research adoption, eager debugging, strong model and transform ecosystem | Use JAX when accelerator research or functional transformations are central |
| Reusable backbones | [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) and [timm](https://huggingface.co/docs/timm/index) | Accessible model cards, pretrained weights, and common adaptation APIs | Use the authors' repository when reproduction requires nonstandard operators or preprocessing |
| Experiment records | Local JSON/CSV fixtures first | Credential-free and inspectable in course notebooks | Add MLflow or Weights & Biases when distributed runs, artifact lineage, and team workflows justify a service |

The course teaches the primitive before the framework. A framework example must expose preprocessing, tensor shapes, device placement, loss, metrics, and failure behavior instead of hiding them behind a one-call demo.

## Vision foundation and open-vocabulary systems

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) | Common processor/model interfaces for CLIP, SigLIP2, DINOv2, and Grounding DINO | Immutable model revisions, model cards, cached artifacts, familiar PyTorch inference | Processor drift, cache/network behavior, checkpoint licenses, remote-code trust, and rapidly changing APIs |
| [OpenAI CLIP](https://github.com/openai/CLIP) / [OpenCLIP](https://github.com/mlfoundations/open_clip) | Dual-encoder reproduction and broad image–text checkpoint comparison | Transparent normalized similarity and widely adopted zero-shot workflow | Training-data provenance, prompt/vocabulary sensitivity, calibration, and checkpoint-specific terms |
| [Big Vision](https://github.com/google-research/big_vision) / SigLIP2 | Author reference for sigmoid image–text alignment and current multilingual/dense variants | Research configurations and official checkpoints | JAX stack, preprocessing, artifact storage, license, and benchmark parity with PyTorch paths |
| [DINOv2](https://github.com/facebookresearch/dinov2) / [DINOv3](https://github.com/facebookresearch/dinov3) | Global and patch-level self-supervised foundation features | Strong reusable dense representations and author implementations | Pin code and weights separately; DINOv3 uses gated weights and a custom license |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Language-conditioned boxes and phrase grounding | Influential open-vocabulary DETR-style reference with Apache-2.0 code | Token/phrase semantics, native dependencies, thresholds, source age, checkpoint/data provenance |
| [Meta SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3) | Concept-, exemplar-, and visual-prompt detection, segmentation, and tracking | Unified current promptable interface across images and video | Gated 848M-parameter weights, custom SAM License, CUDA-oriented stack, and prompt/domain evaluation |
| [PEFT](https://huggingface.co/docs/peft/index) | Adapters and LoRA where a vision architecture is supported | Small trainable artifacts and common configuration patterns | Target-module selection, base/adapter compatibility, export, drift, and security review |

Course 09 uses transparent local proxies for its credential-free default experiments and marks them `foundation_model=false`. Optional official observations are isolated, revision-pinned, disabled by default, and stored separately from local evidence. No local proxy result is a claim about CLIP, SigLIP2, DINO, Grounding DINO, or SAM.

## Vision-language model systems

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) | Common processors, multimodal chat templates, generation, and adaptation | Broad model-family support through familiar PyTorch interfaces | Processor/chat-template semantics, immutable revisions, remote code, artifact hashes, checkpoint licenses, and model-specific input limits |
| [PEFT](https://huggingface.co/docs/peft/index) / [TRL](https://huggingface.co/docs/trl/index) | LoRA/adapters and multimodal post-training | Standardized trainable-module and trainer components | Vision versus language targets, packing, base–adapter compatibility, regression risk, export, and data governance |
| [vLLM](https://docs.vllm.ai/) | High-throughput serving for supported multimodal models | Continuous batching, paged caches, and common API surfaces | Supported modality/model matrix, image limits, processor parity, cache isolation, scheduling, and release-specific behavior |
| [SGLang](https://docs.sglang.ai/) | Structured generation and high-performance multimodal serving | Constrained outputs, prefix reuse, and serving-oriented programs | Backend/model maturity, constraint semantics, deployment complexity, and observability |
| [llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/tree/master/docs/multimodal) / MLX-VLM | Local, quantized, and edge experiments | Device-oriented runtimes and offline inference paths | Conversion provenance, projector/processor compatibility, unsupported operators, quantization drift, and quality parity |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) / [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | Broad multimodal evaluation harnesses | Task adapters, repeatable prompts, and model integrations | Benchmark contamination, dataset terms, judge dependence, prompt/version drift, and whether aggregate scores match the deployment task |
| [FiftyOne](https://docs.voxel51.com/) / [CVAT](https://docs.cvat.ai/docs/) / [Label Studio](https://labelstud.io/guide/) | Visual failure review and evidence adjudication | Slice inspection, region review, correction, and team workflows | Access control, retention, ontology/version identity, annotator agreement, and export validation |

Intermediate 01 implements projection, resampling, cross-attention, causal generation, and evidence scoring directly in PyTorch before showing a disabled Transformers adapter using `AutoProcessor`, the model's multimodal chat template, `AutoModelForImageTextToText`, and `generate`. Optional observations are revision-pinned and explicitly ineligible for model comparison until artifact hashes, licenses, processor files, runtime, and approvals are complete. Capability metrics, visual-evidence ablations, structured-output validation, and target-hardware profiling remain required regardless of serving framework.

Intermediate 02 keeps the reasoning executor framework-free so learners can inspect topological ordering, four-valued fact propagation, allow-listed tool contracts, evidence binding, deterministic replay, contradiction checks, counterfactuals, and review policy directly. NumPy supplies exact local geometry and arithmetic, pandas supplies event/slice analysis, and Pillow/Matplotlib render evidence. Pydantic or JSON Schema, NetworkX, OpenTelemetry, and a pinned Transformers model verifier are reviewed as production extensions, but schema validity, graph libraries, traces, and model agreement are never treated as factual verification or action authority.

Intermediate 03 uses Pillow, NumPy, pandas, and Matplotlib for a fully observable synthetic document pipeline: page rendering, coordinate transforms, OCR error injection, reading-order graphs, span-aware tables, field binding, versioned normalization, provenance replay, template shift, and perturbation slices. Tesseract 5.5.3 is the optional local OCR baseline; Microsoft Table Transformer detection/structure checkpoints and PaddleOCR-VL-1.6 are immutable-revision, disabled-by-default comparisons. PyMuPDF is reviewed but not imposed because its AGPL/commercial licensing needs deployment-specific approval. Optional results remain separate from local proxy evidence until model, processor, license, artifact hashes, source data, and target runtime are recorded.

Intermediate 04 implements BM25, semantic and visual feature proxies, structured retrieval, late interaction, reciprocal-rank fusion, deterministic reranking, canonical evidence assembly, retrieval metrics, ACL/freshness enforcement, bounded generation, and claim-level citation verification directly with standard Python, NumPy, pandas, Pillow, and Matplotlib. FAISS 1.15.0 remains course-local and optional; BGE-M3, SigLIP 2, BGE reranker v2 M3, ColQwen2.5/ColPali, and a non-authoritative Qwen3-VL relevance scorer are immutable-revision, disabled-by-default comparisons. A vector store or model is not considered ready until filter semantics, tenant isolation, updates/deletion, code/model/processor versions, artifact hashes, license, target runtime, and source-held-out evaluation are recorded.

## Video-language systems

| Tool or model | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| [TorchCodec](https://docs.pytorch.org/torchcodec/stable/) | Timestamp-aware audio/video decoding into PyTorch tensors | Official PyTorch-domain decoder with frame, clip, metadata, and presentation-time interfaces | FFmpeg compatibility, seek accuracy, variable-frame-rate behavior, corrupted media, memory, and target-platform packaging |
| [VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae) | Masked video representation and action-classification study | Established video self-supervision design with a common Transformers processor/model path | Clip sampling and label taxonomy dominate validity; the referenced Kinetics checkpoint is non-commercial and must not become a production default |
| [PE-AV](https://huggingface.co/facebook/pe-av-large) | Joint audio–video–text retrieval and representation | One shared embedding space with an official Transformers interface and Apache-2.0 model-card license | Two-billion-parameter operations, media preprocessing, audio/video synchronization, artifact identity, and source-domain evaluation |
| [Qwen3-VL](https://arxiv.org/abs/2511.21631) | Current video-conditioned multimodal generation experiments | Strong long-context and temporal modeling research direction with common processor/generation interfaces | Token cost, sampling sensitivity, generated-claim verification, model-card terms, serving support, and rapidly changing releases |
| [InternVideo2.5](https://arxiv.org/abs/2501.12386) | Long-video multimodal research comparison | Hierarchical memory framing for long contexts and detailed temporal understanding | Research implementation maturity, checkpoint/runtime requirements, reproducibility, and deployment license |
| [Video-MME](https://arxiv.org/abs/2405.21075), [LongVideoBench](https://arxiv.org/abs/2407.15754), [EgoSchema](https://arxiv.org/abs/2308.09126) | External video-understanding evaluation context | Complementary duration, long-context, and temporal-reasoning task designs | Contamination, judge dependence, subtitle/audio conditions, answer format, licensing, and mismatch with the target workflow |

Intermediate 05 keeps decoding and learned inference out of its credential-free default so timing, sampling, interval, retrieval, citation, counterfactual, and streaming contracts remain inspectable. Its `local_temporal_representation_proxy` and `local_video_language_proxy` are explicitly not foundation models or quality benchmarks. Optional PE-AV, VideoMAE, and Qwen3-VL adapters are revision-pinned and disabled; production upgrades should use an approved decoder, preserve presentation timestamps and clock uncertainty, authorize before retrieval, and separately benchmark decode, sampling, encoding, indexing, localization, generation, and verification on target hardware.

## Visual-agent systems

| Tool or pattern | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| Direct typed loop | Contract-first visual-agent development and evaluation | Maximum visibility into state, policy, validation, budgets, traces, and stopping | Durability, concurrency, distributed execution, and migrations remain application work |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) | Stateful graph workflows with loops, checkpoints, and human interrupts | Explicit state, nodes, edges, conditional routing, persistence ecosystem, and replay patterns | Graph shape does not define resource authorization, evidence validity, safe side effects, or business approval |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Provider-native tools, orchestration, sessions, handoffs, tracing, and guardrail hooks | Maintained Python/TypeScript runtimes and integration with Responses API capabilities | Keep credentials optional; validate schemas/results; apply independent tenant/resource policy and approval |
| [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) | Interoperable discovery and invocation through tool gateways | Tool schemas plus standardized HTTP authorization and protected-resource discovery | Transport identity/scopes do not replace per-operation capability, resource, result, and business-policy checks |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da) | Optional open multimodal proposal generation | Common Transformers path, image/video input, structured generation, Apache-2.0 official repository/model-card metadata | Pin model revision, keep `trust_remote_code=False`, allow-list tools, evaluate injection/selection/stopping, and never grant direct execution |

Intermediate 06 implements the typed state machine, nine narrow local tools, capability and resource policy, semantic input checks, output postconditions, evidence promotion, retries, budgets, loop detection, counterfactuals, security tests, and Site C reporting directly with standard Python, NumPy, pandas, Pillow, and Matplotlib. The default `local_planner_proxy` is not a foundation-agent benchmark. Optional Qwen3-VL, LangGraph, OpenAI Agents SDK, and MCP mappings are disabled; their abstractions do not replace the executor's authority boundary.

## Task frameworks

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| [PyTorch + torchvision detection](https://docs.pytorch.org/vision/stable/models.html#object-detection) | Transparent training/evaluation primitives and maintained Faster R-CNN, RetinaNet, FCOS, and SSD baselines | Tensor-level box, IoU, focal-loss, NMS, model, and weights APIs | Beta/evolving detector APIs, preprocessing/weight-enum parity, export support, and target-hardware profiling |
| [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | Hungarian assignment experiments for set prediction | Trusted rectangular linear-sum assignment primitive | Cost construction, normalization, device transfer, and scaling remain explicit design choices |
| [OpenMMLab](https://openmmlab.com/) (`MMDetection`, `MMSegmentation`, `MMTracking`, `MMDeploy`) | Reproducible task research and architecture comparison | Large configuration/model ecosystem; consistent task runners | Configuration complexity, cross-package version compatibility, and deployment operator support |
| [Detectron2](https://detectron2.readthedocs.io/) | Detection and segmentation research | Strong reference implementations and extensible components | Confirm maintenance cadence and platform compatibility for a new production commitment |
| [Ultralytics](https://docs.ultralytics.com/) | YOLO26/YOLOE-26 detection, segmentation, pose, tracking, and export prototypes | Low-friction train/predict/export workflow, including documented end-to-end and open-vocabulary paths | AGPL-3.0 or enterprise licensing, abstraction boundaries, checkpoint/export versions, and benchmark comparability |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/tasks/object_detection) | DETR/RT-DETR and Grounding DINO-style model/processor workflows | Model cards, processors, checkpoints, and interoperable training APIs | Immutable revisions, remote-code trust, checkpoint license, preprocessing parity, and rapidly changing model APIs |
| [PyTorch + torchvision segmentation](https://docs.pytorch.org/vision/stable/models.html#semantic-segmentation) | Transparent semantic and instance segmentation baselines | Maintained FCN, DeepLabV3, LRASPP, and Mask R-CNN model/weight contracts plus tensor-level losses | Label interpolation, ignore IDs, output stride, preprocessing/weight parity, beta APIs, and target-hardware profiling |
| [segmentation-models-pytorch](https://smp.readthedocs.io/) | Rapid task-specific encoder–decoder experiments | Common U-Net-family architectures, broad encoder catalogue, and composable losses | Encoder weight provenance, package compatibility, abstraction depth, export operators, and benchmark parity |
| [Meta SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3) | Current promptable concept segmentation and tracking research | Text, exemplar, point, box, and mask prompting with a shared image/video foundation architecture | Gated 848M-parameter checkpoint, custom SAM License, CUDA-oriented stack, immutable revision/checkpoint hashes, prompt provenance, and domain evaluation |
| [Meta SAM 2.1](https://github.com/facebookresearch/sam2) | Permissively licensed visual-prompt image/video segmentation | Smaller official checkpoints and Apache-2.0 code/weights with interactive image/video APIs | It is not the newest concept-prompted generation; still review compute, checkpoint hash, data, and domain transfer |

Do not compare frameworks using headline metrics copied from model pages. Re-run the same dataset split, preprocessing, resolution, precision, hardware, and metric implementation.

## Self-supervised learning and embedding systems

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| PyTorch + torchvision | Transparent objective and evaluation primitives | Autograd, common transforms, official supervised baselines, distributed building blocks | Augmentation validity, global-batch semantics, collapse monitoring, and probe design remain explicit responsibilities |
| Official DINO/DINOv2/DINOv3 repositories | Paper reproduction and author-released checkpoints | Reference objectives, model families, pretrained global and dense features | Pin repository revision, checkpoint hash, preprocessing, source trust, model license, hardware, and nonstandard dependencies |
| Hugging Face Transformers | Model/processor-style foundation-feature reuse | DINOv2 APIs, hidden states, model cards, cached artifacts | Processor defaults, remote artifacts/code, access, licenses, and fast-moving APIs |
| `timm` | Broad backbone and pretraining research | Consistent model creation and extensive weight catalogue | Verify the exact recipe, upstream data, weight license, representation output, and normalization |
| lightly / solo-learn | Packaged SSL training and research recipes | Losses, heads, memory banks, teacher–student methods, distributed workflows | Framework abstractions can hide matrix contracts, distributed batch assumptions, and recipe coupling |
| FAISS | Large-scale embedding retrieval | Exact and approximate nearest-neighbour indexes with CPU/GPU options | Measure recall loss, filtering, updates, feature-version migration, tenant isolation, and memory |

Teach the primitive first: view generation, similarity matrix, target mapping, stop-gradient, EMA, masking, and downstream evaluation should remain inspectable before adopting a packaged SSL trainer.

## Metric learning and retrieval systems

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| PyTorch + torchvision | Siamese, contrastive, triplet, supervised-contrastive, and proxy-learning experiments | Transparent objective, sampler, mining, normalization, and official supervised encoder APIs | Batch composition, false negatives, numerical stability, distributed mining semantics, and evaluation contracts remain explicit responsibilities |
| [scikit-learn nearest neighbours](https://scikit-learn.org/stable/modules/neighbors.html) | Portable exact-search baselines and small corpora | Familiar API, auditable brute-force behavior, and useful parity checks | Not a large-scale vector-serving layer; confirm metric conventions, normalization, and memory scaling |
| [FAISS](https://github.com/facebookresearch/faiss) | Local exact and approximate dense-vector indexes | `IndexFlat` ground truth, HNSW, inverted files, product quantization, CPU/GPU options, and index serialization | Tune ANN against exact recall; review native-runtime compatibility, filtering, deletion/update behavior, memory overhead, ID mapping, and embedding migrations |
| [Qdrant](https://qdrant.tech/documentation/) | Governed vector service with payload filtering | Persistent collections, metadata filters, distributed deployment, and service observability | Adds an operating service; review tenancy, backups, consistency, network boundaries, cost, and client/server version compatibility |
| [FiftyOne](https://docs.voxel51.com/) | Visual retrieval inspection and review queues | Links embeddings, samples, labels, similarity, duplicates, and human review | Keep metrics and index artifacts portable; review data access, plugin/version governance, and scaling architecture |

The Course 07 default is an exact NumPy/scikit-learn baseline plus FAISS `IndexFlatIP` and HNSW on L2-normalized `float32` vectors. Because FAISS introduces a native runtime, it remains in Course 07's tested requirements and the contributor/CI environment rather than the repository-wide learner extra used for Courses 01–06. A hosted vector database is deliberately optional: production adoption requires metadata filtering, access control, backup/restore, observability, update semantics, and migration evidence—not only a nearest-neighbour demo.

## Tracking, keypoint, and pose systems

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| NumPy + [SciPy assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | Transparent tracking-by-detection teaching and small controlled systems | Inspectable IoU, motion, appearance, gates, Hungarian assignment, lifecycle, and failure attribution | The developer owns state estimation, metric semantics, performance, camera-motion handling, and every operating policy |
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) | Reference HOTA, CLEAR, and identity evaluation | Official HOTA implementation and MOTChallenge evaluation kit | Pin the source revision; preserve exact dataset formatting, ignore/crowd rules, thresholds, and sequence aggregation; dataset licenses are separate from MIT code |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | Detection-based MOT where low-confidence recovery is useful | Clear two-stage association idea and MIT reference code | Detector/checkpoint coupling, source age, dependency compatibility, thresholds, identity evidence, and production packaging |
| [OC-SORT](https://github.com/noahcao/OC_SORT) / [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | Stronger motion handling or camera-motion/appearance integration | Representative observation-centric and multi-cue trackers | Research-code maintenance, licenses, detector/re-ID checkpoints, camera assumptions, dependency conflicts, and export path |
| [MMPose](https://github.com/open-mmlab/mmpose) / RTMPose | Broad pose research, model zoo, and deployment comparison | Human, animal, hand, face, whole-body, and real-time model families; multiple export backends | Cross-package versions, config/checkpoint pinning, trained-weight/data license, preprocessing, compiled operators, and domain fit |
| [torchvision Keypoint R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html) | Common-SDK human-keypoint baseline | Familiar weights enum, transforms, tensors, and CPU smoke-test path | Beta detection APIs, COCO-human landmark contract, download size, latency, export behavior, and no applicability to arbitrary industrial landmarks |

Course 08 keeps the first implementation inside the notebook and uses variable timestamps, a separate detector stream, association/lifecycle sweeps, a ByteTrack-inspired teaching comparison, and explicit source-shift attribution. It exports MOTChallenge-style text for an optional pinned TrackEval run but never renames local teaching metrics as official HOTA or IDF1. ByteTrack, MMPose/RTMPose, and torchvision checkpoint execution stay disabled by default, so the CPU learner path has no new native or remote-model dependency.

## Data, annotation, and evaluation

| Tool | Role | Review notes |
| --- | --- | --- |
| [CVAT](https://docs.cvat.ai/docs/) | Image/video annotation and review | Strong task workflow and automation; design consensus, QA, identity, and export validation explicitly |
| [Label Studio](https://labelstud.io/guide/) | Flexible multimodal labeling | Useful when vision labels sit beside text/audio; verify template and export semantics |
| [FiftyOne](https://docs.voxel51.com/) | Dataset exploration, slices, embeddings, and error analysis | Especially useful for sample-level detection/segmentation analysis; keep metric definitions portable |
| [Albumentations](https://albumentations.ai/docs/) | Image augmentation | Broad transform set; test joint image/mask/box/keypoint geometry and seed reproducibility |
| [COCO API](https://github.com/cocodataset/cocoapi) | Reference task metrics and formats | Valuable compatibility baseline; document deviations and crowd/ignore handling |

Required data checks include provenance, license, consent and privacy basis, duplicates/leakage, label schema, annotator agreement, slices, camera/site/time independence, and transformations applied before evaluation.

## Spatial and 3D tooling

| Tool | Best fit | Constraints to review |
| --- | --- | --- |
| [Open3D](https://www.open3d.org/docs/latest/) | Point clouds, RGB-D, registration, reconstruction, and visualization | Coordinate conventions, units, device support, and tensor/legacy API differences |
| [COLMAP](https://colmap.github.io/) | Structure-from-motion and multi-view stereo baselines | Camera calibration, feature assumptions, compute cost, and sparse/dense failure diagnosis |
| [Nerfstudio](https://docs.nerf.studio/) | NeRF and Gaussian-splatting research workflows | Dataset conversion, camera poses, viewer/runtime dependencies, and export path |
| [gsplat](https://docs.gsplat.studio/) | Differentiable Gaussian splatting | CUDA/hardware fit, rasterizer compatibility, memory scaling, and benchmark protocol |
| [PyTorch3D](https://pytorch3d.org/docs/) | Differentiable 3D operators and rendering | Build compatibility, coordinate systems, and whether the required operator is actively supported |
| [OpenCV calib3d](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) | Calibration, epipolar geometry, PnP, stereo, and triangulation baselines | Camera/distortion model, pixel convention, robust-estimator defaults, degeneracy checks, and versioned calibration evidence |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | General relative depth and separately trained metric variants | Output contract, processor/interpolation parity, checkpoint-specific license, domain/edge evaluation, and immutable weights |
| [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) | Camera-aware monocular metric depth and point prediction | Intrinsics contract, uncertainty calibration, camera/domain shift, model license, and GPU/runtime fit |
| [DUSt3R / MASt3R / MUSt3R](https://github.com/naver/dust3r) | Learned point maps, dense correspondence, and feed-forward multi-view reconstruction research | Non-commercial checkpoint/code terms can apply; inspect alignment, scale, confidence, memory, and dependency coupling |
| [VGGT](https://github.com/facebookresearch/vggt) / [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) | Unified feed-forward cameras, depth, point maps, tracks, and static/dynamic reconstruction research | Checkpoint access and licenses differ; review view-count memory, CUDA requirements, source contamination/reproduction notes, and task-specific geometry |

Every spatial lesson must state coordinate frames, handedness, units, camera model, calibration assumptions, occlusion behavior, and geometric evaluation—not only visual quality.

Advanced 01 keeps the core path in NumPy/SciPy and treats OpenCV, COLMAP, Open3D, learned depth, and feed-forward reconstruction as optional layers. Its reviewed source revisions are recorded in the course constraints and notebook; those pins do not imply that all optional dependencies belong in one environment or that a code license grants rights to every checkpoint.

## Embodied and simulation tooling

| Tool | Best fit | Constraints to review |
| --- | --- | --- |
| [LeRobot](https://huggingface.co/docs/lerobot/) | Open robot-learning datasets, policies, and hardware integrations | Dataset/action schema, control rate, embodiment transfer, and model/hardware support |
| [MuJoCo](https://mujoco.readthedocs.io/) | Fast contact-rich physics and control research | Simulation-to-real gap, sensors, contact parameters, and task reproducibility |
| [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) / Isaac Lab | Photoreal simulation, synthetic data, and GPU robot learning | NVIDIA hardware/software dependency, licensing, determinism, and deployment separation |
| [ROS 2](https://docs.ros.org/en/rolling/) | Robot middleware and system integration | Real-time boundaries, message contracts, QoS, identity/security, and lifecycle management |

Physical-action labs default to simulation, bounded action spaces, explicit stop conditions, recorded state/action traces, and human approval before any real actuator path.

## Deployment and operations

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| [ONNX Runtime](https://onnxruntime.ai/docs/) | Cross-platform CPU/GPU/mobile inference | Multiple execution providers and a portable interchange path | Operator/export parity, dynamic shapes, preprocessing, and provider-specific behavior |
| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) / Torch-TensorRT | NVIDIA GPU and edge optimization | Mixed precision, graph optimization, low-latency runtimes | NVIDIA dependency, engine compatibility, unsupported operators, calibration, and numeric drift |
| [OpenVINO](https://docs.openvino.ai/) | Intel CPU/GPU/NPU deployment | Hardware-aware optimization and broad model conversion | Device/plugin support and conversion fidelity |
| [Core ML](https://apple.github.io/coremltools/docs-guides/) | Apple on-device deployment | Integration with Apple silicon and app tooling | OS/hardware targets, conversion support, and preprocessing parity |
| [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/) | Multi-model GPU serving | Dynamic batching, model repositories, metrics, and multiple backends | Operational weight, GPU economics, model scheduling, and tenant isolation |

Benchmark exported artifacts against the source model on a locked validation set. Record accuracy delta, warm and cold latency, throughput, memory, power where relevant, artifact size, startup time, hardware/software versions, and failure behavior.

## Tooling review required in every lesson

Each non-documentation lesson must answer:

1. What is the minimal primitive implementation?
2. Which maintained tools package that primitive, and at what abstraction level?
3. What are their license, model-weight, dataset, and remote-code implications?
4. What input/output contracts and preprocessing conventions differ?
5. Can the result be exported, observed, reproduced, and run on the target hardware?
6. What does the selected tool make easier, and what failure can it hide?
7. What evidence would justify switching tools?
