# Computer vision tooling review

> Reviewed: 2026-09-01. Recheck releases, hardware support, model licenses, and project health before standardizing a production stack.

Tools are selected per lesson, not imposed as one universal framework. Every tooling decision should compare maintenance, portability, observability, licensing, reproducibility, hardware fit, exportability, and operational cost.

## Default learning stack

| Layer | Default | Why it is the teaching default | When to choose something else |
| --- | --- | --- | --- |
| Arrays and baselines | [NumPy](https://numpy.org/doc/stable/) + [Pillow](https://pillow.readthedocs.io/) | Makes dtype, range, channels, and transforms visible | Use OpenCV for optimized classical pipelines and camera/video I/O |
| Training | [PyTorch](https://pytorch.org/docs/stable/) + [torchvision](https://docs.pytorch.org/vision/stable/) | Broad research adoption, eager debugging, strong model and transform ecosystem | Use JAX when accelerator research or functional transformations are central |
| Reusable backbones | [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) and [timm](https://huggingface.co/docs/timm/index) | Accessible model cards, pretrained weights, and common adaptation APIs | Use the authors' repository when reproduction requires nonstandard operators or preprocessing |
| Experiment records | Local JSON/CSV fixtures first | Credential-free and inspectable in course notebooks | Add MLflow or Weights & Biases when distributed runs, artifact lineage, and team workflows justify a service |

The course teaches the primitive before the framework. A framework example must expose preprocessing, tensor shapes, device placement, loss, metrics, and failure behavior instead of hiding them behind a one-call demo.

## Task frameworks

| Tool | Best fit | Strengths | Constraints to review |
| --- | --- | --- | --- |
| [PyTorch + torchvision detection](https://docs.pytorch.org/vision/stable/models.html#object-detection) | Transparent training/evaluation primitives and maintained Faster R-CNN, RetinaNet, FCOS, and SSD baselines | Tensor-level box, IoU, focal-loss, NMS, model, and weights APIs | Beta/evolving detector APIs, preprocessing/weight-enum parity, export support, and target-hardware profiling |
| [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | Hungarian assignment experiments for set prediction | Trusted rectangular linear-sum assignment primitive | Cost construction, normalization, device transfer, and scaling remain explicit design choices |
| [OpenMMLab](https://openmmlab.com/) (`MMDetection`, `MMSegmentation`, `MMTracking`, `MMDeploy`) | Reproducible task research and architecture comparison | Large configuration/model ecosystem; consistent task runners | Configuration complexity, cross-package version compatibility, and deployment operator support |
| [Detectron2](https://detectron2.readthedocs.io/) | Detection and segmentation research | Strong reference implementations and extensible components | Confirm maintenance cadence and platform compatibility for a new production commitment |
| [Ultralytics](https://docs.ultralytics.com/) | YOLO26/YOLOE-26 detection, segmentation, pose, tracking, and export prototypes | Low-friction train/predict/export workflow, including documented end-to-end and open-vocabulary paths | AGPL-3.0 or enterprise licensing, abstraction boundaries, checkpoint/export versions, and benchmark comparability |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/tasks/object_detection) | DETR/RT-DETR and Grounding DINO-style model/processor workflows | Model cards, processors, checkpoints, and interoperable training APIs | Immutable revisions, remote-code trust, checkpoint license, preprocessing parity, and rapidly changing model APIs |

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

Every spatial lesson must state coordinate frames, handedness, units, camera model, calibration assumptions, occlusion behavior, and geometric evaluation—not only visual quality.

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
