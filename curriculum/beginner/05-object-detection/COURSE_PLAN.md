# Course 05 implementation plan

## Learning arc

`output contract → boxes and annotation → IoU → matching and AP → dense assignment → pyramids and losses → NMS → one-stage/two-stage systems → DETR set prediction → open vocabulary → source-aware enterprise decision`

The chapter will organize detector families around representation, localization, scale, assignment, classification, duplicate handling, loss design, and evaluation. It will not catalogue model releases.

## Scenario and boundaries

An industrial assembly line must detect multiple housings, fasteners, and contamination regions per image. Factories A and B form development data; Factory C is held out to expose source shift. Procedural images keep the default notebook redistributable, credential-free, deterministic, and CPU safe. Results are teaching evidence, not a safety certification.

## Notebook vertical slice

1. Define and validate `xyxy`, `xywh`, normalized, COCO-like, VOC-like, and YOLO-like contracts.
2. Implement box conversion, IoU, greedy detection matching, precision/recall, interpolated AP, and class/size slices.
3. Generate the source-aware multi-object corpus and visualize annotations.
4. Build anchor candidates, inspect assignment coverage, and compare anchor-based with anchor-free responsibility.
5. Train a tiny anchor-free grid detector; expose objectness, class, and localization losses.
6. Implement class-aware NMS and verify it against `torchvision.ops.nms`.
7. Sweep confidence and NMS thresholds; compare clean and held-out-source AP, recall, duplicates, latency, and small-object behavior.
8. Build a Hungarian cost matrix with `scipy.optimize.linear_sum_assignment` and contrast dense/NMS inference with set prediction.
9. Review guarded TorchVision, Transformers, Ultralytics, and Grounding DINO integration paths with version, checkpoint, remote-code, and license controls.
10. Save raw detections, evaluation tables, threshold experiments, source/size slices, and an enterprise detector decision artifact.

## Technology choices

- PyTorch and torchvision for the executable model, image tensors, drawing, and reference box/NMS operations.
- NumPy, pandas, Pillow, Matplotlib, and SciPy for transparent geometry, tables, visuals, and Hungarian matching.
- No default Ultralytics dependency: the current ecosystem is reviewed and an opt-in adapter is shown, but AGPL-3.0/enterprise licensing is a deployment gate.
- No default network downloads. Maintained pretrained detectors and open-vocabulary models are optional extensions, not evidence in the local benchmark.

## Diagrams

- `detection-output-contract.svg`
- `anchor-vs-anchor-free.svg`
- `feature-pyramid.svg`
- `nms-duplicate-removal.svg`
- `dense-vs-set-prediction.svg`
- `closed-vs-open-vocabulary.svg`

Every diagram will retain its validated coordinate specification under `assets/specs/` and be rendered deterministically.

## Completion gates

- The notebook executes top to bottom on CPU without credentials or network access.
- Manual box, IoU, NMS, AP, and matching primitives are assertion-backed.
- At least one failure is injected and mitigated.
- Evaluation is sliced by source and object size and includes confidence/NMS sensitivity.
- The chapter distinguishes local measurements, official benchmark reports, and unresolved production evidence.
- Root navigation, the Learning Hub, focused checkpoint, tests, and Pages deployment are updated together.
