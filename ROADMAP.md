# Curriculum roadmap

The curriculum progresses from observable image mechanics to model evaluation and production operations. Topic numbers restart within each level.

## Beginner

1. **Image fundamentals** — arrays, channels, dtype, range, normalization, thresholding, and measurement. *(available)*
2. **Filtering and edges** — convolution, kernels, smoothing, gradients, and noise trade-offs.
3. **Features and matching** — corners, descriptors, correspondence, and geometric verification.
4. **First image classifier** — datasets, splits, baselines, training loops, and confusion matrices.

## Intermediate

1. **Convolutional networks** — receptive fields, feature hierarchies, architecture choices, and transfer learning.
2. **Object detection** — localization, IoU, non-maximum suppression, mAP, and error diagnosis.
3. **Semantic and instance segmentation** — masks, loss functions, class imbalance, and boundary metrics.
4. **Data quality and augmentation** — leakage, label noise, invariance, synthetic data, and dataset shift.
5. **Evaluation and explainability** — calibration, slices, attribution limits, robustness, and release gates.

## Advanced

1. **Video understanding and tracking** — temporal sampling, motion, identity association, and drift.
2. **Vision transformers** — tokenization, attention, scaling behavior, and hybrid architectures.
3. **Multimodal vision-language systems** — embeddings, grounding, retrieval, and evaluation.
4. **Robustness and domain adaptation** — corruptions, out-of-distribution inputs, adaptation, and uncertainty.
5. **Efficient inference** — profiling, batching, quantization, pruning, compilation, and edge constraints.
6. **Production operations** — serving, observability, drift detection, rollback, and incident response.

## Enterprise

1. **Responsible vision systems** — privacy, biometric risk, consent, accessibility, and human oversight.
2. **Governance and assurance** — lineage, documentation, auditability, vendor review, and regulatory controls.
3. **Fleet and platform operations** — device lifecycle, model registry, staged rollout, SLOs, and cost controls.

## Definition of done for a lesson

A lesson is complete only when its README, notebook, `lab.py`, local assets, checkpoint, tests, links, and Hub registry entry all pass automated validation. The notebook must execute without credentials and include a baseline, at least two experiments, evaluation, a failure injection, mitigation, and production guidance.
