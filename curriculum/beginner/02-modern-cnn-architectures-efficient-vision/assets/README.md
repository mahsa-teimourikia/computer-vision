# Course 02 diagrams

These reusable conceptual diagrams complement the notebook's runtime-generated measurement plots:

- [`cnn-family-evolution.svg`](cnn-family-evolution.svg) — architecture families as design philosophies rather than a universal ranking;
- [`residual-block.svg`](residual-block.svg) — identity versus projection shortcuts;
- [`efficient-convolution.svg`](efficient-convolution.svg) — dense, grouped, depthwise, and pointwise channel connectivity;
- [`systems-metrics.svg`](systems-metrics.svg) — static, analytic, and measured deployment evidence; and
- [`pareto-model-selection.svg`](pareto-model-selection.svg) — dominated versus non-dominated model choices.

Each SVG has a versioned coordinate specification in [`specs/`](specs/), an opaque accessible background, a title and description, declared ports, and deterministic routing. Reproduce and validate them from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/beginner/02-modern-cnn-architectures-efficient-vision/assets/specs/*.json
```

The notebook still generates all empirical plots at runtime and writes decision artifacts to its ignored `.artifacts/architecture_benchmark/` directory. This prevents illustrative static diagrams from being mistaken for measurements from the learner's hardware.
