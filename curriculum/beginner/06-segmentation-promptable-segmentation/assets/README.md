# Course 06 diagrams

These SVGs are rendered deterministically from coordinate specifications in [`specs/`](specs/):

- `segmentation-taxonomy.svg`
- `unet-encoder-decoder.svg`
- `semantic-vs-instance.svg`
- `mask-metrics.svg`
- `mask-rcnn.svg`
- `query-mask-classification.svg`
- `promptable-segmentation.svg`
- `detector-segmenter-pipeline.svg`

Re-render them from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/beginner/06-segmentation-promptable-segmentation/assets/specs/*.json
```

Every asset has an opaque background, SVG title and description, explicit groups/nodes/ports/routes, and a reproducible source specification.
