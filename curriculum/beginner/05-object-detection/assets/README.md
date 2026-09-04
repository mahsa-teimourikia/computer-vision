# Course 05 diagrams

These SVGs are rendered deterministically from coordinate specifications in [`specs/`](specs/):

- `detection-output-contract.svg`
- `anchor-vs-anchor-free.svg`
- `feature-pyramid.svg`
- `nms-duplicate-removal.svg`
- `dense-vs-set-prediction.svg`
- `detr-matching.svg`
- `detection-error-taxonomy.svg`
- `closed-vs-open-vocabulary.svg`

Re-render them from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/beginner/05-object-detection/assets/specs/*.json
```

Each asset has an opaque background, SVG title and description, explicit groups/nodes/ports/routes, and a reproducible source specification.
