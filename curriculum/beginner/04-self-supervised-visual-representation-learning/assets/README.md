# Course 04 visual assets

The SVGs in this directory are deterministic teaching diagrams. Each rendered asset has a versioned coordinate specification in `specs/`, an opaque background, accessible title and description, declared ports, and validated connector geometry.

Regenerate and validate them from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/beginner/04-self-supervised-visual-representation-learning/assets/specs/*.json
```

| Asset | Teaching purpose |
| --- | --- |
| `architecture-objective-shift.svg` | Separate information-flow architecture from the representation-learning objective |
| `ssl-paradigms.svg` | Organize SSL by learning paradigm rather than model name |
| `contrastive-learning.svg` | Trace two-view contrastive learning and NT-Xent |
| `teacher-student-learning.svg` | Explain student gradients, stop-gradient teacher targets, and EMA |
| `masked-image-modeling.svg` | Trace visible patches through an asymmetric masked autoencoder |
| `representation-evaluation.svg` | Connect frozen global and patch features to downstream evidence |
