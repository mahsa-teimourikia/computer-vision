# Course 02 diagrams

Each SVG in this directory is generated deterministically from the coordinate-based JSON specification in [`specs/`](specs/). The assets use an opaque background, accessible title/description metadata, and repository-relative paths so they render in GitHub, the chapter, and the notebook.

- `checked-reasoning-pipeline.svg` — observable artifacts from question through checked conclusion.
- `reasoning-dependency-dag.svg` — fact dependencies for the maintenance rule.
- `evidence-contracts.svg` — capability-specific evidence requirements.
- `deterministic-tool-boundary.svg` — semantic perception versus exact computation.
- `counterfactual-verification.svg` — relevant sensitivity and irrelevant invariance.
- `multi-image-binding.svg` — source-bound before/after facts.
- `reasoning-failure-taxonomy.svg` — earliest-boundary failure attribution.
- `enterprise-reasoning-architecture.svg` — governed inference, verification, review, and authorization separation.

Regenerate and validate this course's diagrams from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/intermediate/02-multimodal-reasoning-verification/assets/specs/*.json
```
