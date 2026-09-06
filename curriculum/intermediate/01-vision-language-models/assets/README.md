# Course 01 diagrams

Each SVG in this directory is generated deterministically from the coordinate-based JSON specification in [`specs/`](specs/). The assets use an opaque background, accessible title/description metadata, and repository-relative paths so they render in both the chapter and notebook.

- `dual-encoder-vs-generative-vlm.svg` — compatibility scoring versus conditional language generation.
- `visual-token-interfaces.svg` — pooled, full-patch, selected, and compressed interfaces.
- `connector-patterns.svg` — projection, query resampling, and cross-attention.
- `fusion-taxonomy.svg` — late through deeper fusion choices.
- `multimodal-training-stages.svg` — initialization through governed adaptation.
- `resolution-token-budget.svg` — detail, tiling, token, and systems trade-offs.
- `visual-evidence-ablation.svg` — correct, blank, wrong, and counterfactual inputs.
- `vlm-evaluation-contract.svg` — separate answer, evidence, robustness, systems, and governance gates.

Regenerate and validate this course's diagrams from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/intermediate/01-vision-language-models/assets/specs/*.json
```
