# Advanced 03 visual assets

These course-owned, accessible SVGs are rendered deterministically from the coordinate specifications in `specs/`.

| Asset | Teaching purpose |
| --- | --- |
| `world-model-loop.svg` | Separate observations, persistent state, action-conditioned dynamics, and decoded futures. |
| `observation-vs-state.svg` | Show why visible pixels are not a complete Markov state. |
| `world-model-taxonomy.svg` | Compare latent, pixel/video, interactive, 4D, and object-centric families. |
| `dynamic-scene-representations.svg` | Compare per-frame, scene-flow, canonical-deformation, dynamic-Gaussian, and object-state representations. |
| `rollout-error.svg` | Explain teacher-forcing mismatch and compounding open-loop error. |
| `counterfactual-actions.svg` | Hold initial state fixed while actions produce different futures. |
| `world-model-planning.svg` | Keep world model, scorer, policy choice, and true execution distinct. |
| `model-exploitation.svg` | Trace how a model blind spot becomes an optimizer-selected physical failure. |

Regenerate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/advanced/03-dynamic-scenes-world-models/assets/specs/*.json
```
