# Advanced 04 visual assets

These course-owned, accessible SVGs are rendered deterministically from validated coordinate specifications in `specs/`.

| Asset | Teaching purpose |
| --- | --- |
| `embodied-closed-loop.svg` | Keep the entire observation-to-feedback contract visible. |
| `model-role-taxonomy.svg` | Distinguish interpretation, prediction, action selection, tracking, and multimodal action. |
| `embodiment-contract.svg` | Show every field needed before an action can be interpreted. |
| `grounding-affordance.svg` | Separate semantic binding from embodiment-conditioned interaction regions. |
| `action-space-taxonomy.svg` | Compare action representations and their missing contracts. |
| `action-chunking-feedback.svg` | Contrast stale open-loop execution with receding-horizon replanning. |
| `action-verification-gateway.svg` | Make proposal, validation, authorization, execution, and verification distinct. |
| `stale-observation.svg` | Expose the timing chain and the freshness gate. |

Regenerate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/advanced/04-embodied-vision-vla-models/assets/specs/*.json
```
