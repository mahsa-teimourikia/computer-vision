# Visual-agent course assets

Each SVG is rendered deterministically from the adjacent versioned JSON layout specification. The assets use an opaque neutral canvas, accessible title/description text, stable semantic colors, and repository-relative links suitable for GitHub and the notebook.

- `visual-agent-loop.svg` — the bounded observe/plan/authorize/execute/verify/update/stop lifecycle.
- `agent-state.svg` — facts, unknowns, evidence, results, budgets, and status in typed state.
- `tool-gateway.svg` — proposal flow through policy into the approved registry.
- `planning-vs-execution.svg` — proposal authority separated from execution authority.
- `verification-loop.svg` — result validation and evidence promotion.
- `agent-failure-taxonomy.svg` — failure attribution across the system.
- `cross-modal-agent.svg` — image, document, and video evidence for one decision.
- `human-approval-boundary.svg` — recommendation, approval, and external-action separation.

Regenerate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/intermediate/06-visual-agents/assets/specs/*.json
```
