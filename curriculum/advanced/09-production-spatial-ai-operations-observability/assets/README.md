# Advanced 09 diagram assets

These reusable SVG diagrams are rendered deterministically from the validated JSON specifications in [`specs/`](specs/). Each file includes an accessible title and description.

- `deployment-configuration-graph.svg` — complete deployment identity;
- `registry-lifecycle.svg` — separated registration and promotion authority;
- `time-valid-frame-graph.svg` — observation-time transforms and closure evidence;
- `capability-observability-graph.svg` — component-to-impact dependency;
- `drift-evidence-classes.svg` — immediate proxy versus delayed outcome evidence;
- `progressive-delivery-gates.svg` — shadow/canary evidence gates;
- `stateful-rollback.svg` — configuration and state compatibility; and
- `incident-recovery-loop.svg` — containment, diagnosis, verification, and audit.

Regenerate them with:

```bash
python scripts/render_course_diagrams.py curriculum/advanced/09-production-spatial-ai-operations-observability/assets/specs/*.json
```
