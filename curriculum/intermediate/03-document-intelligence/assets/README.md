# Course 03 diagrams

Each SVG is generated deterministically from the coordinate-based JSON specification in [`specs/`](specs/). The assets have an opaque background, accessible title/description metadata, and repository-relative paths for GitHub and notebook rendering.

- `document-intelligence-pipeline.svg` — structured transformation with provenance preserved.
- `document-types.svg` — digital, scanned, hybrid, and raster routing.
- `ocr-pipeline.svg` — detection and recognition as separate contracts.
- `reading-order.svg` — naive interleaving versus column-aware order.
- `layout-taxonomy.svg` — common page region types.
- `table-structure.svg` — rows, columns, cells, and merged spans.
- `document-evidence-graph.svg` — field-to-document lineage.
- `enterprise-document-architecture.svg` — hostile intake through review and bounded output.

Regenerate and validate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/intermediate/03-document-intelligence/assets/specs/*.json
```
