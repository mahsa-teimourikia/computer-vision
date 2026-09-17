# Course 04 diagrams

Each SVG is generated deterministically from the coordinate-based JSON specification in [`specs/`](specs/). Every asset has an opaque background, accessible title/description metadata, and repository-relative paths for GitHub and notebook rendering.

- `multimodal-rag-pipeline.svg` — authorized evidence retrieval through verified answer.
- `retrieval-units.svg` — hierarchy from document/page to cells and visual regions.
- `multi-index-retrieval.svg` — routed lexical, semantic, structured, and visual indexes.
- `hybrid-retrieval.svg` — rank fusion and reranking without raw-score confusion.
- `evidence-assembly.svg` — canonical deduplication, hierarchy restoration, and sufficiency.
- `citation-contract.svg` — atomic claims bound to supporting evidence.
- `acl-aware-retrieval.svg` — trusted prefiltering before any scorer.
- `rag-failure-taxonomy.svg` — earliest-boundary failure attribution.

Regenerate and validate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/intermediate/04-multimodal-retrieval-rag/assets/specs/*.json
```
