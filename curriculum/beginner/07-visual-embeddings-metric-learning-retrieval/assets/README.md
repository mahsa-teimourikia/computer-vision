# Course 07 visual assets

Every diagram is rendered deterministically from the matching coordinate specification in `specs/` by `scripts/render_course_diagrams.py`.

- `embedding-space.svg` — images, encoder, vectors, and task-defined neighbourhoods;
- `siamese-network.svg` — two inputs and shared-weight encoders;
- `triplet-learning.svg` — anchor, positive, negative, and margin geometry;
- `hard-negative-mining.svg` — embed, retrieve, review, and feedback loop;
- `retrieval-pipeline.svg` — query encoding, versioned index, filters, ranking, and results;
- `exact-vs-ann.svg` — exhaustive versus candidate-pruned search;
- `embedding-versioning.svg` — blue/green encoder and index migration; and
- `retrieval-failure-taxonomy.svg` — sample-level retrieval failure categories.

SVGs include a title, description, opaque background, and semantic alt text so they remain readable in GitHub, the notebook, and the Learning Hub.
