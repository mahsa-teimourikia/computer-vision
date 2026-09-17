# Advanced 05 diagram assets

These SVGs are rendered deterministically from the coordinate specifications in `specs/`:

- `spatial-memory-loop.svg` — observation, validation, memory, query, planning, and feedback;
- `map-representation-taxonomy.svg` — metric, topological, semantic, object, and graph views;
- `occupancy-evidence.svg` — free, occupied, and unknown sensor evidence;
- `association-errors.svg` — duplicate versus merged identities;
- `object-memory-lifecycle.svg` — observed, remembered, uncertain, moved, and retired states;
- `scene-graph-provenance.svg` — candidate, provenance, relation validation, and accepted edges;
- `hierarchical-navigation.svg` — goal, place route, local path, and replanning; and
- `plan-invalidation.svg` — trusted memory update, versioning, dependency check, and invalidation.

Run `python scripts/render_course_diagrams.py curriculum/advanced/05-spatial-memory-scene-graphs-navigation/assets/specs/*.json` from the repository root after editing a specification.
