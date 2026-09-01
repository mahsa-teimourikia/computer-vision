# Diagram assets

The six course diagrams are deterministic, accessible SVGs generated from the coordinate-based JSON specifications in [`specs/`](specs/).

From the repository root, validate the geometry and regenerate every asset with:

```bash
python scripts/render_course_diagrams.py \
  curriculum/beginner/01-modern-computer-vision-foundations/assets/specs/*.json
```

The validator checks IDs, canvas containment, group containment, node clearance, port bindings, and edge endpoints. The rendered SVGs include an opaque background, title, description, and semantic color system. They were also rasterized and visually inspected at 1200 px.

