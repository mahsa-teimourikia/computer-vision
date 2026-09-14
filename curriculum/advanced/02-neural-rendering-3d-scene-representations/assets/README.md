# Diagram assets

Every SVG is deterministically rendered from the matching coordinate specification in `specs/`. Assets use an opaque light/dark-safe canvas and include an accessible title and description.

Reproduce and validate the diagrams from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/advanced/02-neural-rendering-3d-scene-representations/assets/specs/*.json
```

- `representation-taxonomy.svg` — explicit, implicit, and hybrid learned scenes.
- `differentiable-rendering-loop.svg` — renderer, observation, loss, gradient, and update.
- `volume-rendering-ray.svg` — ordered samples, transmittance, weights, RGB, opacity, and expected ray distance.
- `camera-split.svg` — camera-level train/development/test roles and interpolation/extrapolation.
- `geometry-appearance-disagreement.svg` — matched RGB with incompatible density-derived ray distance.
- `nerf-vs-gaussian.svg` — field sampling versus Gaussian projection and shared compositing.
- `gaussian-projection.svg` — 3D covariance transformed into a screen-space ellipse.
- `gaussian-density-control.svg` — clone/split topology, prune, a conceptually separate optional opacity reset, and lineage; the notebook labels split-child opacity initialization independently.
