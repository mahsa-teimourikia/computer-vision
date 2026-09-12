# Diagram assets

Each SVG is deterministically rendered from the matching coordinate specification in `specs/`. Every asset includes an accessible title and description and uses an opaque, light/dark-safe canvas.

Reproduce and validate every diagram from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/advanced/01-3d-vision-spatial-intelligence/assets/specs/*.json
```

- `coordinate-frames.svg` — explicit world-to-camera-to-image-to-pixel transforms.
- `pinhole-projection.svg` — perspective division and intrinsic mapping.
- `epipolar-geometry.svg` — two-view epipolar plane and line constraint.
- `stereo-depth.svg` — rectified disparity to depth.
- `triangulation-uncertainty.svg` — strong versus weak ray intersection geometry.
- `sfm-pipeline.svg` — verified structure-from-motion and refinement stages.
- `representation-landscape.svg` — query-driven 3D representation choices.
- `spatial-evidence-pipeline.svg` — frame-aware, uncertainty-bound metric decisions.
