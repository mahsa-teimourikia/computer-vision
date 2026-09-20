# Advanced track

Move from multimodal evidence systems into metric spatial representations, learned 3D priors, dynamic worlds, adaptation, and embodied perception. Advanced courses expose algorithms, conditioning, uncertainty, optimization, scale, and failure boundaries rather than treating 3D systems as viewers or model APIs.

## Available

- [01 · 3D Vision & Spatial Intelligence: From Camera Geometry to Metric Scene Understanding](01-3d-vision-spatial-intelligence/README.md) — declare frames and cameras; project and back-project; calibrate; verify correspondences; derive stereo depth; triangulate; estimate pose; reconstruct point clouds; propagate uncertainty; and make source-held-out metric decisions.
- [02 · Neural Rendering & 3D Scene Representations: From NeRFs to 3D Gaussian Splatting](02-neural-rendering-3d-scene-representations/README.md) — generate rays; derive and test volume rendering; split held-out cameras; separate appearance from geometry; project and splat anisotropic Gaussians; govern density control; and report source-held-out scene evidence.
- [03 · Dynamic Scenes & World Models: From 4D Scene State to Action-Conditioned Futures](03-dynamic-scenes-world-models/README.md) — separate observation from persistent state; fit action-conditioned dynamics; evaluate recursive and stochastic futures; preserve identity; check physical constraints; expose planner exploitation; and report source-held-out dynamics evidence.
- [04 · Embodied Vision & Vision-Language-Action Models: From Visual Grounding to Closed-Loop Action](04-embodied-vision-vla-models/README.md) — bind goals to physical entities; condition affordances on embodiment; compare action representations and behavioral cloning; validate frames, freshness, reach, and collision; issue simulation-only permits; verify postconditions; and report held-out embodiment evidence.
- [05 · Spatial Memory, Scene Graphs & Navigation: From Observations to Persistent World Knowledge](05-spatial-memory-scene-graphs-navigation/README.md) — combine noisy localization, occupancy evidence, persistent object identity, relation provenance, typed current/historical queries, semantic candidate verification, A*, topology, active perception, replanning, and governed memory updates.

## Planned progression

```text
01 3D Vision & Spatial Intelligence
  → 02 Neural Rendering & 3D Scene Representations
  → 03 Dynamic Scenes & World Models
  → 04 Embodied Vision & Vision-Language-Action Models
  → 05 Spatial Memory, Scene Graphs & Navigation
  → multimodal adaptation and production spatial operations
```

Complete Intermediate [Visual Agents](../intermediate/06-visual-agents/README.md) before starting this track. Preserve its evidence, authorization, and bounded-execution contracts when spatial systems later connect perception to action.
