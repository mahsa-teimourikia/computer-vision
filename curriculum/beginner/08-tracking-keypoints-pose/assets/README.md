# Course 08 visual assets

The nine SVGs are generated deterministically from coordinate specifications in `specs/` by the repository diagram renderer. Each SVG includes an accessible title, description, stable palette, contained labels, and explicit flow.

Regenerate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/beginner/08-tracking-keypoints-pose/assets/specs/*.json
```
