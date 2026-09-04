# Course 09 visual assets

The eight SVGs are generated deterministically from coordinate specifications in `specs/` by the repository diagram renderer. Each SVG includes an accessible title, description, stable palette, contained labels, and explicit flow.

Regenerate from the repository root:

```bash
python scripts/render_course_diagrams.py curriculum/beginner/09-vision-foundation-models-open-vocabulary/assets/specs/*.json
```
