# Contributing

Thank you for helping build the Computer Vision Field Guide.

## Add or improve a lesson

1. Open an issue describing the learning gap, prerequisites, scenario, and success criteria.
2. Create `curriculum/<level>/<number-topic-slug>/` with `README.md`, one descriptive notebook, `lab.py`, and `assets/`.
3. Teach in this sequence: motivation, mental model, foundations, mechanics, architecture choices, worked example, implementation, experiments, evaluation, failures, production, exercises, and references.
4. Keep the default path deterministic, credential-free, CPU-friendly, and safe. Put optional model downloads or provider examples behind clearly documented steps.
5. Add focused tests and update the curriculum index, roadmap, Hub registry, and checkpoint.
6. Run `make test`, `make notebook-check`, and `make links` before opening a pull request.

Use primary papers, standards, official documentation, and maintained implementations. Do not invent benchmark numbers, claims, or citations. Keep datasets small, redistributable, and documented in `data/README.md`.
