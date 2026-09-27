# Contributing

Thank you for helping build the Computer Vision Field Guide.

Start with the supported Python versions and environment options in [INSTALLATION.md](INSTALLATION.md). The root environment is the maintained contributor/CI contract; each course also keeps a minimal `requirements.txt` plus a fully pinned `constraints-tested.txt` for focused study.

## Add or improve a lesson

1. Open an issue describing the learning gap, prerequisites, scenario, and success criteria.
2. Create `curriculum/<level>/<number-topic-slug>/` with `README.md`, one self-contained descriptive notebook, `requirements.txt`, `constraints-tested.txt`, and `assets/`.
3. Teach in this sequence: motivation, mental model, foundations, mechanics, architecture choices, tooling landscape, state of the art, worked example, implementation, experiments, evaluation, failures, production, exercises, and references.
4. Keep the default path deterministic, credential-free, CPU-friendly, and safe. Put optional model downloads or provider examples behind clearly documented steps.
5. Add focused tests and update the curriculum index, roadmap, Hub registry, and explanatory checkpoint. A checkpoint must include at least five questions, exactly one defensible answer per question, and feedback that identifies that answer and the review topic after submission.
6. Run `make check` before opening a pull request. For a faster documentation-only pass, use `make test`, `make quiz-check`, and `make links`; run `make notebook-check COURSE=<course-directory>` while iterating on one notebook.

Course and implementation plans are development artifacts. Use them while designing substantial lessons, but remove lesson-local `COURSE_PLAN.md` files before merge unless the repository adopts the same retained-plan convention for every completed course.

Use primary papers, standards, official documentation, and maintained implementations. Do not invent benchmark numbers, claims, or citations. Keep datasets small, redistributable, and documented in `data/README.md`.

Every lesson that discusses current models must date its review, separate established practice from emerging practice and research prototypes, cite primary/official sources close to the claims, and compare at least two relevant tools using the criteria in [TOOLING.md](TOOLING.md). Never promote a benchmark result without documenting the task, dataset, metric, split, resolution, hardware, and evaluation limitations.

Notebook code should use common maintained libraries and SDKs appropriate to the topic. Prefer NumPy, Matplotlib, Pillow/OpenCV, PyTorch/torchvision, Hugging Face, OpenMMLab, FiftyOne, Open3D, LeRobot, ONNX Runtime, and comparable ecosystem standards when they clarify the actual workflow. Keep every required import in the notebook, pin or report important versions, and isolate optional heavyweight integrations so the default path remains runnable.

When updating a dependency, change the root extra and every affected course constraint together, verify the supported Python range, run `pip check`, execute all notebooks in a clean environment, and record the validation date only after those checks pass. Do not silently broaden an unbounded range or claim a package combination is tested based only on dependency resolution.
