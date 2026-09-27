"""Validate topic contracts and repository-local Markdown/Hub links."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def validate_topics() -> None:
    topics = sorted(ROOT.glob("curriculum/*/[0-9][0-9]-*/README.md"))
    if not topics:
        raise AssertionError("At least one complete curriculum topic is required")
    for readme in topics:
        topic = readme.parent
        notebooks = list(topic.glob("*.ipynb"))
        if len(notebooks) != 1:
            raise AssertionError(f"{topic.relative_to(ROOT)} must own exactly one notebook")
        for required in (topic / "assets", topic / "requirements.txt", topic / "constraints-tested.txt"):
            if not required.exists():
                raise AssertionError(f"Missing required topic resource: {required.relative_to(ROOT)}")
        if list(topic.glob("*.py")):
            raise AssertionError(
                f"Teaching code must stay in the notebook; unexpected Python module in {topic.relative_to(ROOT)}"
            )
        data = json.loads(notebooks[0].read_text(encoding="utf-8"))
        if data.get("nbformat") != 4 or not data.get("cells"):
            raise AssertionError(f"Invalid notebook: {notebooks[0].relative_to(ROOT)}")


def validate_markdown_links() -> None:
    for document in ROOT.rglob("*.md"):
        if ".git" in document.parts:
            continue
        for raw_target in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.split()[0].strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            local = (document.parent / unquote(parsed.path)).resolve()
            if not local.exists():
                raise AssertionError(
                    f"Broken local link in {document.relative_to(ROOT)}: {target}"
                )


def validate_hub() -> None:
    page = (ROOT / "hub/index.html").read_text(encoding="utf-8")
    required = [
        "Learn",
        "Lab",
        "Checkpoint",
        "Modern Computer Vision Foundations",
        "Modern CNN Architectures &amp; Efficient Vision",
        "Vision Transformers",
        "Self-Supervised Visual Representation Learning",
        "Object Detection",
        "Segmentation &amp; Promptable Segmentation",
        "Visual Embeddings, Metric Learning &amp; Retrieval",
        "Tracking, Keypoints &amp; Pose",
        "Vision Foundation Models &amp; Open-Vocabulary Vision",
        "Vision-Language Models",
        "Multimodal Reasoning &amp; Verification",
        "Document Intelligence",
        "Multimodal Retrieval &amp; RAG",
        "Video-Language Understanding",
        "Visual Agents",
        "3D Vision &amp; Spatial Intelligence",
        "Neural Rendering &amp; 3D Scene Representations",
        "Dynamic Scenes &amp; World Models",
        "Embodied Vision &amp; Vision-Language-Action Models",
        "Spatial Memory, Scene Graphs &amp; Navigation",
        "Multimodal Adaptation &amp; Continual Learning",
        "Robustness, Uncertainty &amp; Failure Recovery",
        "Efficient Spatial &amp; Multimodal Inference",
        "Production Spatial AI Operations &amp; Observability",
        "oneplusi.io",
    ]
    for text in required:
        if text not in page:
            raise AssertionError(f"Hub is missing required content: {text}")
    for target in re.findall(r'(?:href|src)="([^"]+)"', page):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        local = ROOT / "hub" / target.split("?")[0].split("#")[0]
        if not local.exists():
            raise AssertionError(f"Broken Hub asset link: {target}")

    topics = sorted(ROOT.glob("curriculum/*/[0-9][0-9]-*/README.md"))
    course_paths = [topic.parent.relative_to(ROOT).as_posix() for topic in topics]
    available_ids = re.findall(r'<button class="lesson-card available(?: selected)?"[^>]+data-workspace="([^"]+)"', page)
    workspace_ids = re.findall(r'data-workspace-panel="([^"]+)"', page)
    quiz_forms = re.findall(r'<form class="quiz-form"[^>]*>(.*?)</form>', page, flags=re.DOTALL)
    if len(available_ids) != len(course_paths) or set(available_ids) != set(workspace_ids):
        raise AssertionError("Every published course must map to one available card and one workspace")
    if len(quiz_forms) != len(course_paths):
        raise AssertionError("Every published course must own exactly one Hub checkpoint")
    for course_path in course_paths:
        if f"{course_path}/README.md" not in page or f"{course_path}/lab.ipynb" not in page:
            raise AssertionError(f"Hub is missing chapter or notebook links for {course_path}")
    for index, form in enumerate(quiz_forms, start=1):
        fieldsets = re.findall(r"<fieldset>(.*?)</fieldset>", form, flags=re.DOTALL)
        if len(fieldsets) < 5:
            raise AssertionError(f"Checkpoint {index} must contain at least five questions")
        names = []
        for question in fieldsets:
            options = re.findall(r'<input type="radio" name="([^"]+)" value="([01])"', question)
            if len(options) < 3 or sum(value == "1" for _, value in options) != 1:
                raise AssertionError(f"Checkpoint {index} has an invalid question contract")
            names.append(options[0][0])
        if len(names) != len(set(names)):
            raise AssertionError(f"Checkpoint {index} reuses a radio name")
    for required_behavior in ("localStorage", "answer-feedback", "reset-progress", "course-progress"):
        source = page if required_behavior in {"reset-progress", "course-progress"} else (ROOT / "hub/app.js").read_text(encoding="utf-8")
        if required_behavior not in source:
            raise AssertionError(f"Hub is missing learner-state behavior: {required_behavior}")


def validate_environment() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    devcontainer = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text(encoding="utf-8"))
    workflow = (ROOT / ".github/workflows/validate-learning.yml").read_text(encoding="utf-8")
    if not python_version.startswith("3.13") or "3.13" not in devcontainer["image"]:
        raise AssertionError(".python-version and the development container must use the tested Python 3.13 line")
    for expected in ("actions/checkout@v7", "actions/setup-python@v7", f"python-version: '{python_version}'"):
        if expected not in workflow:
            raise AssertionError(f"Validation workflow is missing {expected}")
    if project["requires-python"] != ">=3.12,<3.14":
        raise AssertionError("Project Python support must match the installation guide")

    declared = {}
    dependency_groups = [project["dependencies"], *project["optional-dependencies"].values()]
    for requirement in (item for group in dependency_groups for item in group):
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;]+)", requirement)
        if match:
            declared[match.group(1).lower()] = match.group(2)
    for constraints in ROOT.glob("curriculum/*/[0-9][0-9]-*/constraints-tested.txt"):
        constraint_source = constraints.read_text(encoding="utf-8")
        if not constraint_source.startswith("# Tested together on ") or "Python 3.13.5" not in constraint_source.splitlines()[0]:
            raise AssertionError(f"{constraints.relative_to(ROOT)} must begin with its tested runtime record")
        for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", constraint_source, flags=re.MULTILINE):
            if name.lower() in declared and declared[name.lower()] != version:
                raise AssertionError(f"{constraints.relative_to(ROOT)} disagrees with pyproject.toml for {name}")
    for requirements in ROOT.glob("curriculum/*/[0-9][0-9]-*/requirements.txt"):
        source = requirements.read_text(encoding="utf-8")
        if "constraints-tested.txt" not in source or "jupyterlab" not in source:
            raise AssertionError(f"{requirements.relative_to(ROOT)} must use tested constraints and include JupyterLab")

    installation = (ROOT / "INSTALLATION.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for guide_text, make_target in (
        ("setup-course", "setup-course:"),
        ("setup-learner", "setup-learner:"),
        ("quiz-check", "quiz-check:"),
        ("notebook-check", "notebook-check:"),
        ("make check", "check:"),
    ):
        if guide_text not in installation or make_target not in makefile:
            raise AssertionError(f"Installation workflow is missing {guide_text}")


if __name__ == "__main__":
    validate_topics()
    validate_markdown_links()
    validate_hub()
    validate_environment()
    print("Validated curriculum structure, notebook JSON, Markdown links, Hub checkpoints, and environment metadata.")
