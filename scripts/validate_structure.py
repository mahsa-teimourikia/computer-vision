"""Validate topic contracts and repository-local Markdown/Hub links."""

from __future__ import annotations

import json
import re
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


if __name__ == "__main__":
    validate_topics()
    validate_markdown_links()
    validate_hub()
    print("Validated curriculum structure, notebook JSON, Markdown links, and Hub assets.")
