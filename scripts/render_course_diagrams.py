"""Validate coordinate-based course diagram specs and render deterministic SVGs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def port_point(node: dict, port_name: str) -> tuple[int, int]:
    bounds = node["bounds"]
    port = node["ports"][port_name]
    side, offset = port["side"], port.get("offset", 0.5)
    x, y, width, height = (bounds[key] for key in ("x", "y", "width", "height"))
    if side == "left":
        return x, round(y + height * offset)
    if side == "right":
        return x + width, round(y + height * offset)
    if side == "top":
        return round(x + width * offset), y
    if side == "bottom":
        return round(x + width * offset), y + height
    raise ValueError(f"Unknown port side: {side}")


def overlaps(first: dict, second: dict, clearance: int = 0) -> bool:
    return not (
        first["x"] + first["width"] + clearance <= second["x"]
        or second["x"] + second["width"] + clearance <= first["x"]
        or first["y"] + first["height"] + clearance <= second["y"]
        or second["y"] + second["height"] + clearance <= first["y"]
    )


def validate(spec: dict) -> None:
    required = {"version", "id", "title", "purpose", "output", "canvas", "layout", "style", "groups", "nodes", "edges", "validation"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"{spec.get('id', 'diagram')}: missing keys {sorted(missing)}")

    node_ids = [node["id"] for node in spec["nodes"]]
    edge_ids = [edge["id"] for edge in spec["edges"]]
    group_ids = [group["id"] for group in spec["groups"]]
    if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
        raise ValueError(f"{spec['id']}: node and edge IDs must be unique")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"{spec['id']}: group IDs must be unique")

    width, height, margin = (spec["canvas"][key] for key in ("width", "height", "margin"))
    nodes = {node["id"]: node for node in spec["nodes"]}
    groups = {group["id"]: group for group in spec["groups"]}
    for node in spec["nodes"]:
        bounds = node["bounds"]
        if bounds["x"] < margin or bounds["y"] < margin or bounds["x"] + bounds["width"] > width - margin or bounds["y"] + bounds["height"] > height - margin:
            raise ValueError(f"{spec['id']}: node {node['id']} leaves the safe canvas")
        if node["group"] not in groups:
            raise ValueError(f"{spec['id']}: node {node['id']} references unknown group")
        group = groups[node["group"]]["bounds"]
        if not (
            bounds["x"] >= group["x"]
            and bounds["y"] >= group["y"]
            and bounds["x"] + bounds["width"] <= group["x"] + group["width"]
            and bounds["y"] + bounds["height"] <= group["y"] + group["height"]
        ):
            raise ValueError(f"{spec['id']}: group does not contain node {node['id']}")

    for index, first in enumerate(spec["nodes"]):
        for second in spec["nodes"][index + 1 :]:
            if overlaps(first["bounds"], second["bounds"], clearance=20):
                raise ValueError(f"{spec['id']}: nodes overlap or lack clearance: {first['id']}, {second['id']}")

    for edge in spec["edges"]:
        source, target = edge["from"], edge["to"]
        if source["node"] not in nodes or target["node"] not in nodes:
            raise ValueError(f"{spec['id']}: edge {edge['id']} references unknown node")
        if source["port"] not in nodes[source["node"]]["ports"] or target["port"] not in nodes[target["node"]]["ports"]:
            raise ValueError(f"{spec['id']}: edge {edge['id']} references unknown port")
        route = [tuple(point) for point in edge["route"]]
        if route[0] != port_point(nodes[source["node"]], source["port"]):
            raise ValueError(f"{spec['id']}: edge {edge['id']} does not start at its source port")
        if route[-1] != port_point(nodes[target["node"]], target["port"]):
            raise ValueError(f"{spec['id']}: edge {edge['id']} does not end at its target port")
        if any(x < margin or y < margin or x > width - margin or y > height - margin for x, y in route):
            raise ValueError(f"{spec['id']}: edge {edge['id']} leaves the safe canvas")


def render(spec: dict) -> str:
    canvas, style = spec["canvas"], spec["style"]
    palette = style["semantic_colors"]
    width, height = canvas["width"], canvas["height"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{html.escape(spec["title"])}</title>',
        f'<desc id="description">{html.escape(spec["output"]["alt_text"])}</desc>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#16324F" flood-opacity="0.10"/></filter>',
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#52606D"/></marker>',
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="{canvas["background"]}"/>',
        f'<text x="{canvas["margin"]}" y="58" font-family="{style["font_family"]}" font-size="28" font-weight="700" fill="{style["title_color"]}">{html.escape(spec["title"])}</text>',
    ]

    for group in spec["groups"]:
        bounds = group["bounds"]
        parts.append(
            f'<rect x="{bounds["x"]}" y="{bounds["y"]}" width="{bounds["width"]}" height="{bounds["height"]}" rx="20" fill="{group["fill"]}" stroke="{group["stroke"]}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{bounds["x"] + 22}" y="{bounds["y"] + 34}" font-family="{style["font_family"]}" font-size="16" font-weight="700" fill="{style["title_color"]}">{html.escape(group["label"])}</text>'
        )

    for edge in spec["edges"]:
        points = " ".join(f"{x},{y}" for x, y in edge["route"])
        color = "#52606D" if edge.get("kind") == "primary" else "#8492A6"
        dash = "" if edge.get("kind") == "primary" else ' stroke-dasharray="8 7"'
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>' )
        if edge.get("label"):
            label_x, label_y = edge.get("label_at", edge["route"][len(edge["route"]) // 2])
            parts.append(f'<rect x="{label_x - 58}" y="{label_y - 17}" width="116" height="24" rx="6" fill="{canvas["background"]}"/>')
            parts.append(f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-family="{style["font_family"]}" font-size="13" font-weight="600" fill="#52606D">{html.escape(edge["label"])}</text>')

    for node in spec["nodes"]:
        bounds = node["bounds"]
        colors = palette[node["type"]]
        parts.append(
            f'<rect x="{bounds["x"]}" y="{bounds["y"]}" width="{bounds["width"]}" height="{bounds["height"]}" rx="16" fill="{colors["fill"]}" stroke="{colors["stroke"]}" stroke-width="2" filter="url(#shadow)"/>'
        )
        label_lines = node["label"].split("\n")
        label_y = bounds["y"] + 34
        for offset, line in enumerate(label_lines):
            parts.append(
                f'<text x="{bounds["x"] + bounds["width"] / 2}" y="{label_y + offset * 22}" text-anchor="middle" font-family="{style["font_family"]}" font-size="17" font-weight="700" fill="#16324F">{html.escape(line)}</text>'
            )
        if node.get("subtitle"):
            subtitle_y = label_y + len(label_lines) * 22 + 6
            for offset, line in enumerate(node["subtitle"].split("\n")):
                parts.append(
                    f'<text x="{bounds["x"] + bounds["width"] / 2}" y="{subtitle_y + offset * 18}" text-anchor="middle" font-family="{style["font_family"]}" font-size="13" fill="#52606D">{html.escape(line)}</text>'
                )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path)
    args = parser.parse_args()
    for spec_path in args.specs:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        validate(spec)
        output = ROOT / spec["output"]["svg"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(spec), encoding="utf-8")
        print(f"validated and rendered: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
