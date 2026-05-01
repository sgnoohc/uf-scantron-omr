"""Convert a directory of .omr.json files to a single flexam SDF file.

Usage:
    python3 to_sdf.py <json_dir> <out.sdf> [--layout sdf_layout.json]

The layout JSON pins each field to a fixed column in the SDF line. Columns
are 0-indexed; the output line is padded with spaces to `line_width` and
terminated with `line_ending`. Edit sdf_layout.json to change which question
goes where (e.g. if the grader expects a different question ordering).

A field entry has:
  col:        starting column (0-indexed)
  width:      number of characters
  source:     "last_name" | "first_initial" | "middle_initial" | "uf_id" | "answers"
  questions:  required when source == "answers" — list of question numbers,
              one character per slot in column order.
  align:      "left" or "right"  (for string fields; default left)
  pad:        pad character for string fields (default " ")
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

DEFAULT_LAYOUT = Path(__file__).resolve().parent / "sdf_layout.json"


def load_layout(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def encode_answer(value, layout: dict) -> str:
    """Map an OMR answer value (str | list | None) to a single SDF character."""
    if value is None:
        return layout["blank_char"]
    if isinstance(value, list):
        return layout["ambiguous_char"]
    return layout["answer_encoding"].get(str(value).upper(),
                                         layout["ambiguous_char"])


def render_line(record: dict, layout: dict) -> str:
    """Build one SDF line from one .omr.json record."""
    line = [layout["blank_char"]] * layout["line_width"]
    for field in layout["fields"]:
        col, width = field["col"], field["width"]
        src = field["source"]

        if src == "answers":
            answers = record.get("answers") or {}
            questions = field["questions"]
            if len(questions) != width:
                raise ValueError(
                    f"answers field at col {col}: width={width} but "
                    f"{len(questions)} questions given")
            chars = [encode_answer(answers.get(str(q)) or answers.get(q),
                                   layout) for q in questions]
            text = "".join(chars)
        else:
            raw = record.get(src)
            text = "" if raw is None else str(raw).upper()
            if len(text) > width:
                text = text[:width]
            pad = field.get("pad", " ")
            if field.get("align", "left") == "right":
                text = text.rjust(width, pad)
            else:
                text = text.ljust(width, pad)

        if col + width > layout["line_width"]:
            raise ValueError(
                f"field at col {col} width {width} overflows line_width "
                f"{layout['line_width']}")
        for i, ch in enumerate(text):
            line[col + i] = ch

    return "".join(line)


def collect_records(json_dir: Path) -> list[dict]:
    """Load all *.omr.json files in dir. If a file is a list, flatten it."""
    paths = sorted(json_dir.rglob("*.omr.json"))
    out = []
    for p in paths:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            out.extend(data)
        else:
            out.append(data)
    return out


def sort_records(records: list[dict], by: str) -> list[dict]:
    if by == "last_name":
        return sorted(records,
                      key=lambda r: ((r.get("last_name") or "").upper(),
                                     (r.get("first_initial") or "")))
    if by == "uf_id":
        return sorted(records, key=lambda r: r.get("uf_id") or "")
    if by == "file":
        return sorted(records, key=lambda r: r.get("file") or "")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json_dir", type=Path, help="directory of .omr.json files")
    ap.add_argument("out_sdf", type=Path, help="output .sdf path")
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT,
                    help="layout JSON (default: sdf_layout.json beside this script)")
    ap.add_argument("--sort", choices=["none", "last_name", "uf_id", "file"],
                    default="last_name", help="record ordering (default: last_name)")
    args = ap.parse_args()

    if not args.json_dir.exists():
        sys.exit(f"not found: {args.json_dir}")
    layout = load_layout(args.layout)
    records = collect_records(args.json_dir)
    if not records:
        sys.exit(f"no .omr.json files under {args.json_dir}")
    records = sort_records(records, args.sort)

    lines = [render_line(r, layout) for r in records]
    args.out_sdf.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_sdf, "wb") as f:
        f.write(layout["line_ending"].join(lines).encode("ascii", "replace"))
        f.write(layout["line_ending"].encode("ascii"))
    print(f"wrote {args.out_sdf} — {len(lines)} records, "
          f"{layout['line_width']} cols/line")


if __name__ == "__main__":
    main()
