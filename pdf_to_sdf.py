"""End-to-end pipeline: directory of raw scanned PDFs → single SDF.

For each PDF in the input directory (typical DRC bulk scan output —
multi-page, only one or two pages contain the OMR scantron):

  1. Find the OMR page by red-pixel density (other pages are skipped).
  2. Deskew the form via 4-fiducial perspective warp.
  3. Decode all fields (LAST_NAME, FI, MI, UFID, SECTION,
     TEST_FORM_CODE, all 80 answers) using the fiducial-anchored
     uniform-pitch grid reader.
  4. Render each decoded record to a fixed-width SDF line per
     `sdf_layout.json`.
  5. Write the combined SDF file (CRLF line endings, sorted by
     LAST_NAME by default).

No intermediate per-PDF JSON files are written; if you want them, pass
`--save-json combined.json` to also dump every record.

Usage:
    python3 pdf_to_sdf.py <input_dir> <output.sdf>
                          [--layout sdf_layout.json]
                          [--sort last_name|uf_id|file|none]
                          [--save-json PATH]
                          [--ext .pdf] [--dpi 300]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from reader import process_pdf_grid
from to_sdf import DEFAULT_LAYOUT, load_layout, render_line, sort_records


def find_pdfs(root: Path, ext: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(f"*{ext}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input_dir", type=Path,
                    help="directory of scanned PDFs (or a single PDF)")
    ap.add_argument("out_sdf", type=Path,
                    help="output SDF file path")
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT,
                    help="SDF layout JSON (default: sdf_layout.json beside this script)")
    ap.add_argument("--sort", choices=["none", "last_name", "uf_id", "file"],
                    default="last_name", help="record ordering (default: last_name)")
    ap.add_argument("--save-json", type=Path, default=None,
                    help="optionally also dump a combined JSON file")
    ap.add_argument("--ext", default=".pdf",
                    help="file extension (default: .pdf)")
    ap.add_argument("--dpi", type=int, default=300,
                    help="render DPI (default: 300)")
    args = ap.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"not found: {args.input_dir}")
    pdfs = find_pdfs(args.input_dir, args.ext)
    if not pdfs:
        sys.exit(f"no {args.ext} files in {args.input_dir}")

    layout = load_layout(args.layout)

    records: list[dict] = []
    n_ok = n_err = 0
    for pdf in pdfs:
        try:
            r = process_pdf_grid(str(pdf), dpi=args.dpi)
        except Exception as e:
            r = {"file": str(pdf), "error": f"{type(e).__name__}: {e}"}
        r["file"] = pdf.name  # store basename only
        records.append(r)
        if r.get("error"):
            n_err += 1
            print(f"  [SKIP] {pdf.name}: {r['error']}")
        else:
            n_ok += 1
            print(f"  {pdf.name}: name={r.get('last_name')}, "
                   f"id={r.get('uf_id')}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(records, indent=2))
        print(f"\nwrote JSON: {args.save_json} ({len(records)} records)")

    valid_records = [r for r in records if not r.get("error")]
    valid_records = sort_records(valid_records, args.sort)
    lines = [render_line(r, layout) for r in valid_records]

    args.out_sdf.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_sdf, "wb") as f:
        f.write(layout["line_ending"].join(lines).encode("ascii", "replace"))
        f.write(layout["line_ending"].encode("ascii"))

    print(f"\nwrote SDF: {args.out_sdf}")
    print(f"  {n_ok} records written, {n_err} skipped, "
           f"{layout['line_width']} cols/line")


if __name__ == "__main__":
    main()
