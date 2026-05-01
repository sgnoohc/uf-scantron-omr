"""OMR pipeline CLI: scan a directory of exam PDFs and extract OMR answers.

Usage:
    python3 omr.py <input_dir> [--out=combined.json] [--annotate=annot_dir]
                                 [--ext=.pdf] [--dpi=300]

Behavior:
  * Walks <input_dir> for files matching --ext (default .pdf).
  * For each PDF, finds the OMR page (red-density detection), warps to
    canonical coords, reads bubbles using template.json, and writes
    <pdf-stem>.omr.json next to the PDF.
  * If --out is given, writes a single consolidated JSON instead.
  * If --annotate is given, saves per-PDF visualizations to that directory.

Run `python3 build_template.py <sample.pdf> <page_idx>` once to (re)calibrate
template.json before processing exams.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from reader import load_template, process_pdf, process_pdf_grid


def find_pdfs(root: Path, ext: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(f"*{ext}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input_dir", type=Path, help="directory or single PDF")
    ap.add_argument("--out", type=Path, help="combined JSON output path "
                    "(default: write <stem>.omr.json next to each PDF)")
    ap.add_argument("--annotate", type=Path, help="directory to save annotated PNGs")
    ap.add_argument("--ext", default=".pdf", help="file extension (default .pdf)")
    ap.add_argument("--dpi", type=int, default=300, help="render DPI (default 300)")
    ap.add_argument("--template", type=Path,
                    default=Path(__file__).parent / "template.json",
                    help="path to template.json (default: ./template.json)")
    ap.add_argument("--legacy-template", action="store_true",
                    help="use the old template-based reader (default is "
                         "the grid-based reader anchored on the 4 "
                         "fiducial squares + timing stripes)")
    args = ap.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"not found: {args.input_dir}")

    template = load_template(args.template)
    pdfs = find_pdfs(args.input_dir, args.ext)
    if not pdfs:
        sys.exit(f"no {args.ext} files in {args.input_dir}")

    if args.annotate:
        args.annotate.mkdir(parents=True, exist_ok=True)

    results = []
    for pdf in pdfs:
        annot = (str(args.annotate / f"{pdf.stem}.png")
                 if args.annotate else None)
        try:
            if args.legacy_template:
                r = process_pdf(str(pdf), template, dpi=args.dpi,
                                annotate_to=annot)
            else:
                r = process_pdf_grid(str(pdf), dpi=args.dpi)
        except Exception as e:
            r = {"file": str(pdf), "error": f"{type(e).__name__}: {e}"}
        # Use just the basename in the saved record for portability
        r["file"] = pdf.name
        results.append(r)
        msg = (r.get("error")
               or f"page {r.get('omr_page')}, "
                  f"name={r.get('last_name')}, id={r.get('uf_id')}")
        print(f"{pdf}: {msg}")

        if not args.out:
            out_path = pdf.with_suffix(".omr.json")
            out_path.write_text(json.dumps(r, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out} ({len(results)} records)")
    else:
        print(f"\nwrote {len(results)} per-PDF .omr.json files")


if __name__ == "__main__":
    main()
