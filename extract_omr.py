"""Extract OMR pages from each PDF in a directory.

For each input PDF, finds the page(s) that contain the scantron form
(red-pixel density above threshold) and writes a new PDF containing just
those pages. The output is a much smaller per-student PDF that can be
visually graded or fed into a different OMR pipeline.

Usage:
    python3 extract_omr.py <input_dir> <output_dir> [--front-only]
                                                    [--threshold 0.02]
                                                    [--ext .pdf]

Defaults extract every page above the red-density threshold (typically
the front and back of the scantron). Use --front-only to keep only the
single page with the highest score.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import fitz

from utils import find_omr_pages


def find_pdfs(root: Path, ext: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(f"*{ext}"))


def extract(pdf_path: Path, out_path: Path, threshold: float,
            front_only: bool) -> tuple[list[int], str | None]:
    """Write a new PDF at out_path containing the OMR page(s) of pdf_path.
    Returns (page_indices_kept, error_or_None)."""
    candidates = find_omr_pages(str(pdf_path), threshold=threshold)
    if not candidates:
        return [], "no OMR page detected"

    if front_only:
        keep = [candidates[0][0]]
    else:
        # All candidate pages, in original page order (not red-score order),
        # so front comes before back in the output.
        keep = sorted(p for p, _ in candidates)

    src = fitz.open(str(pdf_path))
    out = fitz.open()
    for idx in keep:
        out.insert_pdf(src, from_page=idx, to_page=idx)
    out.save(str(out_path))
    out.close()
    src.close()
    return keep, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input_dir", type=Path, help="directory or single PDF")
    ap.add_argument("output_dir", type=Path,
                    help="directory to write extracted PDFs into")
    ap.add_argument("--front-only", action="store_true",
                    help="keep only the single page with highest red density")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="red-density threshold (default 0.02)")
    ap.add_argument("--ext", default=".pdf",
                    help="file extension (default .pdf)")
    args = ap.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"not found: {args.input_dir}")
    pdfs = find_pdfs(args.input_dir, args.ext)
    if not pdfs:
        sys.exit(f"no {args.ext} files in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = 0
    for pdf in pdfs:
        out_path = args.output_dir / f"{pdf.stem}.omr.pdf"
        kept, err = extract(pdf, out_path, args.threshold, args.front_only)
        if err:
            print(f"  [SKIP] {pdf.name}: {err}")
            n_err += 1
        else:
            print(f"  {pdf.name}: pages {kept} → {out_path.name}")
            n_ok += 1
    print(f"\nextracted {n_ok} PDFs, {n_err} skipped → {args.output_dir}")


if __name__ == "__main__":
    main()
