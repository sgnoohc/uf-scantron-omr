# OMR — UF Form LR1 scantron reader

Reads scanned exam PDFs that include a [UF Form LR1](https://drc.dso.ufl.edu/)
scantron sheet, finds the OMR page inside each PDF, deskews it, locates every
bubble using the form's printed fiducials and timing stripes (no per-scan
template needed), and decodes every field — last name, FI / MI, UF ID,
section, test form code, all 80 answers — into JSON. A second tool exports
the JSON to a fixed-width SDF file for flexam.

## Quick start

### One-shot: directory of raw scanned PDFs → single SDF

The simplest path. Works on raw DRC bulk-scan PDFs (each PDF can be
multi-page, the OMR page is auto-detected by red-pixel density):

```bash
# 1. clone + install (one-time)
git clone https://github.com/sgnoohc/uf-scantron-omr.git
cd uf-scantron-omr
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. PDF directory → flexam-ready SDF (one command)
python3 pdf_to_sdf.py /path/to/scans/ /path/to/out.sdf
```

That's it. The SDF is sorted by LAST_NAME, fixed-width 116 cols/line,
CRLF terminated, ready for flexam.

Optionally also dump a combined JSON for inspection:

```bash
python3 pdf_to_sdf.py /path/to/scans/ out.sdf --save-json out.json
```

### Step-by-step (for QA / inspection)

If you want to spot-check before committing to the SDF, run the steps
individually:

```bash
# A. extract per-PDF JSON next to each PDF
python3 omr.py /path/to/scans/

# B. render an annotated PDF — every page shows detected fills as green
#    rings on the original scan with a sidebar listing all decoded fields
python3 annotate_omr.py /path/to/scans/ /path/to/annotated.pdf

# C. fold the JSONs into the flexam SDF
python3 to_sdf.py /path/to/scans/ /path/to/out.sdf
```

---

## Output format

```json
{
  "file": "exam.pdf",
  "omr_page": 2,
  "red_score": 0.108,
  "last_name": "EXAMPLE",
  "first_initial": "X",
  "middle_initial": null,
  "test_form_code": "A",
  "uf_id": "12345678",
  "section": null,
  "answers": {"1": "C", "2": "D", "3": "B", "4": "C", ..., "80": "B"}
}
```

- Unfilled bubble → `null`.
- Multiple bubbles in one row → list, e.g. `"5": ["A", "C"]`.
- `omr_page` is the 0-indexed page in the source PDF.

---

## Prerequisites

| | |
|---|---|
| Python | 3.9+ |
| OS | macOS, Linux, Windows (WSL) |
| Disk | ~70 MB for Python deps |
| System packages | none — PyMuPDF and `opencv-python-headless` ship their own libs |

Install: `pip install -r requirements.txt` (PyMuPDF, opencv-python-headless, numpy).

Alternatives:
- `make venv && source .venv/bin/activate` — one-shot venv setup
- `make install` — user-site `pip install --user`
- `make docker && make run DIR=/path/to/scans` — fully isolated container

---

## CLI reference

### `pdf_to_sdf.py` — one-shot end-to-end pipeline

```
python3 pdf_to_sdf.py <input_dir> <output.sdf>
                      [--layout sdf_layout.json]
                      [--sort last_name|uf_id|file|none]
                      [--save-json PATH]
                      [--ext .pdf] [--dpi 300]
```

| flag | default | meaning |
|---|---|---|
| `<input_dir>` | required | directory of scanned PDFs (or single PDF) |
| `<output.sdf>` | required | path to write the combined SDF |
| `--layout` | `./sdf_layout.json` | column layout for the SDF |
| `--sort` | `last_name` | record ordering |
| `--save-json` | unset | also dump combined JSON to this path |
| `--ext` | `.pdf` | file extension to scan for |
| `--dpi` | `300` | rasterization DPI |

Records that fail OMR detection are reported on stderr and skipped from
the SDF (so the SDF only contains usable rows).

### `omr.py` — extract OMR data to JSON

```
python3 omr.py <input> [--out FILE] [--ext .pdf] [--dpi 300]
                       [--legacy-template] [--template PATH]
                       [--annotate DIR]
```

| flag | default | meaning |
|---|---|---|
| `<input>` | required | PDF file or directory (recursed) |
| `--out FILE` | unset | write one combined JSON; otherwise `<stem>.omr.json` next to each PDF |
| `--ext` | `.pdf` | extension when `<input>` is a directory |
| `--dpi` | `300` | rasterization DPI |
| `--legacy-template` | off | use the old per-scan template-fit reader (default is the fiducial grid reader) |
| `--template PATH` | `./template.json` | only used with `--legacy-template` |
| `--annotate DIR` | unset | save canonical-warp PNGs (legacy reader only) |

### `annotate_omr.py` — visualize the read

```
python3 annotate_omr.py <input> <output.pdf>
                        [--combined | --per-pdf] [--ext .pdf] [--dpi 300]
```

Produces a multi-page PDF where each page shows the original scan with:
- Blue horizontal lines through every printed timing stripe (47 rows)
- Blue vertical lines through every bubble column (28 cols, anchored on the 4
  black fiducial squares)
- Green rings around every bubble identified as filled
- A right-side sidebar listing the decoded fields (last_name, FI, MI, UFID,
  section, form code, all 80 answers)

Use this to spot-check accuracy. Misreads are visually obvious.

### `to_sdf.py` — JSON → flexam SDF

```
python3 to_sdf.py <json_dir> <out.sdf> [--layout sdf_layout.json]
                                       [--sort last_name|uf_id|file|none]
```

Folds every `*.omr.json` in `<json_dir>` into one fixed-width text file (CRLF
line endings, 116 cols/line). Default layout (edit `sdf_layout.json` to
change):

| cols | content |
|---|---|
| 0-11 | LAST_NAME (left-aligned) |
| 12 | FI |
| 13 | MI |
| 14-21 | UF_ID (right-aligned) |
| 30-34 | answers Q76-Q80 |
| 35-44 | answers Q1-Q10 |
| 46-55 | answers Q11-Q20 |
| 78-87 | answers Q21-Q30 |
| 111-115 | answers Q31-Q35 |

Encoding: A=1, B=2, C=3, D=4, E=5. Blank = space, ambiguous (multiple bubbles
in one row) = `*`.

### `extract_omr.py` — pre-extract OMR pages from each PDF

```
python3 extract_omr.py <input_dir> <output_dir> [--front-only]
                                                [--threshold 0.02] [--ext .pdf]
```

For each input PDF, finds the page(s) above the red-pixel density threshold
and writes a smaller per-student PDF containing just those pages. Useful as
a pre-processing step when the source PDFs are large multi-page scans.

---

## How it works

1. **OMR-page detection** (`utils.find_omr_pages`). Every page is rendered at
   low DPI; the page with the highest red-pixel density wins (the form
   prints in red).
2. **Form quad detection** (`utils.find_form_quad`). The largest red-ink
   contour gives the form's 4 corners. A bubble-cluster fallback handles
   scans where paper bleed inflates the red-ink boundary.
3. **Deskew** (`utils.warp_to_canonical`). Perspective warp maps the quad
   to a canonical 1200×1800 image. Rotation (0°, 90°, 180°, 270°) is
   auto-picked by stripe-uniformity scoring.
4. **Fiducial detection** (`annotate_omr._detect_top_fiducials`). The 4
   black squares at the top of the form are located by connected-component
   analysis with template-matching fallback.
5. **Stripe-grid fit** (`utils._fit_stripe_grid`). Iteratively reweighted
   least squares fit of `y(i, x) = c_0 + i*sp + slope*x` to the 47 printed
   left-margin timing stripes plus the 4 fiducial centers. Drops outlier
   stripes; emits a 47-row grid even when some stripes are missing from
   the scan (e.g. clipped scans).
6. **Column grid from fiducials**. Form-unit indices `k = 2, 3, 21, 23` are
   pinned to the 4 fiducial X positions; least-squares fit gives a
   uniform-pitch column grid for `k = 1..28`. UF Form LR1 uses ONE pitch
   across LAST_NAME / FI / MI / UF_ID / SECTION / answer blocks /
   TEST_FORM_CODE.
7. **Bubble scoring** (`reader.score_bubble`). For each (col k, row i), the
   exact perpendicular intersection of the column line with the row line is
   computed (accounting for tilt); a small patch is sampled on the
   CLAHE-enhanced **red channel** of the warped image. Pencil graphite
   absorbs all wavelengths; red-ink form printing reflects strongly in red.
   The score is the fraction of pixels with intensity below `DARK_THRESH`.
8. **Field decode** (`reader._winner` and friends). Per-region decode rules:
   - Answers: argmax per row (one choice per question)
   - LAST_NAME / FI / MI / UF_ID / SECTION: argmax per column (one symbol
     per column)
   - TEST_FORM_CODE: single-row scan
   The winner picker has 4 fallbacks for marginal scores: absolute
   threshold, leading margin, above-median, and z-score (≥ 2σ above the
   column's other bubbles, for digits/answers only).

---

## Form layout reference

In stripe-row indices (i = 0 is the topmost timing-stripe row):

| stripe i | content | columns (form-unit k) |
|---|---|---|
| 2 | TEST_FORM_CODE (A B C D E) | 24-28 |
| 4-30 | LAST_NAME / FI / MI (WRITE row + A-Z) | 1-12 / 13 / 14 |
| 4-13 | answers Q1-Q10 | 17-21 (left) |
| 4-13 | answers Q41-Q50 | 24-28 (right) |
| 15-24 | answers Q11-Q20 / Q51-Q60 | 17-21 / 24-28 |
| 26-35 | answers Q21-Q30 / Q61-Q70 | 17-21 / 24-28 |
| 34-43 | UF_ID (digits 0-9 top-to-bottom) | 1-8 |
| 34-43 | SECTION (digits 0-9) | 9, 11, 12, 13 (non-contiguous on form) |
| 37-46 | answers Q31-Q40 / Q71-Q80 | 17-21 / 24-28 |

This layout is encoded in `reader.GRID_LAYOUT`. Edit it to support different
form variants.

---

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `no OMR page detected` | scantron not the dominant red-ink page | lower threshold in `utils.find_omr_pages` |
| `error: fiducial/stripe detection failed` | top of form clipped from scan | re-scan with full top margin visible |
| LAST_NAME has trailing underscores | student wrote a shorter name | not a bug — `_` denotes blank columns |
| LAST_NAME has LEADING underscores | student right-justified the name (e.g. a 6-letter name in cols 7-12) | not a bug |
| UF_ID partial (`12_4567_`) | student left bubbles unfilled in those columns | spot-check via `annotate_omr.py` to confirm |
| Faintly-marked scan reads as None | very light pencil pressure | already mitigated by red-channel + CLAHE; if still failing, increase `clipLimit` in `reader.green_channel` |
| Multiple bubbles flagged in a column | student bubbled two answers | `decode_column_winner` returns `None` → underscore in output |
| Wrong digit shifted by ±1 | `GRID_LAYOUT` row range off by one | check stripe-row mapping for the affected region |

---

## Repo layout

```
omr/
├── pdf_to_sdf.py         # CLI: PDF dir → single SDF (one-shot pipeline)
├── omr.py                # CLI: PDFs → .omr.json (intermediate step)
├── annotate_omr.py       # CLI: PDFs → annotated debug PDF (QA)
├── to_sdf.py             # CLI: .omr.json dir → fixed-width SDF
├── extract_omr.py        # CLI: pull OMR pages out of multi-page PDFs
├── reader.py             # grid reader (read_omr_grid) + legacy template reader
├── build_template.py     # legacy template auto-calibration (only needed for --legacy-template)
├── utils.py              # PDF rendering, red-density page pick, deskew, stripe fit
├── sdf_layout.json       # column layout for SDF export
├── template.json         # legacy template (only used by --legacy-template)
├── fiducial_marker.png   # template image for fiducial template-matching fallback
├── requirements.txt
├── Makefile              # venv / install / docker / run targets
├── Dockerfile
└── README.md
```

---

## Caveats

- **Spot-check before grading.** Run `annotate_omr.py` and flip through the
  resulting PDF before trusting any large batch.
- **Severely skewed scans** (>2° rotation) may still align imperfectly even
  with the slope-aware grid fit. Re-scan rather than fight the warp.
- **The fiducial grid assumes a 4-fiducial UF Form LR1.** Other scantron
  forms need their own column-anchor strategy.
- **Faint pencil scans** are partially recovered by the red-channel + CLAHE
  pipeline but may still leave some columns blank if the pressure is too
  light.

---

## License

MIT — see [LICENSE](./LICENSE).
