"""OMR reader: load template, score each bubble, decode fields."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from utils import (warp_omr, render_page, find_form_quad,
                   detect_timing_marks, map_stripes_to_canonical,
                   filter_stripes, EXPECTED_STRIPES)
from build_template import detect_bubbles, fit_grid, REGIONS

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "template.json"

# Pixel intensity below this counts as "dark" (a pencil mark) on the green
# channel, where red form ink is suppressed but pencil marks (gray) remain.
DARK_THRESH = 150

# Absolute fill threshold: a bubble whose dark-pixel fraction is >= this
# counts as "definitely filled" regardless of context. Empty bubbles score
# ~0.25 (just the printed letter); a clean filled bubble scores ~0.95.
FILL_THRESH = 0.55

# Per-kind decoding thresholds. "letters" (LAST_NAME / FI / MI) have many
# unfilled columns and so need stricter rules to avoid false-positive
# letters. "digits" (UF_ID / SECTION) typically have ALL columns filled,
# often lightly, so we can be more permissive about light pencil marks.
# Each tuple is (margin_thresh, above_median_thresh).
KIND_THRESHOLDS = {
    "letters":    (0.18, 0.20),
    "digits":     (0.12, 0.13),
    "answers":    (0.18, 0.20),
    "single_row": (0.18, 0.20),
    "codes":      (0.18, 0.20),
}

# Patch radius in pixels (canonical 1200x1800). Bubble inner radius is ~9.
PATCH_R = 9


def load_template(path: str | Path = TEMPLATE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def score_bubble(green: np.ndarray, x: float, y: float, r: int = PATCH_R) -> float:
    """Fraction of dark pixels in a square patch centered at (x, y)."""
    h, w = green.shape
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r + 1), min(h, int(y) + r + 1)
    patch = green[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float((patch < DARK_THRESH).mean())


def score_grid(green: np.ndarray, region: dict) -> np.ndarray:
    """Return rows×cols score matrix for a region's bubble grid."""
    xs = region["xs"]
    ys = region["ys"]
    if xs is None or ys is None:
        return np.zeros((region["rows"], region["cols"]))
    out = np.zeros((len(ys), len(xs)))
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            out[r, c] = score_bubble(green, x, y)
    return out


def _winner(scores: np.ndarray, labels, kind: str = "letters"):
    """Pick the winning bubble in a 1-D score array.

    Returns:
        - None: no bubble is plausibly filled
        - label (str): single clear winner
        - list: multiple bubbles tied/ambiguous

    A bubble wins if any of these are true:
      1. Score ≥ FILL_THRESH (definitely filled)
      2. Score leads runner-up by ≥ margin AND score ≥ 0.35
         (true fill whose absolute score was suppressed by misalignment)
      3. Score is ≥ above_median above the median of all scores in
         the group AND score ≥ 0.35 (rescues light pencil marks)
      4. Z-score fallback: if scores are tightly clustered (low std),
         a small absolute lead can still be a real fill. Used for very
         faint scans (faint scans) where everything is dim — the FILLED
         bubble still stands out 2+ std above the column's mean even
         though absolute differences are small.

    Margin and above_median thresholds depend on `kind` — see
    KIND_THRESHOLDS at top of file."""
    n = len(scores)
    if n == 0:
        return None
    margin_thresh, above_median_thresh = KIND_THRESHOLDS.get(
        kind, KIND_THRESHOLDS["letters"])
    sorted_idx = np.argsort(-scores)
    top_score = float(scores[sorted_idx[0]])
    second_score = float(scores[sorted_idx[1]]) if n > 1 else 0.0
    median_score = float(np.median(scores))

    if top_score >= FILL_THRESH:
        # If the top score is clearly the winner (leads runner-up by a
        # solid margin), prefer it even when other bubbles also cross
        # FILL_THRESH. CLAHE-amplified scans can lift printed-letter
        # strokes inside empty bubbles to 0.55-0.65, but a true filled
        # bubble usually scores > 0.85 — the gap is the discriminator.
        if top_score - second_score >= 0.25:
            return labels[int(sorted_idx[0])]
        filled = [int(i) for i in range(n) if scores[i] >= FILL_THRESH]
        if len(filled) == 1:
            return labels[filled[0]]
        return [labels[i] for i in filled]

    if top_score >= 0.35:
        if top_score - second_score >= margin_thresh:
            return labels[int(sorted_idx[0])]
        if top_score - median_score >= above_median_thresh:
            return labels[int(sorted_idx[0])]

    # Z-score fallback for faint scans: top must be ≥ 0.40 AND at least
    # 2.0 σ above the OTHERS' mean (excluding the top itself, so a single
    # fill doesn't inflate σ). Restricted to digit/answer kinds — letter
    # columns often have legitimately ALL low scores (no fill at all)
    # which a z-score would falsely promote to a winner.
    if kind in ("digits", "answers") and top_score >= 0.40:
        others = np.delete(scores, sorted_idx[0])
        o_mean = float(np.mean(others))
        o_std = float(np.std(others))
        if o_std > 0.01 and (top_score - o_mean) / o_std >= 2.0:
            return labels[int(sorted_idx[0])]
    return None


def decode_answers(scores: np.ndarray, row_labels, col_labels) -> dict:
    """One choice per row. Returns {row_label: col_label or None}."""
    return {rlabel: _winner(scores[r], col_labels, kind="answers")
            for r, rlabel in enumerate(row_labels)}


def decode_column_winner(scores: np.ndarray, row_labels, col_labels,
                         kind: str = "letters"):
    """One filled row per column. Used for LAST_NAME, UF_ID, SECTION, FI, MI.
    Skips any blank-labeled (" ") leading rows (legacy writing-row support).
    """
    r0 = 0
    while r0 < len(row_labels) and not row_labels[r0].strip():
        r0 += 1
    out = []
    for c in range(scores.shape[1]):
        col = scores[r0:, c]
        winner = _winner(col, row_labels[r0:], kind=kind)
        out.append(winner if isinstance(winner, str) else None)
    return out


def decode_single_row(scores: np.ndarray, row_labels, col_labels,
                      kind: str = "single_row"):
    """Single row of choice bubbles: pick the one filled."""
    if scores.shape[0] == 0:
        return None
    return _winner(scores[0], col_labels, kind=kind)


def decode_codes(scores: np.ndarray, row_labels, col_labels):
    """Each row decoded as a single-row choice. Returns list."""
    return [decode_single_row(scores[i:i+1], [row_labels[i]], col_labels)
            for i in range(scores.shape[0])]


def green_channel(warped: np.ndarray) -> np.ndarray:
    """Pencil-mark intensity map. The form's red ink (bubble outlines and
    printed digit/letter labels) reflects strongly in the RED channel,
    keeping it bright (~200-250). White paper is bright too (~250).
    Pencil graphite, by contrast, absorbs all wavelengths and shows as
    LOW intensity in every channel — including red. So using the red
    channel as the scoring image isolates pencil marks from both white
    paper AND printed red-ink labels.

    Combined with CLAHE for local contrast stretching, faint pencil on
    very faint scans (raw 40-intensity drop) becomes a clear score
    above FILL_THRESH while empty bubbles (printed letters in red ink)
    stay below threshold.

    The legacy name `green_channel` is kept for backward compatibility
    with annotate_omr.py and others; the implementation uses red.
    """
    r = warped[:, :, 2]  # BGR → red is index 2
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    return clahe.apply(r)


def refit_template(warped: np.ndarray, template: dict,
                   stripe_canon_ys: list[float] | None = None) -> dict:
    """Re-fit each region against this specific scan, mirroring
    build_template's logic so per-scan template generation matches what
    calibration would produce.

    Y-position priority:
      1. Stripe-anchored if stripe_canon_ys covers the region's stripes.
      2. Shared-Y from a sibling region with more bubble centers.
      3. Per-region bubble grid fit.
    X positions always come from per-region bubble clustering."""
    from build_template import SHARED_Y_GROUPS, snap_ys_to_stripes
    centers = detect_bubbles(warped)
    raw = {}
    for name, region in template["regions"].items():
        x0, y0, x1, y1 = region["bbox"]
        m = ((centers[:, 0] >= x0) & (centers[:, 0] <= x1) &
             (centers[:, 1] >= y0) & (centers[:, 1] <= y1))
        sub = centers[m]
        xs, ys_fit = fit_grid(sub, region["rows"], region["cols"])

        ys = None
        if (stripe_canon_ys is not None and ys_fit is not None
                and len(ys_fit) == region["rows"]):
            ys = snap_ys_to_stripes(ys_fit, stripe_canon_ys)

        raw[name] = {"region": region, "xs": xs, "ys_fit": ys_fit,
                     "ys": ys, "n_centers": len(sub)}

    # Shared-Y donation
    for _g, members in SHARED_Y_GROUPS.items():
        members = [m for m in members if m in raw]
        if not members:
            continue
        donor = max(members, key=lambda n: raw[n]["n_centers"])
        donor_ys = raw[donor]["ys"] or raw[donor]["ys_fit"]
        donor_rows = raw[donor]["region"]["rows"]
        if donor_ys is None or len(donor_ys) != donor_rows:
            continue
        for n in members:
            if raw[n]["ys"] is not None:
                continue
            if raw[n]["region"]["rows"] == donor_rows:
                raw[n]["ys"] = donor_ys

    fitted = {"canonical": template["canonical"], "regions": {}}
    for name, info in raw.items():
        region = info["region"]
        new_region = dict(region)
        if info["xs"] is not None and len(info["xs"]) == region["cols"]:
            new_region["xs"] = info["xs"]
        ys = info["ys"] or info["ys_fit"]
        if ys is not None and len(ys) == region["rows"]:
            new_region["ys"] = ys
        fitted["regions"][name] = new_region
    return fitted


def read_omr(warped: np.ndarray, template: dict, refit: bool = False) -> dict:
    if refit:
        template = refit_template(warped, template)
    green = green_channel(warped)
    out: dict[str, Any] = {}
    for name, region in template["regions"].items():
        scores = score_grid(green, region)
        kind = region["kind"]
        rl, cl = region["row_labels"], region["col_labels"]
        if kind == "answers":
            out[name] = decode_answers(scores, rl, cl)
        elif kind in ("letters", "digits"):
            cols = decode_column_winner(scores, rl, cl, kind=kind)
            out[name] = "".join(c if c else "_" for c in cols).rstrip("_") or None
        elif kind == "single_row":
            out[name] = decode_single_row(scores, rl, cl)
        elif kind == "codes":
            out[name] = decode_codes(scores, rl, cl)
    # Flatten all answer sub-blocks into a single dict {"1": "A", "2": "C", ...}
    # Keys are strings so the in-memory shape matches the JSON shape (JSON
    # doesn't support int keys; using strings everywhere keeps round-tripping
    # and downstream code consistent).
    answers = {}
    for k in list(out.keys()):
        if k.startswith("answers_q"):
            for q, choice in out.pop(k).items():
                answers[int(q)] = choice
    out["answers"] = {str(q): answers[q] for q in sorted(answers)}
    return out


def annotate(warped: np.ndarray, template: dict, decoded: dict) -> np.ndarray:
    """Render filled bubbles on a copy of the warped image."""
    vis = warped.copy()
    green = green_channel(warped)
    for name, region in template["regions"].items():
        if region["xs"] is None:
            continue
        for r, y in enumerate(region["ys"]):
            for c, x in enumerate(region["xs"]):
                s = score_bubble(green, x, y)
                color = (0, 255, 0) if s >= FILL_THRESH else (180, 180, 180)
                cv2.circle(vis, (int(x), int(y)), 9, color, 1 if s < FILL_THRESH else 2)
    return vis


# ---------- Grid-based reader (fiducial-anchored, uniform pitch) ----------
#
# UF Form LR1 column layout (form-unit indices, 1-indexed from leftmost):
#   k = 1..12   LAST_NAME
#   k = 13      FI
#   k = 14      MI
#   k = 17..21  left answer block A..E (Q1-40)
#   k = 24..28  right answer block / TEST_FORM_CODE A..E (Q41-80)
#
# Stripe row layout (i = stripe index, 0-based):
#   i = 2          TEST_FORM_CODE single row
#   i = 4..30      LAST_NAME / FI / MI: 27 rows = WRITE row + A..Z (1+26)
#   i = 4..13      Q1-Q10
#   i = 15..24     Q11-Q20
#   i = 26..35     Q21-Q30
#   i = 37..46     Q31-Q40
#   (right answers Q41-50/51-60/61-70/71-80 mirror the left rows at
#    cols 24..28)
#   i = 35..44     UF_ID (8 cols at k=1..8) and SECTION (4 cols at
#                   k=9, 11, 12, 13 — non-contiguous on the form)

ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")

# Region layout: (col_ks, row_is, row_labels, kind)
# col_ks = list of form-unit column indices
# row_is = list of stripe indices for each row of the region
GRID_LAYOUT = {
    "test_form_code":  (list(range(24, 29)), [2],          [""],            "single_row"),
    "last_name":       (list(range(1, 13)),  list(range(4, 31)), [" "] + ALPHABET, "letters"),
    "first_initial":   ([13],                list(range(4, 31)), [" "] + ALPHABET, "letters"),
    "middle_initial":  ([14],                list(range(4, 31)), [" "] + ALPHABET, "letters"),
    "answers_q01_10":  (list(range(17, 22)), list(range(4, 14)),  [str(q) for q in range(1, 11)],  "answers"),
    "answers_q11_20":  (list(range(17, 22)), list(range(15, 25)), [str(q) for q in range(11, 21)], "answers"),
    "answers_q21_30":  (list(range(17, 22)), list(range(26, 36)), [str(q) for q in range(21, 31)], "answers"),
    "answers_q31_40":  (list(range(17, 22)), list(range(37, 47)), [str(q) for q in range(31, 41)], "answers"),
    "answers_q41_50":  (list(range(24, 29)), list(range(4, 14)),  [str(q) for q in range(41, 51)], "answers"),
    "answers_q51_60":  (list(range(24, 29)), list(range(15, 25)), [str(q) for q in range(51, 61)], "answers"),
    "answers_q61_70":  (list(range(24, 29)), list(range(26, 36)), [str(q) for q in range(61, 71)], "answers"),
    "answers_q71_80":  (list(range(24, 29)), list(range(37, 47)), [str(q) for q in range(71, 81)], "answers"),
    # UF_ID and SECTION digit rows: digits 0-9 top-to-bottom, spanning
    # stripe i=34..43. The earlier i=35..44 range was off by one and
    # missed the digit-0 row (e.g., an example UFID col = 0 was at
    # i=34, outside the range, leaving an underscore).
    "uf_id":           (list(range(1, 9)),   list(range(34, 44)), DIGITS, "digits"),
    "section":         ([9, 11, 12, 13],     list(range(34, 44)), DIGITS, "digits"),
}

# Column labels per region (only for letter/digit regions where output
# is read as one symbol per column).
GRID_COL_LABELS = {
    "test_form_code":  ["A", "B", "C", "D", "E"],
    "last_name":       [str(c) for c in range(1, 13)],
    "first_initial":   ["FI"],
    "middle_initial":  ["MI"],
    "answers_q01_10":  ["A", "B", "C", "D", "E"],
    "answers_q11_20":  ["A", "B", "C", "D", "E"],
    "answers_q21_30":  ["A", "B", "C", "D", "E"],
    "answers_q31_40":  ["A", "B", "C", "D", "E"],
    "answers_q41_50":  ["A", "B", "C", "D", "E"],
    "answers_q51_60":  ["A", "B", "C", "D", "E"],
    "answers_q61_70":  ["A", "B", "C", "D", "E"],
    "answers_q71_80":  ["A", "B", "C", "D", "E"],
    "uf_id":           [str(c) for c in range(1, 9)],
    "section":         ["1", "2", "3", "4"],
}


def _compute_grid_params(warped: np.ndarray
                         ) -> tuple[float, float, float, float, float] | None:
    """Compute (c_0, sp, h_slope, col_intercept, col_slope) for the
    fiducial-anchored uniform-pitch grid. Returns None on failure
    (missing fiducials or stripes).

    c_0, sp, h_slope: stripe grid (y(i, x) = c_0 + i*sp + h_slope*x)
    col_intercept, col_slope: column grid (x(k) = col_intercept + col_slope*k)
    """
    from annotate_omr import (_detect_top_fiducials, _fit_stripe_grid_params)
    fiducial_xs = _detect_top_fiducials(warped)
    if len(fiducial_xs) < 4:
        return None
    ks_fid = np.array([2.0, 3.0, 21.0, 23.0])
    xs_fid = np.array([float(x) for x in fiducial_xs[:4]])
    A = np.vstack([ks_fid, np.ones(4)]).T
    col_slope, col_intercept = np.linalg.lstsq(A, xs_fid, rcond=None)[0]
    gp = _fit_stripe_grid_params(warped)
    if gp is None:
        return None
    c_0, sp, h_slope = gp
    return (float(c_0), float(sp), float(h_slope),
             float(col_intercept), float(col_slope))


def _intersection_xy(k: int, i: int, c_0: float, sp: float, h_slope: float,
                      col_intercept: float, col_slope: float
                      ) -> tuple[float, float]:
    """Canonical (x, y) where the column-k vertical (perpendicular to
    horizontals) crosses the row-i horizontal."""
    x0 = col_intercept + col_slope * k
    s2_p1 = 1.0 + h_slope * h_slope
    i_sp_norm = i * sp / s2_p1
    y_int = c_0 + i_sp_norm + h_slope * x0
    x_int = x0 - h_slope * i_sp_norm
    return float(x_int), float(y_int)


def read_omr_grid(warped: np.ndarray) -> dict:
    """Decode all OMR fields using the fiducial-anchored uniform grid.
    Each bubble is sampled at the EXACT perpendicular intersection of
    the column line through form-unit k and the stripe line at index
    i — same positions as the green-ring overlay in annotate_omr.

    Returns a flat dict with keys: last_name, first_initial,
    middle_initial, uf_id, section, test_form_code, answers (dict
    {"1": "A", ...}). Missing fields are None.
    """
    params = _compute_grid_params(warped)
    if params is None:
        return {"error": "fiducial/stripe detection failed"}
    c_0, sp, h_slope, col_intercept, col_slope = params
    green = green_channel(warped)

    out: dict[str, Any] = {}
    for name, (col_ks, row_is, row_labels, kind) in GRID_LAYOUT.items():
        col_labels = GRID_COL_LABELS[name]
        scores = np.zeros((len(row_is), len(col_ks)))
        for r, i in enumerate(row_is):
            for c, k in enumerate(col_ks):
                x_int, y_int = _intersection_xy(
                    k, i, c_0, sp, h_slope, col_intercept, col_slope)
                scores[r, c] = score_bubble(green, x_int, y_int)

        if kind == "answers":
            out[name] = decode_answers(scores, row_labels, col_labels)
        elif kind in ("letters", "digits"):
            cols = decode_column_winner(scores, row_labels, col_labels,
                                          kind=kind)
            out[name] = ("".join(c if c else "_" for c in cols).rstrip("_")
                          or None)
        elif kind == "single_row":
            out[name] = decode_single_row(scores, row_labels, col_labels)

    # Flatten answers
    answers: dict[int, Any] = {}
    for k in list(out.keys()):
        if k.startswith("answers_q"):
            for q, choice in out.pop(k).items():
                answers[int(q)] = choice
    out["answers"] = {str(q): answers[q] for q in sorted(answers)}
    return out


def process_pdf_grid(pdf_path: str, dpi: int = 300) -> dict:
    """Grid-based pipeline: find OMR page, warp, decode via grid."""
    from utils import find_omr_pages
    candidates = find_omr_pages(pdf_path)
    if not candidates:
        return {"file": pdf_path, "error": "no OMR page detected"}
    best = None
    for page_idx, score in candidates:
        img = render_page(pdf_path, page_idx, dpi=dpi)
        warped = warp_omr(img)
        if warped is None:
            continue
        decoded = read_omr_grid(warped)
        if decoded.get("last_name") or decoded.get("uf_id"):
            best = (page_idx, score, warped, decoded)
            break
        if best is None:
            best = (page_idx, score, warped, decoded)
    if best is None:
        return {"file": pdf_path, "error": "no readable OMR page"}
    page_idx, score, warped, decoded = best
    return {
        "file": pdf_path,
        "omr_page": page_idx,
        "red_score": round(score, 4),
        **decoded,
    }


def process_pdf(pdf_path: str, template: dict, dpi: int = 300,
                annotate_to: str | None = None) -> dict:
    """Find the OMR page in pdf_path, read it, return decoded result."""
    from utils import find_omr_pages
    candidates = find_omr_pages(pdf_path)
    if not candidates:
        return {"file": pdf_path, "error": "no OMR page detected"}
    # Front side has UF_ID; if multiple red pages, pick the one whose warp
    # successfully reads a UF_ID + LAST_NAME (front, not back).
    best = None
    for page_idx, score in candidates:
        img = render_page(pdf_path, page_idx, dpi=dpi)
        warped = warp_omr(img)
        if warped is None:
            continue
        # Detect this scan's own timing-mark stripes and re-anchor row Ys.
        # Only use stripes when we get roughly the expected count at
        # uniform spacing — otherwise the noise from over- or under-
        # detection would corrupt the row positions.
        quad = find_form_quad(img)
        form_top_y = None
        if quad is not None:
            from utils import order_quad
            ordered = order_quad(quad)
            form_top_y = float(min(ordered[0][1], ordered[1][1]))
        stripes_raw = (detect_timing_marks(img, form_top_y=form_top_y)
                       if quad is not None else [])
        # detect_timing_marks now returns exactly EXPECTED_STRIPES positions
        # (or empty if the fit failed), so no further filtering is needed.
        stripe_canon = (map_stripes_to_canonical(stripes_raw, quad)
                        if quad is not None and len(stripes_raw) == EXPECTED_STRIPES
                        else None)
        per_scan_tpl = refit_template(warped, template, stripe_canon)
        decoded = read_omr(warped, per_scan_tpl, refit=False)
        if decoded.get("last_name") or decoded.get("uf_id"):
            best = (page_idx, score, warped, decoded, per_scan_tpl)
            break
        if best is None:
            best = (page_idx, score, warped, decoded, per_scan_tpl)
    if best is None:
        return {"file": pdf_path, "error": "no readable OMR page"}
    page_idx, score, warped, decoded, used_tpl = best
    if annotate_to:
        cv2.imwrite(annotate_to, annotate(warped, used_tpl, decoded))
    return {
        "file": pdf_path,
        "omr_page": page_idx,
        "red_score": round(score, 4),
        **decoded,
    }
