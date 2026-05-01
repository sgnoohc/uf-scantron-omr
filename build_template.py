"""Build template.json by auto-fitting bubble grids on a sample OMR.

Strategy:
  1. Render and warp the sample to canonical 1200x1800.
  2. Detect all bubble centers (HoughCircles on the red mask).
  3. For each named region, take centers inside a rough bbox and auto-fit a
     regular grid by clustering Y values (rows) and X values (columns).
  4. Save the inferred grid + labels to template.json.
  5. Save an overlay PNG for visual verification.

Region bboxes are tuned to the UF Form LR1 scantron after deskew.

Run:
    python3 build_template.py [sample.pdf] [page_idx]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import cv2
import numpy as np

from utils import (render_page, warp_omr, red_mask, find_form_quad,
                   detect_timing_marks, map_stripes_to_canonical,
                   CANON_W, CANON_H)

HERE = Path(__file__).resolve().parent

ALPHA = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 26
DIGITS = list("0123456789")
CHOICE = list("ABCDE")

# Each region: rough bbox (x0, y0, x1, y1) in canonical 1200x1800 coords +
# expected (rows, cols) layout + labels. The bbox just has to contain only
# this region's circles. Auto-fit infers exact centers.
def _ans(qfrom, qto, x0, x1, y0, y1, stripes=None):
    return dict(bbox=(x0, y0, x1, y1), rows=10, cols=5,
                row_labels=[str(q) for q in range(qfrom, qto + 1)],
                col_labels=CHOICE, kind="answers", stripes=stripes)


REGIONS = {
    # 27-row layout: 1 writing-cell row at top + 26 alphabet rows. Including
    # the writing row in the fit prevents off-by-one shifts when the
    # writing-cell bubbles get detected by Hough — otherwise the fit picks
    # those up as "row 0" and shifts every alphabet row down by 1.
    "last_name":      dict(bbox=(20,  150,  540, 1180), rows=27, cols=12,
                           row_labels=[" "] + ALPHA,
                           col_labels=[str(i+1) for i in range(12)],
                           kind="letters"),
    "first_initial":  dict(bbox=(540, 150,  590, 1180), rows=27, cols=1,
                           row_labels=[" "] + ALPHA, col_labels=["FI"],
                           kind="letters"),
    "middle_initial": dict(bbox=(590, 150,  640, 1180), rows=27, cols=1,
                           row_labels=[" "] + ALPHA, col_labels=["MI"],
                           kind="letters"),
    "test_form_code": dict(bbox=(985,  55, 1200,  110), rows=1,  cols=5,
                           row_labels=[""], col_labels=CHOICE, kind="single_row"),
    "answers_q01_10": _ans(1, 10,  680, 940,  140,  520),
    "answers_q11_20": _ans(11, 20, 680, 940,  555,  935),
    "answers_q21_30": _ans(21, 30, 680, 940,  970, 1350),
    "answers_q31_40": _ans(31, 40, 680, 940, 1385, 1790),
    "answers_q41_50": _ans(41, 50, 940, 1200, 140,  520),
    "answers_q51_60": _ans(51, 60, 940, 1200, 555,  935),
    "answers_q61_70": _ans(61, 70, 940, 1200, 970, 1350),
    "answers_q71_80": _ans(71, 80, 940, 1200, 1385, 1790),
    # UF_ID / SECTION: 10 rows of digits 0-9. Writing row excluded — its
    # bubbles aren't always reliably detected and including them shifts
    # the fit by 1 row.
    "uf_id":          dict(bbox=(20, 1310,  395, 1680), rows=10, cols=8,
                           row_labels=DIGITS,
                           col_labels=[str(i+1) for i in range(8)],
                           kind="digits"),
    "section":        dict(bbox=(440, 1310, 660, 1680), rows=10, cols=4,
                           row_labels=DIGITS,
                           col_labels=[str(i+1) for i in range(4)],
                           kind="digits"),
    "special_codes":  dict(bbox=(195, 1690, 660, 1795), rows=2, cols=10,
                           row_labels=["SP1", "SP2"], col_labels=DIGITS,
                           kind="codes"),
}


def detect_bubbles(warped: np.ndarray) -> np.ndarray:
    mask = red_mask(warped, thresh=25)
    gray = cv2.bitwise_not(mask)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
        param1=100, param2=12, minRadius=8, maxRadius=14,
    )
    if circles is None:
        return np.zeros((0, 2))
    return circles[0, :, :2]


def cluster_axis(values: np.ndarray, gap_thresh: float = 14.0):
    """Cluster a 1D array of coords into groups separated by gaps > gap_thresh.
    Returns list of (center, count) sorted by ascending position."""
    if len(values) == 0:
        return []
    v = np.sort(values)
    groups = [[v[0]]]
    for x in v[1:]:
        if x - groups[-1][-1] < gap_thresh:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [(float(np.mean(g)), len(g)) for g in groups]


def select_clusters(clusters: list, expected: int) -> list:
    """Pick the `expected` largest clusters by count, then sort by position."""
    if not clusters:
        return []
    if len(clusters) == expected:
        return [c for c, _ in clusters]
    if len(clusters) > expected:
        top = sorted(clusters, key=lambda c: -c[1])[:expected]
        return sorted(c for c, _ in top)
    positions = sorted(c for c, _ in clusters)
    return np.linspace(positions[0], positions[-1], expected).tolist()


def _is_uniformly_spaced(positions: list[float], tol: float = 0.15) -> bool:
    """Check that adjacent positions all share roughly the same spacing."""
    if len(positions) < 3:
        return True
    diffs = np.diff(positions)
    median = float(np.median(diffs))
    if median <= 0:
        return False
    return bool(np.all(np.abs(diffs - median) <= tol * median))


def _fit_uniform_grid(positions: list[float], expected: int,
                      tol: float = 0.20) -> list[float] | None:
    """Find best-fitting regular grid of `expected` positions covering
    the candidate `positions`. Returns None if no good fit found."""
    if len(positions) < max(3, expected // 2):
        return None
    arr = np.array(sorted(positions))
    diffs = np.diff(arr)
    if len(diffs) == 0:
        return None
    median_sp = float(np.median(diffs))
    if median_sp <= 0:
        return None
    best_inliers = 0
    best_a = best_b = None
    for anchor_pos in arr:
        for sp_off in np.arange(-2.0, 2.5, 0.5):
            sp = median_sp + sp_off
            if sp <= 0:
                continue
            indices = (arr - anchor_pos) / sp
            rounded = np.round(indices).astype(int)
            err = np.abs(indices - rounded)
            mask = err < tol
            n = int(mask.sum())
            if n < max(3, expected // 2):
                continue
            if n > best_inliers:
                best_inliers = n
                A = np.vstack([rounded[mask], np.ones(n)]).T
                slope, intercept = np.linalg.lstsq(A, arr[mask],
                                                    rcond=None)[0]
                best_a = float(intercept)
                best_b = float(slope)
    if best_a is None:
        return None
    # Anchor index 0 to whichever position gives the smallest extrapolation
    # past the detected range.
    min_idx = int(np.round((arr[0] - best_a) / best_b))
    intercept = best_a + min_idx * best_b
    return [intercept + i * best_b for i in range(expected)]


def fit_grid(centers: np.ndarray, rows: int, cols: int):
    """Cluster centers into rows and cols. Returns (xs[cols], ys[rows]).
    If counts mismatch, drop smallest clusters or linspace-fill."""
    if len(centers) == 0:
        return None, None
    y_clusters = cluster_axis(centers[:, 1])
    x_clusters = cluster_axis(centers[:, 0])
    return select_clusters(x_clusters, cols), select_clusters(y_clusters, rows)


def snap_ys_to_stripes(ys_fit: list[float], stripe_canon_ys: list[float],
                       max_dist: float = 25.0) -> list[float] | None:
    """Snap each fitted Y position to the nearest timing-stripe Y.

    Stripes are the form's printed row markers — one per visible row —
    so they're the most precise row anchors available. We use the bubble
    cluster fit only to choose WHICH stripes (i.e., which row indices),
    then return the stripes' canonical Ys for sub-pixel precision.

    Returns None if any fitted Y has no nearby stripe (sign of bad fit).
    """
    if not ys_fit or not stripe_canon_ys:
        return None
    arr = np.asarray(stripe_canon_ys)
    snapped = []
    used = set()
    for y in ys_fit:
        diffs = np.abs(arr - y)
        idx = int(np.argmin(diffs))
        if diffs[idx] > max_dist:
            return None
        if idx in used:
            return None  # two fitted rows snapped to same stripe = bad fit
        used.add(idx)
        snapped.append(float(arr[idx]))
    # Ensure monotonic (sorted ascending)
    if any(snapped[i] >= snapped[i + 1] for i in range(len(snapped) - 1)):
        return None
    return snapped


# Regions whose rows are physically aligned in the form share Y positions —
# fit Y from the region with the MOST bubble data, then reuse those ys for
# the smaller regions. This avoids drift from sparse-data fits (e.g. FI/MI
# have only 27 centers vs LAST_NAME's 324, so FI/MI's row spacing is noisier
# and can shift by a full row across the alphabet).
SHARED_Y_GROUPS = {
    "alphabet": ["last_name", "first_initial", "middle_initial"],
    "digits":   ["uf_id", "section"],
}


def build_template(warped: np.ndarray, stripe_canon_ys: list[float] | None = None):
    """Build a template from a warped sample.

    Y-position priority for each region:
      1. Stripe-anchored (best): if `stripe_canon_ys` is supplied AND covers
         the stripe indices declared in REGIONS[name]["stripes"], use the
         exact canonical Y of each timing-mark stripe.
      2. Shared-Y: regions in the same SHARED_Y_GROUPS row group reuse the
         Y positions of the donor with the most detected bubble centers.
         Catches the case where FI/MI's sparse 27-center fit drifts by
         a full row across the alphabet relative to LAST_NAME's 324-center
         fit.
      3. Per-region grid fit (fallback).
    """
    centers = detect_bubbles(warped)
    print(f"detected {len(centers)} bubble centers")
    template = {"canonical": {"w": CANON_W, "h": CANON_H}, "regions": {}}

    raw = {}
    for name, spec in REGIONS.items():
        x0, y0, x1, y1 = spec["bbox"]
        m = ((centers[:, 0] >= x0) & (centers[:, 0] <= x1) &
             (centers[:, 1] >= y0) & (centers[:, 1] <= y1))
        sub = centers[m]
        xs, fitted_ys = fit_grid(sub, spec["rows"], spec["cols"])

        # Step 1: snap fitted ys to the nearest detected stripe Ys.
        # This gives sub-pixel precision and is robust to per-scan warp
        # variation. Falls back to fitted_ys if snapping fails.
        ys = None
        ys_source = None
        if stripe_canon_ys is not None and fitted_ys is not None and \
                len(fitted_ys) == spec["rows"]:
            snapped = snap_ys_to_stripes(fitted_ys, stripe_canon_ys)
            if snapped is not None:
                ys = snapped
                ys_source = "stripes"

        raw[name] = {
            "spec": spec, "xs": xs, "fitted_ys": fitted_ys,
            "n_centers": len(sub),
            "ys": ys, "ys_source": ys_source,
        }

    # Step 2: shared-Y donation for regions still without anchored ys.
    for _g, members in SHARED_Y_GROUPS.items():
        donor = max(members, key=lambda n: raw[n]["n_centers"])
        donor_info = raw[donor]
        donor_ys = donor_info["ys"] if donor_info["ys"] is not None else donor_info["fitted_ys"]
        donor_rows = donor_info["spec"]["rows"]
        if donor_ys is None or len(donor_ys) != donor_rows:
            continue
        for n in members:
            if raw[n]["ys"] is not None:
                continue  # already anchored (e.g. via stripes)
            if raw[n]["spec"]["rows"] == donor_rows:
                raw[n]["ys"] = donor_ys
                raw[n]["ys_source"] = "shared(" + donor + ")"

    # Step 3: assemble template, fall back to per-region fit for the rest.
    for name, info in raw.items():
        spec = info["spec"]
        xs = info["xs"]
        ys = info["ys"] or info["fitted_ys"]
        ys_source = info["ys_source"] or "fit"
        n_xs = 0 if xs is None else len(xs)
        n_ys = 0 if ys is None else len(ys)
        ok = n_xs == spec["cols"] and n_ys == spec["rows"]
        flag = "OK" if ok else "MISMATCH"
        print(f"  [{flag}] {name}: {info_str(info['n_centers'], n_xs, n_ys, spec, ys_source)}")
        template["regions"][name] = {
            "kind": spec["kind"], "bbox": list(spec["bbox"]),
            "rows": spec["rows"], "cols": spec["cols"],
            "row_labels": spec["row_labels"], "col_labels": spec["col_labels"],
            "xs": xs, "ys": ys,
            "detected": int(info["n_centers"]), "ok": bool(ok),
            "ys_source": ys_source,
        }
    return template, centers


def info_str(n_sub, n_xs, n_ys, spec, ys_source):
    return (f"detected {n_sub} centers, "
            f"fitted {n_xs}x{n_ys} grid "
            f"(expected {spec['cols']}x{spec['rows']}; ys={ys_source})")


def overlay(warped: np.ndarray, template: dict, centers: np.ndarray | None = None) -> np.ndarray:
    vis = warped.copy()
    if centers is not None:
        for x, y in centers:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 200, 255), -1)
    for name, r in template["regions"].items():
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 1)
        if r["xs"] is None or r["ys"] is None:
            continue
        for y in r["ys"]:
            for x in r["xs"]:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), 1)
    return vis


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    img = render_page(pdf, page, dpi=300)
    warped = warp_omr(img)
    if warped is None:
        sys.exit("failed to find form quad")
    cv2.imwrite(str(HERE / "_warped.png"), warped)
    quad = find_form_quad(img)
    form_top_y = None
    if quad is not None:
        from utils import order_quad
        ordered = order_quad(quad)
        form_top_y = float(min(ordered[0][1], ordered[1][1]))
    stripes_orig = detect_timing_marks(img, form_top_y=form_top_y)
    stripes_canon = (map_stripes_to_canonical(stripes_orig, quad)
                     if quad is not None and stripes_orig else None)
    print(f"detected {len(stripes_orig) if stripes_orig else 0} timing stripes")
    template, centers = build_template(warped, stripes_canon)
    out = HERE / "template.json"
    with open(out, "w") as f:
        json.dump(template, f, indent=2)
    print(f"wrote {out}")
    vis = overlay(warped, template, centers)
    cv2.imwrite(str(HERE / "template_overlay.png"), vis)
    print(f"wrote {HERE / 'template_overlay.png'}")


if __name__ == "__main__":
    main()
