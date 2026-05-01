"""Render annotated debug PDFs showing OMR detection results.

For each PDF in the input directory, this script:
  1. Runs the OMR pipeline (find page, warp, refit template, decode).
  2. Draws every template grid position as a small gray dot, so you can
     see whether the grid is aligned with the printed bubbles.
  3. Outlines bubbles classified as "filled" with a thick green ring.
  4. Prints row labels (A–Z / 0–9) and column numbers at the margins.
  5. Adds a sidebar with the decoded fields (name, UF ID, answers,
     etc.) so you can spot misreads at a glance.

Two output modes:
  * `--combined` (default): one multi-page PDF with one page per scan,
    sorted alphabetically. Easy to flip through and spot errors.
  * `--per-pdf`: one PDF per input scan, written to the output directory.

Usage:
    python3 annotate_omr.py <input_dir> <output_path>
                            [--combined | --per-pdf]
                            [--ext .pdf]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

from reader import (
    load_template, read_omr, refit_template,
    score_bubble, green_channel, FILL_THRESH,
    read_omr_grid, GRID_LAYOUT,
)
from utils import (
    warp_omr, render_page, find_form_quad, find_omr_pages,
    detect_timing_marks, map_stripes_to_canonical, EXPECTED_STRIPES,
    order_quad, _stripe_candidates, _stripe_candidates_centroid,
    canonical_to_original, CANON_W, CANON_H, red_mask,
)

# Visual tuning
GRID_DOT_COLOR = (200, 200, 200)   # BGR — pale gray for unfilled grid markers
GRID_DOT_RADIUS = 2
STRIPE_LINE_COLOR = (255, 100, 50)   # BGR — strong blue for stripe-aligned rows
STRIPE_LINE_THICK = 1
COL_LINE_COLOR = (220, 100, 0)       # BGR — strong blue for column guides
COL_LINE_THICK = 2
FILLED_RING_COLOR = (0, 200, 0)    # green
FILLED_RING_RADIUS = 12
FILLED_RING_THICK = 3
LABEL_COLOR = (0, 0, 200)          # red text for row/col labels
BBOX_COLOR = (180, 100, 100)       # purple for region bbox
ANNOT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _detect_red_bubble_centers(warped: np.ndarray) -> np.ndarray:
    """Detect every printed red bubble outline center on the canonical
    warp via HoughCircles on the red mask. Returns an (N, 2) float32
    array of (x, y) positions — typically ~900 anchors on UF Form LR1.
    """
    mask = red_mask(warped, thresh=25)
    gray = cv2.bitwise_not(mask)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
        param1=100, param2=12, minRadius=8, maxRadius=14,
    )
    if circles is None:
        return np.zeros((0, 2), dtype=np.float32)
    return circles[0, :, :2].astype(np.float32)


def _detect_red_bubble_circles(warped: np.ndarray) -> np.ndarray:
    """Like `_detect_red_bubble_centers` but returns (N, 3) — (x, y, r)
    for each bubble. Used by the sanity-check overlay that fills each
    detected red outline with black so the user can SEE which bubbles
    were identified."""
    mask = red_mask(warped, thresh=25)
    gray = cv2.bitwise_not(mask)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
        param1=100, param2=12, minRadius=8, maxRadius=14,
    )
    if circles is None:
        return np.zeros((0, 3), dtype=np.float32)
    return circles[0, :, :3].astype(np.float32)


def _cluster_axis(values: np.ndarray, gap_thresh: float = 14.0,
                   min_count: int = 5) -> list[float]:
    """Cluster a 1D coord array into groups separated by gaps >
    gap_thresh, return cluster centers (mean) for clusters with at
    least min_count members. Filters single-bubble noise."""
    if len(values) == 0:
        return []
    v = np.sort(values)
    groups: list[list[float]] = [[float(v[0])]]
    for x in v[1:]:
        if x - groups[-1][-1] < gap_thresh:
            groups[-1].append(float(x))
        else:
            groups.append([float(x)])
    return [float(np.mean(g)) for g in groups if len(g) >= min_count]


def _find_axis_peaks(values: np.ndarray, bin_width: float = 3.0,
                      smooth: int = 5, min_distance: float = 25.0,
                      min_height_frac: float = 0.15) -> list[float]:
    """Find peak X (or Y) positions in a 1D coordinate array via
    histogram + local-maximum detection. Robust to noisy/spurious
    detections in a way 1D-gap clustering is not: a few stray points
    add to bin counts but don't create false peaks unless they cluster.

    bin_width: histogram bin width in canonical px.
    smooth:    box-filter window size (in bins) applied to the
                histogram before peak finding. Smooths out single-bin
                noise.
    min_distance: minimum spacing between peaks (px). Bubbles are ~38
                px apart on UF Form LR1; 25 keeps adjacent columns
                distinct while suppressing detections from the same
                bubble.
    min_height_frac: peaks below this fraction of the global max
                count are discarded (filters background noise).
    """
    if len(values) == 0:
        return []
    v_min, v_max = float(np.min(values)), float(np.max(values))
    if v_max - v_min < bin_width:
        return [float(np.mean(values))]
    bins = np.arange(v_min, v_max + bin_width, bin_width)
    hist, edges = np.histogram(values, bins=bins)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float64) / smooth
        smoothed = np.convolve(hist.astype(np.float64), kernel, mode="same")
    else:
        smoothed = hist.astype(np.float64)
    if smoothed.max() == 0:
        return []
    height_thresh = float(min_height_frac * smoothed.max())
    min_dist_bins = max(1, int(round(min_distance / bin_width)))

    # Greedy peak picking: take the highest bin, suppress neighbors
    # within min_dist_bins, repeat until no peaks above threshold.
    available = smoothed.copy()
    peaks_idx: list[int] = []
    while True:
        idx = int(np.argmax(available))
        if available[idx] < height_thresh:
            break
        peaks_idx.append(idx)
        lo = max(0, idx - min_dist_bins)
        hi = min(len(available), idx + min_dist_bins + 1)
        available[lo:hi] = -1.0
    peaks_idx.sort()
    centers = [float((edges[i] + edges[i + 1]) / 2.0) for i in peaks_idx]
    return centers


def _detect_uniform_columns(
    warped: np.ndarray, slope: float = 0.0,
) -> tuple[list[float], float, float]:
    """Detect bubble-column X positions (in canonical) via red-bubble
    clustering, then fit a single uniform pitch across the whole page.

    Returns (column_xs, pitch, intercept). The form uses ONE uniform
    column pitch across LAST_NAME / FI / MI / UF_ID / SECTION /
    answers / TEST_FORM_CODE — so we cluster every detected red bubble
    center on the X axis and fit `x = intercept + k * pitch` to the
    discovered column centers, where k is each column's integer index
    (zero-based, with the leftmost column being k=0).

    `slope` is the horizontal-grid tilt (dy/dx) from the stripe-grid
    fit. Centers are DESKEWED before clustering: a vertical column at
    canonical x=x0 has its bubble centers drift along x = x0 - slope*y
    (perpendicular direction). Deskewing — adding `slope * y` back to
    each x — collapses each column into a tight vertical line and
    enables clean 1D gap-based clustering. Without this, even a
    half-degree of tilt over 1800 px of canonical height smears column
    X positions by ~15 px (a full column pitch is ~38 px), so simple
    1D clustering merges adjacent columns.

    Empty `column_xs` ⇒ detection failed (caller should fall back to
    fiducial-based columns).
    """
    pts = _detect_red_bubble_centers(warped)
    if len(pts) < 50:
        return [], 0.0, 0.0
    # Deskew so columns are truly vertical.
    xs_deskew = pts[:, 0] + slope * pts[:, 1]
    # Histogram peak detection: more robust than gap-based clustering
    # when individual columns have wide X-spread (residual tilt) or
    # when noise points bridge inter-column gaps. min_distance=25 keeps
    # peaks at least one half-pitch apart (real pitch is ~38 px).
    raw_centers = _find_axis_peaks(xs_deskew, bin_width=3.0, smooth=5,
                                     min_distance=25.0, min_height_frac=0.10)
    if len(raw_centers) < 4:
        return [], 0.0, 0.0
    # Fit uniform pitch: each detected center sits at integer multiples
    # of pitch above the leftmost. Estimate pitch from neighbor diffs,
    # then assign each center its k = round((x - x0) / pitch).
    diffs = np.diff(np.array(raw_centers))
    pitch_est = float(np.median(diffs))
    x0 = float(raw_centers[0])
    ks = [int(round((c - x0) / pitch_est)) for c in raw_centers]
    # Refit pitch and intercept by least squares using assigned ks.
    A = np.vstack([np.array(ks, dtype=np.float64), np.ones(len(ks))]).T
    pitch, intercept = np.linalg.lstsq(A, np.array(raw_centers,
                                                     dtype=np.float64),
                                         rcond=None)[0]
    pitch = float(pitch)
    intercept = float(intercept)
    # Generate the FULL column set: every k from 0 to max_k. This adds
    # back columns that had no bubble-centers (e.g., the gap between
    # LAST_NAME and the answer block — those k's still belong to the
    # uniform grid and may cross labels/text we want to inspect).
    max_k = max(ks)
    column_xs = [intercept + k * pitch for k in range(max_k + 1)]
    return column_xs, pitch, intercept


def _detect_top_fiducial_anchors(warped: np.ndarray) -> list[tuple[float, float]] | None:
    """Return the 4 fiducial-square centroids `[(x, y), ...]` sorted
    left-to-right. Excludes the left-margin timing-stripe column
    (x < 120) so individual stripes — which can pass the size filter
    — aren't mistaken for fiducials.

    Returns None if 4 fiducials can't be found."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    H = gray.shape[0]
    band_h = max(80, H // 18)
    band = gray[:band_h, :]
    _, binary = cv2.threshold(band, 100, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary)
    fid_candidates: list[tuple[float, float, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (15 < w < 45 and 12 < h < 40 and 250 < area < 1500
                and x > 100):  # exclude left-margin stripe column
            fid_candidates.append((float(cents[i][0]), float(cents[i][1]),
                                    int(area)))
    if len(fid_candidates) < 4:
        return None
    # Pick the topmost Y-cluster of 4+ blobs (allowing tilt up to 30 px).
    fid_candidates.sort(key=lambda c: c[1])
    cy_first = fid_candidates[0][1]
    same_row = [(c[0], c[1]) for c in fid_candidates
                if abs(c[1] - cy_first) < 30]
    if len(same_row) < 4:
        return None
    same_row.sort(key=lambda p: p[0])  # sort by x
    return [same_row[0], same_row[1], same_row[-2], same_row[-1]]


def _detect_top_fiducial_y(warped: np.ndarray) -> float | None:
    """Return the average Y of the four black fiducial squares at the
    top of the form."""
    fids = _detect_top_fiducial_anchors(warped)
    if fids is None:
        return None
    return float(np.mean([y for (_x, y) in fids]))


def _fit_stripe_grid_params(
    warped: np.ndarray, max_iter: int = 5, outlier_z: float = 2.5,
) -> tuple[float, float, float] | None:
    """Iteratively fit y(i, x) = c_0 + i*sp + slope*x and return
    (c_0, sp, slope) of the converged model, or None on failure."""
    stripe_centroids = _stripe_candidates_centroid(warped, threshold=100)
    fid_anchors = _detect_top_fiducial_anchors(warped) or []
    if len(stripe_centroids) < 5:
        return None
    stripe_x = 60.0
    constraints: list[list[float]] = []
    for i, y in enumerate(stripe_centroids):
        constraints.append([float(i), stripe_x, float(y), 1.0])
    for (xf, yf) in fid_anchors:
        constraints.append([0.0, float(xf), float(yf), 1.0])
    arr = np.array(constraints, dtype=np.float64)
    for _ in range(max_iter):
        active = arr[arr[:, 3] > 0]
        if len(active) < 4:
            break
        A = np.column_stack([np.ones(len(active)), active[:, 0], active[:, 1]])
        coeffs, _, _, _ = np.linalg.lstsq(A, active[:, 2], rcond=None)
        c_0, sp, slope = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        pred = c_0 + arr[:, 0] * sp + arr[:, 1] * slope
        resid = arr[:, 2] - pred
        active_mask = arr[:, 3] > 0
        active_resid = np.abs(resid[active_mask])
        if len(active_resid) == 0:
            break
        std = float(active_resid.std())
        if std < 1e-3:
            break
        worst_idx_in_active = int(np.argmax(active_resid))
        if active_resid[worst_idx_in_active] < outlier_z * std:
            break
        active_global_idxs = np.where(active_mask)[0]
        global_idx = int(active_global_idxs[worst_idx_in_active])
        arr[global_idx, 3] = 0.0
    active = arr[arr[:, 3] > 0]
    if len(active) < 4:
        return None
    A = np.column_stack([np.ones(len(active)), active[:, 0], active[:, 1]])
    coeffs, _, _, _ = np.linalg.lstsq(A, active[:, 2], rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def _fit_stripe_grid_iterative(
    warped: np.ndarray, expected_stripes: int = 47,
    max_iter: int = 5, outlier_z: float = 2.5,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Fit a uniform-spacing tilted grid to detected stripes + 4
    top-fiducial anchors via iteratively reweighted least squares.

    Model:
        y(i, x) = c_0 + i * sp + slope * x
    where i ∈ {0..N-1} is the stripe index and x is canonical X.
    Three free parameters (c_0, sp, slope) capture the form's vertical
    starting offset, stripe spacing, and any horizontal tilt.

    Constraints:
      - 47 stripes at the timing-mark column (x ≈ 60), one per index i.
      - 4 fiducial centers at i = 0 (the topmost row), at their
        respective canonical x's. These pin the slope/tilt across the
        page — without them the slope would be ill-determined since
        all stripe constraints share the same X.

    Iteration: at each step, fit by least squares, compute residuals,
    drop the worst outlier (>= outlier_z standard deviations), refit.
    Stops when no outliers remain or after `max_iter` iterations.

    Returns a list of (x_left, y_left), (x_right, y_right) endpoint
    pairs in CANONICAL coordinates — one tilted line per stripe. The
    caller inverse-warps these to original-image coords for drawing.
    """
    stripe_centroids = _stripe_candidates_centroid(warped, threshold=100)
    fid_anchors = _detect_top_fiducial_anchors(warped) or []

    if len(stripe_centroids) < 5:
        return []

    # Build initial constraint set: each entry = (i, x, y_observed, weight).
    stripe_x = 60.0
    constraints: list[list[float]] = []
    # Stripes are indexed by their position in the detected list;
    # assume detected[0] is index 0 (topmost stripe).
    for i, y in enumerate(stripe_centroids):
        constraints.append([float(i), stripe_x, float(y), 1.0])
    # Fiducials all sit on stripe index 0 (the topmost form row).
    for (xf, yf) in fid_anchors:
        constraints.append([0.0, float(xf), float(yf), 1.0])

    arr = np.array(constraints, dtype=np.float64)

    # Iteratively least-squares fit, dropping high-residual points.
    for _ in range(max_iter):
        active = arr[arr[:, 3] > 0]
        if len(active) < 4:
            break
        # Least squares solve: y = c_0 + sp * i + slope * x
        # Design matrix columns: [1, i, x]
        A = np.column_stack([np.ones(len(active)), active[:, 0], active[:, 1]])
        b = active[:, 2]
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c_0, sp, slope = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        # Residuals on ALL points (active + already-dropped).
        pred = c_0 + arr[:, 0] * sp + arr[:, 1] * slope
        resid = arr[:, 2] - pred
        # Drop the single worst outlier among ACTIVE points.
        active_mask = arr[:, 3] > 0
        active_resid = np.abs(resid[active_mask])
        if len(active_resid) == 0:
            break
        std = float(active_resid.std())
        if std < 1e-3:
            break
        worst_idx_in_active = int(np.argmax(active_resid))
        worst_resid = float(active_resid[worst_idx_in_active])
        if worst_resid < outlier_z * std:
            break  # converged: no significant outliers
        # Map back to global index in `arr`
        active_global_idxs = np.where(active_mask)[0]
        global_idx = int(active_global_idxs[worst_idx_in_active])
        arr[global_idx, 3] = 0.0  # drop

    # Final fit
    active = arr[arr[:, 3] > 0]
    if len(active) < 4:
        return []
    A = np.column_stack([np.ones(len(active)), active[:, 0], active[:, 1]])
    coeffs, _, _, _ = np.linalg.lstsq(A, active[:, 2], rcond=None)
    c_0, sp, slope = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # Always emit `expected_stripes` (47) lines from the converged
    # model — even if some stripes were cut off in the scan, the
    # printed form has all 47, so we extrapolate the model. some scans
    # 12102024 specifically had its last stripe missing from the scan;
    # using len(stripe_centroids) would drop the last line entirely.
    n_lines = expected_stripes
    W = warped.shape[1]
    x_left = 0.0
    x_right = float(W - 1)
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n_lines):
        y_left = c_0 + i * sp + slope * x_left
        y_right = c_0 + i * sp + slope * x_right
        pairs.append(((x_left, y_left), (x_right, y_right)))
    return pairs


def _detect_stripes_in_warped(warped: np.ndarray) -> list[int]:
    """Run stripe detection on the warped (canonical) image's left margin.
    Uses the centroid-based detector so each returned Y is at the
    geometric MIDDLE of a printed stripe.

    The TOPMOST stripe Y is replaced by the fiducial-row Y when fiducials
    are detectable — this anchors the first horizontal line so it
    bisects both the leftmost timing stripe AND the four black fiducial
    squares at the top of the form (they sit on the same form row).
    """
    stripes = _stripe_candidates_centroid(warped, threshold=100)
    if not stripes:
        return stripes
    fid_y = _detect_top_fiducial_y(warped)
    if fid_y is None:
        return stripes
    # If the fiducial Y is close to the first detected stripe (within a
    # half-spacing), replace the first stripe Y with the fiducial Y so
    # the line bisects both.
    sp = (stripes[-1] - stripes[0]) / max(1, len(stripes) - 1)
    if abs(fid_y - stripes[0]) < sp / 2:
        return [int(round(fid_y))] + stripes[1:]
    # Otherwise insert the fiducial Y as a new topmost line.
    return [int(round(fid_y))] + stripes


_FIDUCIAL_MARKER_PATH = Path(__file__).resolve().parent / "fiducial_marker.png"
_FIDUCIAL_MARKER_CACHE: np.ndarray | None = None


def _get_fiducial_marker() -> np.ndarray | None:
    """Load (and cache) the calibrated fiducial-marker template."""
    global _FIDUCIAL_MARKER_CACHE
    if _FIDUCIAL_MARKER_CACHE is None:
        if not _FIDUCIAL_MARKER_PATH.exists():
            return None
        m = cv2.imread(str(_FIDUCIAL_MARKER_PATH), cv2.IMREAD_GRAYSCALE)
        _FIDUCIAL_MARKER_CACHE = m
    return _FIDUCIAL_MARKER_CACHE


def _detect_top_fiducials(warped: np.ndarray) -> list[int]:
    """Detect the X centers of the four black fiducial squares printed
    at the top of UF Form LR1.

    Tries connected-components detection first (very reliable on these
    forms because fiducials are large solid black blobs). Falls back
    to template-matching only if that fails. Previously template-
    matching was the primary; some scans (some scans) had marker
    correlations of ~0.4, just below the 0.45 acceptance threshold,
    causing fiducial detection to return empty even though the
    fiducials were clearly visible. CC detection sees them fine.
    """
    cc = _detect_top_fiducials_cc(warped)
    if len(cc) == 4:
        return cc
    marker = _get_fiducial_marker()
    if marker is None:
        return cc

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    H, W = gray.shape
    band_h = max(80, H // 20)
    band = gray[:band_h, :]
    mh, mw = marker.shape[:2]
    half_w = W // 2

    def find_top2(region: np.ndarray, x_offset: int) -> list[tuple[float, float]]:
        if region.shape[0] < mh or region.shape[1] < mw:
            return []
        res = cv2.matchTemplate(region, marker, cv2.TM_CCOEFF_NORMED)
        # Greedy NMS: take best, suppress neighborhood, take next.
        out = []
        suppressed = res.copy()
        for _ in range(2):
            _, max_val, _, max_loc = cv2.minMaxLoc(suppressed)
            if max_val < 0.45:
                break
            x = max_loc[0] + mw / 2.0 + x_offset
            y = max_loc[1] + mh / 2.0
            out.append((x, y))
            # Suppress only ±half-marker AROUND THE PEAK so an adjacent
            # fiducial whose top-left is mw away (= ~37 px on the reference scan) is
            # not silenced.
            half_mw, half_mh = mw // 2, mh // 2
            x0 = max(0, max_loc[0] - half_mw)
            x1 = min(suppressed.shape[1], max_loc[0] + half_mw)
            y0 = max(0, max_loc[1] - half_mh)
            y1 = min(suppressed.shape[0], max_loc[1] + half_mh)
            suppressed[y0:y1, x0:x1] = -1.0
        return out

    left_pair = find_top2(band[:, :half_w], 0)
    right_pair = find_top2(band[:, half_w:], half_w)
    if len(left_pair) < 2 or len(right_pair) < 2:
        return _detect_top_fiducials_cc(warped)
    centers = sorted(left_pair + right_pair, key=lambda p: p[0])
    return [int(round(p[0])) for p in centers]


def _detect_top_fiducials_cc(warped: np.ndarray) -> list[int]:
    """Connected-components fallback for fiducial detection (used when
    the template image is missing or template matching fails).

    band_h is set generously (160 px) because the Y position of the
    fiducial row varies by scan: the cleanest reference scan has fiducials at y~30,
    some scans at y~80 (the warp's vertical offset depends on form-quad
    detection). A short band misses fiducials that landed lower.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    H, W = gray.shape
    band_h = max(160, H // 12)
    band = gray[:band_h, :]
    _, binary = cv2.threshold(band, 100, 255, cv2.THRESH_BINARY_INV)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (15 < w < 45 and 12 < h < 40 and 250 < area < 1500
                and x > 50):  # exclude left-margin timing-stripe blobs
            candidates.append((float(centroids[i][0]),
                                float(centroids[i][1]), int(area)))
    if len(candidates) < 4:
        return []
    # Pick the topmost row of 4 — fiducials form a horizontal line near
    # the top, while bubbles below them are at much higher y. Sort by y,
    # take the top y-cluster.
    candidates.sort(key=lambda c: c[1])
    cy_top = candidates[0][1]
    same_row = [c for c in candidates if abs(c[1] - cy_top) < 30]
    if len(same_row) < 4:
        return []
    left = [c for c in same_row if c[0] < W / 2]
    right = [c for c in same_row if c[0] >= W / 2]
    if len(left) < 2 or len(right) < 2:
        return []
    left.sort(key=lambda c: -c[2])
    right.sort(key=lambda c: -c[2])
    chosen = sorted(left[:2] + right[:2], key=lambda c: c[0])
    return [int(round(c[0])) for c in chosen]

# Sidebar geometry
SIDEBAR_WIDTH = 480
SIDEBAR_BG = (250, 250, 250)


def _collect_unique_positions(values: list[float], merge_within: float = 6.0
                              ) -> list[float]:
    """Merge near-duplicate positions (within `merge_within` px) into one.
    Used to build a single page-wide grid from per-region xs/ys."""
    if not values:
        return []
    arr = sorted(values)
    merged = [arr[0]]
    for v in arr[1:]:
        if v - merged[-1] < merge_within:
            merged[-1] = (merged[-1] + v) / 2.0
        else:
            merged.append(v)
    return merged


def annotate_warp(warped: np.ndarray, template: dict,
                   grid_lines: bool = True) -> np.ndarray:
    """Draw template grid + filled-bubble overlay on a copy of the warp.

    Horizontal grid lines are drawn at the actual detected timing-stripe
    positions — the form's printed left-margin stripes are the ground
    truth for row alignment, so any grid line drawn here visually
    coincides with a real stripe on the page. Vertical column lines are
    drawn at the union of every region's column X positions.

    A misaligned read shows up as a green ring sitting between the
    horizontal stripe lines instead of on one of them.
    """
    vis = warped.copy()
    green = green_channel(warped)
    H, W = warped.shape[:2]

    # Horizontal grid: detected stripe Ys in the warped image (one per
    # printed timing mark). These ARE the form's row anchors.
    stripe_ys = _detect_stripes_in_warped(warped)

    # Vertical grid: anchor on the four black fiducial squares printed at
    # the top of the form (two pairs: top-left and top-right). Then
    # extrapolate the 12 LAST_NAME column verticals from the LEFT pair —
    # the two left fiducials sit on LAST_NAME columns 2 and 3, so col 1
    # is at fid[0] - spacing and cols 4..12 are at fid[1] + k*spacing for
    # k = 1..9. FI and MI sit one and two more spacings to the right.
    fiducial_xs = _detect_top_fiducials(warped)
    name_block_xs: list[float] = []
    if len(fiducial_xs) >= 4:
        # 4-point linear fit using ALL fiducials. UF Form LR1 column
        # layout (in form-unit indices, 1-indexed, from the leftmost
        # LAST_NAME column):
        #   col 1, 2, 3, ..., 12 = LAST_NAME (12 cols)
        #   col 13 = FI, col 14 = MI
        #   fid[0] sits on col 2, fid[1] on col 3 (left pair)
        #   fid[2] on col 21, fid[3] on col 23 (right pair, 2 cols apart)
        ks = np.array([2.0, 3.0, 21.0, 23.0])
        xs = np.array([float(x) for x in fiducial_xs[:4]])
        A = np.vstack([ks, np.ones(4)]).T
        col_slope, col_intercept = np.linalg.lstsq(A, xs, rcond=None)[0]
        for k in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            name_block_xs.append(float(col_intercept) + float(col_slope) * k)

    # Vertical lines should be drawn PERPENDICULAR to the horizontal
    # stripe grid (which has slope `s` from the iterative fit). In
    # canonical, horizontal direction is (1, s); perpendicular is
    # (-s, 1). For a vertical line passing through canonical X = x0
    # at the topmost row Y (= c_0 + s*x0), the endpoints are:
    #   top    (y=0):     (x0 + s*y0,             0)
    #   bottom (y=H-1):   (x0 - s*(H-1-y0),     H-1)
    # where y0 = c_0 + s*x0.
    grid_params = _fit_stripe_grid_params(warped)
    H, W = warped.shape[:2]
    if grid_params is not None:
        c_0, sp, slope = grid_params
    else:
        c_0, sp, slope = 0.0, 38.0, 0.0

    def _perpendicular_endpoints(x0: float) -> tuple[tuple[float, float],
                                                      tuple[float, float]]:
        y0 = c_0 + slope * x0
        top = (x0 + slope * y0, 0.0)
        bot = (x0 - slope * (H - 1 - y0), float(H - 1))
        return top, bot

    # Draw the page-wide grid first so per-region annotations render on top.
    if grid_lines:
        for y in stripe_ys:
            cv2.line(vis, (0, int(y)), (W - 1, int(y)),
                     STRIPE_LINE_COLOR, STRIPE_LINE_THICK, cv2.LINE_AA)
        for x in fiducial_xs:
            top, bot = _perpendicular_endpoints(float(x))
            cv2.line(vis, (int(top[0]), int(top[1])),
                     (int(bot[0]), int(bot[1])),
                     COL_LINE_COLOR, COL_LINE_THICK + 1, cv2.LINE_AA)
        for x in name_block_xs:
            top, bot = _perpendicular_endpoints(float(x))
            cv2.line(vis, (int(top[0]), int(top[1])),
                     (int(bot[0]), int(bot[1])),
                     COL_LINE_COLOR, COL_LINE_THICK, cv2.LINE_AA)

    # Step 3: per-region overlays (bbox, dots, fills, labels).
    for name, region in template["regions"].items():
        xs = region.get("xs")
        ys = region.get("ys")
        if not xs or not ys:
            continue

        x0, y0, x1, y1 = region.get("bbox", (0, 0, 0, 0))
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                      BBOX_COLOR, 1, cv2.LINE_AA)

        for y in ys:
            for x in xs:
                cv2.circle(vis, (int(x), int(y)), GRID_DOT_RADIUS,
                           GRID_DOT_COLOR, -1)

        for y in ys:
            for x in xs:
                if score_bubble(green, x, y) >= FILL_THRESH:
                    cv2.circle(vis, (int(x), int(y)), FILLED_RING_RADIUS,
                               FILLED_RING_COLOR, FILLED_RING_THICK)

        if name in ("last_name", "uf_id", "section", "first_initial",
                    "middle_initial", "special_codes"):
            x_label = max(2, int(xs[0]) - 28)
            for ri, y in enumerate(ys):
                lab = region["row_labels"][ri].strip()
                if not lab:
                    continue
                cv2.putText(vis, lab, (x_label, int(y) + 4),
                            ANNOT_FONT, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)

        if name in ("last_name", "uf_id", "section") and len(xs) > 1:
            y_label = max(12, int(min(ys)) - 18)
            for ci, x in enumerate(xs):
                lab = region["col_labels"][ci]
                cv2.putText(vis, lab, (int(x) - 4, y_label),
                            ANNOT_FONT, 0.35, LABEL_COLOR, 1, cv2.LINE_AA)
    return vis


def render_sidebar(decoded: dict, h: int) -> np.ndarray:
    """Right-side info panel summarizing decoded fields."""
    panel = np.full((h, SIDEBAR_WIDTH, 3), SIDEBAR_BG, dtype=np.uint8)
    y = 30

    def line(text, big=False, color=(20, 20, 20)):
        nonlocal y
        scale = 0.65 if big else 0.5
        thick = 2 if big else 1
        cv2.putText(panel, text, (12, y), ANNOT_FONT, scale, color, thick,
                    cv2.LINE_AA)
        y += 28 if big else 22

    line(decoded.get("file", "?"), big=True)
    y += 6
    line(f"page: {decoded.get('omr_page', '?')}    "
         f"red_score: {decoded.get('red_score', 0):.3f}")
    y += 4

    line(f"last_name: {decoded.get('last_name')}", big=True)
    line(f"FI: {decoded.get('first_initial')}    "
         f"MI: {decoded.get('middle_initial')}", big=True)
    line(f"UF_ID: {decoded.get('uf_id')}", big=True)
    line(f"section: {decoded.get('section')}    "
         f"form: {decoded.get('test_form_code')}")
    line(f"special_codes: {decoded.get('special_codes')}")
    y += 10

    line("answers:", big=True)
    answers = decoded.get("answers") or {}
    # 4 columns × 20 rows for 1..80
    col_w = (SIDEBAR_WIDTH - 24) // 4
    for q in range(1, 81):
        col = (q - 1) // 20
        row = (q - 1) % 20
        ax = 12 + col * col_w
        ay = y + row * 18
        a = answers.get(str(q))
        if a is None:
            txt = f"{q:>2}: -"
            color = (160, 160, 160)
        else:
            txt = f"{q:>2}: {a}"
            color = (20, 20, 20)
        cv2.putText(panel, txt, (ax, ay), ANNOT_FONT, 0.4, color, 1,
                    cv2.LINE_AA)
    return panel


def annotate_one(pdf_path: Path, template: dict, dpi: int) -> np.ndarray | None:
    """Run OMR and return an annotated image:
       full original page (left)  +  sidebar with decoded fields (right).
    Annotations (grid lines, fiducials, filled-bubble rings, region
    bboxes) are inverse-warped from canonical coordinates onto the
    original page so users see the entire scan in context.
    """
    candidates = find_omr_pages(str(pdf_path))
    if not candidates:
        return None
    page_idx, score = candidates[0]
    img = render_page(str(pdf_path), page_idx, dpi=dpi)
    warped, M, rotation = warp_omr(img, return_matrix=True)
    if warped is None:
        return None

    # Grid reader: matches the visualization (rings appear exactly
    # where the reader claims fills) and sidebar reflects the data
    # being extracted by omr.py's default pipeline.
    decoded = read_omr_grid(warped)
    per_scan_tpl = template  # legacy template still passed for region bbox
                             # outlines used by annotate_warp (canonical-only)
    decoded["file"] = pdf_path.name
    decoded["omr_page"] = page_idx
    decoded["red_score"] = round(float(score), 4)

    overlay = annotate_full_page(img, warped, per_scan_tpl, M, rotation)
    sidebar = render_sidebar(decoded, overlay.shape[0])
    return np.hstack([overlay, sidebar])


def annotate_full_page(orig_img: np.ndarray, warped: np.ndarray,
                        template: dict, M: np.ndarray,
                        rotation: int | None) -> np.ndarray:
    """Draw the OMR overlay on the FULL original PDF page, with all
    annotations inverse-warped from canonical coords back to the
    page's coordinate system. Returns the annotated original image
    (resized to a reasonable display height).
    """
    page = orig_img.copy()
    Hp, Wp = page.shape[:2]
    green_canon = green_channel(warped)

    def to_orig(canon_pts: np.ndarray) -> np.ndarray:
        return canonical_to_original(canon_pts, M, rotation,
                                      warped_shape=warped.shape[:2])

    # Iteratively fit a 3-parameter grid model to stripe centroids +
    # fiducial centers, then draw each stripe line from the converged
    # model. This averages out per-stripe detection noise and keeps
    # consecutive lines uniformly spaced even on tilted scans.
    grid_lines_pairs = _fit_stripe_grid_iterative(warped)
    for (xy_left, xy_right) in grid_lines_pairs:
        endpoints = np.array([xy_left, xy_right], dtype=np.float32)
        eo = to_orig(endpoints)
        cv2.line(page, (int(eo[0][0]), int(eo[0][1])),
                 (int(eo[1][0]), int(eo[1][1])),
                 STRIPE_LINE_COLOR, STRIPE_LINE_THICK + 1, cv2.LINE_AA)

    # Vertical grid: anchor strictly on the 4 black fiducial squares
    # at the top of the form. UF Form LR1 column layout (form-unit
    # indices, 1-indexed from leftmost LAST_NAME col):
    #   k = 1..12   LAST_NAME
    #   k = 13      FI
    #   k = 14      MI
    #   k = 15..16  gap (no bubbles, between LAST_NAME and answers)
    #   k = 17..21  left answer block A..E (Q1-40)
    #   k = 22..23  gap (between answer blocks)
    #   k = 24..28  right answer block / TEST_FORM_CODE A..E (Q41-80)
    #
    # Fiducials sit at:
    #   fid[0]=k=2, fid[1]=k=3   (left pair, above LAST_NAME)
    #   fid[2]=k=21, fid[3]=k=23 (right pair)
    # Least-squares fit through these 4 points gives col_slope (pitch
    # in canonical px per form-unit) and col_intercept. ALL bubble
    # columns are at integer k on a UNIFORM pitch — so extrapolate
    # k=1..28 to get every vertical. No red-circle clustering — that
    # was dropping columns wherever HoughCircles missed bubbles.
    fiducial_xs = _detect_top_fiducials(warped)
    if len(fiducial_xs) >= 4:
        ks = np.array([2.0, 3.0, 21.0, 23.0])
        xs = np.array([float(x) for x in fiducial_xs[:4]])
        A = np.vstack([ks, np.ones(4)]).T
        col_slope, col_intercept = np.linalg.lstsq(A, xs, rcond=None)[0]
        vertical_xs = [float(col_intercept) + float(col_slope) * k
                        for k in range(1, 29)]
    else:
        vertical_xs = list(fiducial_xs)
    grid_params = _fit_stripe_grid_params(warped)
    if grid_params is not None:
        c_0, sp, h_slope = grid_params
    else:
        c_0, sp, h_slope = 0.0, 0.0, 0.0
    Hwarp = warped.shape[0]
    for x0 in vertical_xs:
        x0 = float(x0)
        y0 = c_0 + h_slope * x0
        # Direction perpendicular to (1, h_slope) is (-h_slope, 1).
        x_top = x0 + h_slope * y0
        y_top = 0.0
        x_bot = x0 - h_slope * (Hwarp - 1 - y0)
        y_bot = float(Hwarp - 1)
        endpoints = np.array([[x_top, y_top], [x_bot, y_bot]],
                             dtype=np.float32)
        eo = to_orig(endpoints)
        cv2.line(page, (int(eo[0][0]), int(eo[0][1])),
                 (int(eo[1][0]), int(eo[1][1])),
                 COL_LINE_COLOR, COL_LINE_THICK + 1, cv2.LINE_AA)

    # Green ring ONLY at intersections where the bubble is actually
    # filled (score >= FILL_THRESH) AND the intersection corresponds to
    # an actual printed bubble per GRID_LAYOUT. This rules out text
    # strokes that score "filled" but live at non-bubble intersections
    # (e.g., WRITE row letters at i=4, "TEST FORM CODE" header at i=2).
    s2_p1 = 1.0 + h_slope * h_slope
    fid_intercept = float(col_intercept) if len(fiducial_xs) >= 4 else 0.0
    fid_slope = float(col_slope) if len(fiducial_xs) >= 4 else 1.0

    def _bubble_xy(k: int, i: int) -> tuple[float, float]:
        x0 = fid_intercept + fid_slope * k
        i_sp_norm = i * sp / s2_p1
        y_int = c_0 + i_sp_norm + h_slope * x0
        x_int = x0 - h_slope * i_sp_norm
        return x_int, y_int

    filled_canon: list[tuple[float, float]] = []
    for region_name, (col_ks, row_is, _row_labels, _kind) in GRID_LAYOUT.items():
        for i in row_is:
            for k in col_ks:
                x_int, y_int = _bubble_xy(k, i)
                if score_bubble(green_canon, x_int, y_int) >= FILL_THRESH:
                    filled_canon.append((x_int, y_int))

    # Compute canonical→original scale for the marker radius so green
    # rings match printed bubble size visually.
    Hc, Wc = warped.shape[:2]
    probe = np.array([[Wc / 2, Hc / 2],
                       [Wc / 2 + 100.0, Hc / 2]], dtype=np.float32)
    probe_orig = to_orig(probe)
    canon_to_orig_scale = (
        float(np.linalg.norm(probe_orig[1] - probe_orig[0])) / 100.0)
    canon_to_orig_scale = max(canon_to_orig_scale, 1.0)
    BUBBLE_R_CANON = 12.0
    ring_r_orig = max(6, int(round(BUBBLE_R_CANON * canon_to_orig_scale)))

    if filled_canon:
        pts_orig = to_orig(np.array(filled_canon, dtype=np.float32))
        for (xo, yo) in pts_orig:
            cv2.circle(page, (int(xo), int(yo)), ring_r_orig,
                       FILLED_RING_COLOR, 3, cv2.LINE_AA)

    # Resize to a max display height so each PDF page isn't huge.
    max_h = 1800
    if Hp > max_h:
        scale = max_h / Hp
        page = cv2.resize(page, (int(Wp * scale), max_h),
                          interpolation=cv2.INTER_AREA)
    return page


def find_pdfs(root: Path, ext: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(f"*{ext}"))


def image_to_pdf_page(out: fitz.Document, img_bgr: np.ndarray,
                       jpeg_quality: int = 70) -> None:
    """Append the image (BGR uint8) as a JPEG-encoded page in `out`."""
    ok, buf = cv2.imencode(".jpg", img_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    h, w = img_bgr.shape[:2]
    page = out.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=buf.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input_dir", type=Path,
                    help="directory or single PDF to annotate")
    ap.add_argument("output_path", type=Path,
                    help="combined PDF output path (default mode), "
                         "or directory if --per-pdf")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--combined", action="store_true",
                   help="combine into a single multi-page PDF (default)")
    g.add_argument("--per-pdf", action="store_true",
                   help="write one annotated PDF per input")
    ap.add_argument("--ext", default=".pdf",
                    help="file extension (default .pdf)")
    ap.add_argument("--dpi", type=int, default=300,
                    help="render DPI (default 300)")
    ap.add_argument("--template", type=Path,
                    default=Path(__file__).parent / "template.json",
                    help="path to template.json")
    args = ap.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"not found: {args.input_dir}")
    pdfs = find_pdfs(args.input_dir, args.ext)
    if not pdfs:
        sys.exit(f"no {args.ext} files in {args.input_dir}")

    template = load_template(args.template)
    per_pdf = args.per_pdf  # default = combined

    if per_pdf:
        args.output_path.mkdir(parents=True, exist_ok=True)
    else:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_doc = fitz.open()

    for pdf in pdfs:
        try:
            annotated = annotate_one(pdf, template, args.dpi)
        except Exception as e:
            print(f"  [SKIP] {pdf.name}: {type(e).__name__}: {e}")
            continue
        if annotated is None:
            print(f"  [SKIP] {pdf.name}: no OMR page")
            continue
        if per_pdf:
            out_path = args.output_path / f"{pdf.stem}.annot.pdf"
            single = fitz.open()
            image_to_pdf_page(single, annotated)
            single.save(str(out_path))
            single.close()
            print(f"  {pdf.name} → {out_path.name}")
        else:
            image_to_pdf_page(out_doc, annotated)
            print(f"  {pdf.name}")

    if not per_pdf:
        out_doc.save(str(args.output_path))
        out_doc.close()
        print(f"\nwrote {args.output_path} ({len(pdfs)} pages)")


if __name__ == "__main__":
    main()
