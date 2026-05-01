"""Shared OMR utilities: PDF rendering, form detection, deskew."""
from __future__ import annotations
import cv2
import numpy as np
import fitz

CANON_W = 1200
CANON_H = 1800


def render_page(pdf_path: str, idx: int, dpi: int = 300) -> np.ndarray:
    doc = fitz.open(pdf_path)
    pix = doc[idx].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def red_mask(img: np.ndarray, thresh: int = 25) -> np.ndarray:
    b, g, r = cv2.split(img)
    red = cv2.subtract(r, cv2.min(g, b))
    _, m = cv2.threshold(red, thresh, 255, cv2.THRESH_BINARY)
    return m


def red_score(img: np.ndarray) -> float:
    return float(red_mask(img, thresh=30).mean()) / 255.0


def find_omr_pages(pdf_path: str, threshold: float = 0.02, dpi: int = 100):
    """Return [(page_idx, score), ...] for pages whose red density exceeds threshold,
    sorted by score descending."""
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    out = []
    for i in range(n):
        img = render_page(pdf_path, i, dpi=dpi)
        s = red_score(img)
        if s > threshold:
            out.append((i, s))
    out.sort(key=lambda x: -x[1])
    return out


def _detect_circles_full(img: np.ndarray, dpi: int = 300) -> np.ndarray:
    """Detect bubble centers anywhere on the page. Used to anchor the form
    boundary independently of paper bleed / ruled-line scan noise."""
    mask = red_mask(img, thresh=25)
    gray = cv2.bitwise_not(mask)
    gray = cv2.medianBlur(gray, 3)
    r_nom = max(8, int(14 * dpi / 300))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=int(r_nom * 1.5),
        param1=100, param2=18,
        minRadius=int(r_nom * 0.6), maxRadius=int(r_nom * 1.6))
    if circles is None:
        return np.zeros((0, 2), dtype=np.float32)
    return circles[0, :, :2].astype(np.float32)


def find_form_quad(img: np.ndarray) -> np.ndarray | None:
    """Find the OMR card boundary (4 corners).

    Two-tier strategy:
      1. Try the red-mask contour approach (cheap, accurate when scan is
         clean). Reject if the resulting quad isn't plausibly portrait
         (UF Form LR1 has h/w ≈ 1.29).
      2. Fall back to a bubble-cluster minAreaRect — bubbles are
         deterministic and only exist inside the form, so the quad is
         scan-noise-resistant. Used when the red-mask approach fails or
         returns a non-portrait blob (e.g., scan with paper bleed on the
         right inflating the form's bounding box).
    """
    red_quad = _find_form_quad_red(img)
    if red_quad is not None and _is_portrait(red_quad):
        return _expand_quad_for_stripes(red_quad)
    pts = _detect_circles_full(img)
    if len(pts) < 200:
        return red_quad
    hull = cv2.convexHull(pts.astype(np.float32))
    rect = cv2.minAreaRect(hull)
    (cx, cy), (w, h), angle = rect
    pad_w, pad_h = 50.0, 90.0
    rect = ((cx, cy), (w + 2 * pad_w, h + 2 * pad_h), angle)
    quad = cv2.boxPoints(rect).astype(np.float32)
    # Same asymmetric expansion as the red-mask path so the left-margin
    # timing stripes (printed OUTSIDE the bubble convex hull) end up
    # inside the warp canvas.
    return _expand_quad_for_stripes(quad)


def _expand_quad_for_stripes(quad: np.ndarray, left_pad_px: float = 110,
                              extra_pad_px: float = 25) -> np.ndarray:
    """Inflate a portrait quad on all sides so that timing stripes
    (printed on the form's left margin OUTSIDE the red-ink boundary)
    end up inside the warp canvas. Asymmetric: extra padding on the
    left because that's where stripes live."""
    quad = order_quad(quad)
    tl, tr, br, bl = quad
    # Build unit vectors along the form's local axes
    top_vec = tr - tl
    side_vec = bl - tl
    top_len = float(np.linalg.norm(top_vec))
    side_len = float(np.linalg.norm(side_vec))
    if top_len < 1 or side_len < 1:
        return quad
    u_horiz = top_vec / top_len
    u_vert = side_vec / side_len
    expand = lambda p, dh, dv: p + dh * u_horiz + dv * u_vert
    new_tl = expand(tl, -left_pad_px, -extra_pad_px)
    new_tr = expand(tr, +extra_pad_px, -extra_pad_px)
    new_br = expand(br, +extra_pad_px, +extra_pad_px)
    new_bl = expand(bl, -left_pad_px, +extra_pad_px)
    return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)


def _is_portrait(quad: np.ndarray, min_ratio: float = 1.22) -> bool:
    """Return True if the rotated rectangle described by `quad` has its
    longer side at least `min_ratio` times its shorter side — UF LR1 is
    always portrait so a near-square quad signals a bad detection
    (typically scan noise inflating the bounding box)."""
    if quad is None or len(quad) < 4:
        return False
    side1 = float(np.linalg.norm(quad[0] - quad[1]))
    side2 = float(np.linalg.norm(quad[1] - quad[2]))
    long_side = max(side1, side2)
    short_side = min(side1, side2)
    if short_side < 1:
        return False
    return long_side / short_side >= min_ratio


def _find_form_quad_red(img: np.ndarray) -> np.ndarray | None:
    """Fallback form-quad detector using red-mask morphology."""
    mask = red_mask(img, thresh=25)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(mask, erode_k, iterations=1)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_k, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50000:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        long_side, short_side = max(w, h), min(w, h)
        if short_side < 1:
            continue
        aspect = long_side / short_side
        if 1.05 < aspect < 2.0:
            candidates.append((area, rect))
    if not candidates:
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        return cv2.boxPoints(rect).astype(np.float32)
    candidates.sort(key=lambda t: -t[0])
    return cv2.boxPoints(candidates[0][1]).astype(np.float32)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order a 4-point quad as [tl, tr, br, bl] in image coords."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_to_canonical(img: np.ndarray, quad: np.ndarray,
                      w: int = CANON_W, h: int = CANON_H,
                      return_matrix: bool = False):
    """Perspective-warp img using quad to a canonical wxh rectangle.
    Auto-detects orientation (long side vertical). If return_matrix,
    also return the perspective matrix M (canonical → original) and the
    final ordered quad used for the transform."""
    quad = order_quad(quad)
    side1 = np.linalg.norm(quad[0] - quad[1])
    side2 = np.linalg.norm(quad[1] - quad[2])
    if side1 > side2:
        quad = np.array([quad[1], quad[2], quad[3], quad[0]], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    if return_matrix:
        return warped, M, quad
    return warped


def detect_orientation(warped: np.ndarray) -> str:
    """Return 'front' if header reads upright (UF/Form LR1) else 'flipped'.
    Heuristic: front side has TEST FORM CODE block in the top-right; back side does not.
    Use the dark count in the top band as proxy for content position.
    """
    h, w = warped.shape[:2]
    # 'UNIVERSITY OF FLORIDA' centered text is near top of front, near bottom of flipped
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    top_band = gray[: h // 20]
    bot_band = gray[-h // 20:]
    # compare dark text density
    return "front" if top_band.mean() > bot_band.mean() else "flipped"


EXPECTED_STRIPES = 47  # UF Form LR1 has 47 timing rows in the left margin


def _stripe_candidates_centroid(img: np.ndarray, threshold: int = 130,
                                 half_w: int = 8) -> list[int]:
    """Find timing-stripe Y centers by taking the midpoint of each
    contiguous dark band on the form's left-margin column.

    Unlike `_stripe_candidates` (peak-detection), this is guaranteed to
    return the geometric MIDDLE of each stripe even when the stripe is
    several pixels tall and not perfectly uniform — drawing a line at
    each returned Y will pass through the centroid of the corresponding
    printed stripe.

    A row counts as "dark" if a majority of its pixels (within the
    selected strip column) are below `threshold`. This is more robust
    than averaging when the strip is wider than the stripe itself.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    search_w = min(W // 4, 800)
    # Strict bin threshold (100, NOT 150) so the densest-transition
    # column points at the actual TIMING STRIPE column, not at the
    # fiducial column (which would be densest at the more lenient 150
    # threshold because fiducials are taller and create a long-running
    # dark region with more text-mark transitions nearby).
    bin_region = gray[:, :search_w] < 100
    transitions = np.abs(np.diff(bin_region.astype(np.int8), axis=0)).sum(axis=0)
    if transitions.max() < 20:
        return []
    peak_x = int(np.argmax(transitions))
    x0 = max(0, peak_x - half_w)
    x1 = min(W - 1, peak_x + half_w)
    strip = gray[:, x0:x1 + 1]
    # Per-row count of dark pixels; row is "dark" if >= 1/3 are below
    # threshold (catches partial-coverage rows where the stripe doesn't
    # span the full strip width).
    dark_count = (strip < threshold).sum(axis=1)
    cols = strip.shape[1]
    row_dark = dark_count >= max(3, cols // 3)
    # Group contiguous dark rows into bands; output centroid Y per band.
    centers: list[int] = []
    in_band = False
    band_start = 0
    min_band_h = 5
    for y, d in enumerate(row_dark):
        if d and not in_band:
            in_band = True
            band_start = y
        elif not d and in_band:
            in_band = False
            band_h = y - band_start
            if band_h >= min_band_h:
                centers.append((band_start + y - 1) // 2)
    if in_band:
        band_h = len(row_dark) - band_start
        if band_h >= min_band_h:
            centers.append((band_start + len(row_dark) - 1) // 2)
    return centers


def _stripe_candidates(img: np.ndarray, threshold: int = 100,
                         half_w: int = 10, distance: int = 25,
                         prominence: int = 20) -> list[int]:
    """Detect candidate stripe Y centers in the left margin.

    1. Locate the stripe column by finding the page column with the most
       dark→light transitions. Threshold of 100 picks up only the strong
       black-stripe edges, not lighter form text.
    2. Take a narrow strip ±half_w around the peak column.
    3. Run scipy.signal.find_peaks on the inverted row-mean signal with a
       minimum spacing constraint to suppress sub-stripe noise.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    search_w = min(W // 4, 800)
    bin_region = gray[:, :search_w] < threshold
    transitions = np.abs(np.diff(bin_region.astype(np.int8), axis=0)).sum(axis=0)
    if transitions.max() < 20:
        return []
    peak_x = int(np.argmax(transitions))
    x_start = max(0, peak_x - half_w)
    x_end = min(W - 1, peak_x + half_w)
    strip = gray[:, x_start:x_end + 1]
    row_mean = strip.mean(axis=1)
    inverted = 255.0 - row_mean
    peaks, _ = find_peaks(inverted, distance=distance, prominence=prominence)
    return peaks.tolist()


def detect_timing_marks(img: np.ndarray, form_top_y: float | None = None) -> list[int]:
    """Detect timing-mark Y centers down the left margin and return EXACTLY
    EXPECTED_STRIPES positions by fitting a robust linear model to detected
    candidates. Returns an empty list if no plausible grid can be fit.

    The fit handles missing stripes (some scans don't show every stripe
    clearly) by extrapolating positions from the candidates that DO fit a
    regular spacing pattern. If `form_top_y` is provided, stripe 0 is
    anchored to it so a scan that fails to detect the topmost stripes still
    produces correctly-indexed positions.
    """
    # Pick the threshold that yields the most candidates — better signal for
    # the linear fit downstream (the fit can handle a few extras).
    cand_100 = _stripe_candidates(img, threshold=100)
    cand_150 = _stripe_candidates(img, threshold=150)
    candidates = cand_150 if len(cand_150) > len(cand_100) else cand_100
    return _fit_stripe_grid(candidates, EXPECTED_STRIPES, form_top_y=form_top_y)


def _fit_stripe_grid(candidates: list[int],
                     expected: int = EXPECTED_STRIPES,
                     tol: float = 0.25,
                     form_top_y: float | None = None) -> list[int]:
    """Fit a linear y = a + b*i model to candidates so we recover exactly
    `expected` positions (filling any missed stripes by extrapolation).

    Index 0 is anchored to the topmost candidate that fits the grid, OR if
    `form_top_y` is given, to the form's top edge — that way scans missing
    the first few stripes still produce correctly indexed grids.
    """
    if len(candidates) < 15:
        return []
    arr = np.array(sorted(candidates))
    diffs = np.diff(arr)
    median_sp = float(np.median(diffs))
    if median_sp < 30 or median_sp > 80:
        return []
    best_inliers = 0
    best_slope = best_intercept = None
    best_inlier_idx = None
    for anchor in arr[:min(15, len(arr))]:
        for period_off in np.arange(-2.0, 2.5, 0.25):
            period = median_sp + period_off
            if period <= 0:
                continue
            indices = (arr - anchor) / period
            rounded = np.round(indices).astype(int)
            err = np.abs(indices - rounded)
            mask = err < tol
            inliers = arr[mask]
            inlier_idx = rounded[mask]
            n = len(inliers)
            if n < 15:
                continue
            if n > best_inliers:
                A = np.vstack([inlier_idx, np.ones(n)]).T
                slope, intercept = np.linalg.lstsq(A, inliers, rcond=None)[0]
                best_inliers = n
                best_slope = float(slope)
                best_intercept = float(intercept)
                best_inlier_idx = inlier_idx
    if best_slope is None:
        return []

    # Anchor stripe 0:
    if form_top_y is not None:
        # The first stripe sits ~5–25 px below the form's top edge. Find the
        # integer index k such that intercept + k*slope ≈ form_top_y + 15;
        # then re-base intercept so index 0 maps to that location.
        k = round((form_top_y + 15.0 - best_intercept) / best_slope)
        intercept = best_intercept + k * best_slope
    else:
        # Fall back to the topmost detected stripe.
        intercept = best_intercept + int(best_inlier_idx.min()) * best_slope
    return [int(round(intercept + i * best_slope)) for i in range(expected)]


def filter_stripes(mids: list[int], expected: int = EXPECTED_STRIPES,
                    tol: float = 0.15) -> list[int]:
    """Pick the longest run of stripes that fit a regular spacing grid.
    Returns a list of stripe Y centers, expected size or empty if no good
    sequence found. tol = allowed fractional deviation from median spacing."""
    if len(mids) < expected // 2:
        return []
    arr = np.array(sorted(mids))
    diffs = np.diff(arr)
    if len(diffs) == 0:
        return []
    median_sp = float(np.median(diffs))
    if median_sp < 30 or median_sp > 80:
        return []
    # Greedy: walk through mids, keeping ones that are ~median_sp from the
    # last kept stripe.
    keep = [arr[0]]
    for v in arr[1:]:
        gap = v - keep[-1]
        # Allow gap to be close to median_sp OR a multiple (1x, 2x, 3x for
        # small detection misses).
        for n in (1, 2, 3):
            target = median_sp * n
            if abs(gap - target) < tol * target:
                # If gap is a multiple, fill in the missing positions with
                # interpolated values so the final list has uniform spacing.
                if n == 1:
                    keep.append(int(v))
                else:
                    for k in range(1, n):
                        keep.append(int(keep[-1] + median_sp))
                    keep.append(int(v))
                break
    return keep


def map_stripes_to_canonical(stripe_ys: list[int], quad: np.ndarray,
                              canon_w: int = CANON_W,
                              canon_h: int = CANON_H) -> list[float]:
    """Project each stripe's original Y coordinate through the same
    perspective transform used for the form warp. The stripes are assumed
    to lie on the form's left edge (x ≈ left edge of quad), so we use the
    quad's left-edge X for the transform input."""
    quad = order_quad(quad)
    side1 = np.linalg.norm(quad[0] - quad[1])
    side2 = np.linalg.norm(quad[1] - quad[2])
    if side1 > side2:
        quad = np.array([quad[1], quad[2], quad[3], quad[0]], dtype=np.float32)
    dst = np.array([[0, 0], [canon_w - 1, 0],
                    [canon_w - 1, canon_h - 1], [0, canon_h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    # Use the LEFT edge X of the form quad (avg of TL.x and BL.x).
    left_x = (quad[0][0] + quad[3][0]) / 2.0
    pts = np.array([[[left_x, y] for y in stripe_ys]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, M)
    return [float(p[1]) for p in transformed[0]]


def _stripe_uniformity_score(warped: np.ndarray) -> int:
    """Score how 'upright' a warped image is by counting timing-mark
    candidates in the LEFT margin AND how well they fit a uniform
    grid. Random binding-hole patterns produce raw counts but their
    spacing is irregular; the form's true 47 stripes have rigorously
    even spacing.

    Returns the number of candidates that fall on a uniform-spacing
    line through the data. Higher = more upright."""
    candidates = _stripe_candidates(warped, threshold=100)
    if len(candidates) < 10:
        return 0
    arr = np.array(sorted(candidates), dtype=float)
    diffs = np.diff(arr)
    median_sp = float(np.median(diffs))
    if median_sp < 25 or median_sp > 80:
        return 0
    # Best linear fit y = a + b*i — count inliers within 0.25 * spacing
    best_inliers = 0
    for anchor in arr[:min(8, len(arr))]:
        for sp in (median_sp - 1, median_sp, median_sp + 1):
            indices = (arr - anchor) / sp
            err = np.abs(indices - np.round(indices))
            inliers = int((err < 0.25).sum())
            if inliers > best_inliers:
                best_inliers = inliers
    return best_inliers


FIXED_STRIPE_FIRST_Y = 60.0    # canonical y of stripe index 0 after rectify
FIXED_STRIPE_SPACING = 37.0    # canonical spacing between consecutive stripes


def _rectify_to_fixed_stripes(warped: np.ndarray) -> np.ndarray:
    """Re-scale `warped` along Y so that the form's 47 timing stripes
    land at FIXED canonical positions (stripe[i] at FIXED_STRIPE_FIRST_Y
    + i * FIXED_STRIPE_SPACING). After this every scan has the same
    canonical Y axis — bubble row positions are no longer scan-dependent.

    Robustly fits a linear y = a + b*i model to inlier stripe candidates
    (uniform-spacing constraint), then computes a Y-only affine that
    maps fitted stripe 0 → FIXED_STRIPE_FIRST_Y and fitted stripe 46 →
    FIXED_STRIPE_FIRST_Y + 46*FIXED_STRIPE_SPACING.
    """
    candidates = _stripe_candidates(warped, threshold=100)
    if len(candidates) < 25:
        return warped
    arr = np.array(sorted(candidates), dtype=float)
    diffs = np.diff(arr)
    median_sp = float(np.median(diffs))
    if median_sp < 25 or median_sp > 80:
        return warped
    # Find the anchor offset that gives the most uniform-grid inliers.
    best_inliers = 0
    best_slope = best_intercept = None
    for anchor in arr[:min(15, len(arr))]:
        for sp_off in np.arange(-1.5, 2.0, 0.5):
            sp = median_sp + sp_off
            if sp <= 0:
                continue
            indices = (arr - anchor) / sp
            rounded = np.round(indices).astype(int)
            err = np.abs(indices - rounded)
            mask = err < 0.20
            n = int(mask.sum())
            if n < 20:
                continue
            if n > best_inliers:
                A = np.vstack([rounded[mask], np.ones(n)]).T
                slope, intercept = np.linalg.lstsq(A, arr[mask],
                                                    rcond=None)[0]
                # Re-anchor so smallest detected index becomes index 0
                min_idx = int(rounded[mask].min())
                intercept = intercept + min_idx * slope
                best_inliers = n
                best_slope = float(slope)
                best_intercept = float(intercept)
    if best_slope is None:
        return warped
    first_src = best_intercept           # fitted source y for stripe 0
    last_src = best_intercept + 46 * best_slope
    last_dst = FIXED_STRIPE_FIRST_Y + 46 * FIXED_STRIPE_SPACING
    if last_src - first_src < 100:
        return warped
    scale = (last_dst - FIXED_STRIPE_FIRST_Y) / (last_src - first_src)
    offset = FIXED_STRIPE_FIRST_Y - scale * first_src
    inv_scale = 1.0 / scale
    inv_offset = -offset / scale
    H, W = warped.shape[:2]
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, inv_scale, inv_offset]], dtype=np.float32)
    out = cv2.warpAffine(warped, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255))
    return out


# Fixed canonical anchor positions for the homography-based warp.
# These are the destination points for: 4 fiducials + first/last timing
# stripe. After warp, every scan produces these EXACT canonical positions,
# regardless of the original DPI/tilt/translation.
FID_CANON = np.array([
    [175.0, 30.0], [213.0, 30.0],     # left pair
    [985.0, 30.0], [1062.0, 30.0],    # right pair
], dtype=np.float32)
STRIPE_FIRST_CANON = np.array([15.0, 50.0], dtype=np.float32)
STRIPE_LAST_CANON  = np.array([15.0, 1742.0], dtype=np.float32)


def _detect_raw_fiducials(gray: np.ndarray, y_tol: float = 80.0) -> np.ndarray | None:
    """Find the 4 black fiducial squares in raw scan coords.

    Strategy: top half of page, threshold-and-connected-components, filter
    to fiducial size range (area 600-2500, ~30-60 px square). Group blobs
    by Y proximity (allowing up to y_tol px difference for tilted scans);
    take the topmost group of >= 4 blobs and pick its two leftmost + two
    rightmost, sorted left-to-right.
    """
    H = gray.shape[0]
    top = gray[:int(H * 0.5), :]
    _, binary = cv2.threshold(top, 100, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 600 < area < 2500 and 25 < w < 60 and 20 < h < 50:
            blobs.append((float(cents[i][0]), float(cents[i][1])))
    if len(blobs) < 4:
        return None
    blobs.sort(key=lambda b: b[1])  # sort by y
    for i in range(len(blobs)):
        cy_i = blobs[i][1]
        row = [b for b in blobs if abs(b[1] - cy_i) < y_tol]
        if len(row) >= 4:
            row.sort(key=lambda b: b[0])
            return np.array([row[0], row[1], row[-2], row[-1]],
                            dtype=np.float32)
    return None


def _detect_raw_stripe_endpoints(gray: np.ndarray, fid_x: float
                                  ) -> tuple[int, int, int] | None:
    """Find the FIRST and LAST timing-stripe Y in raw scan coords, plus
    the X column they're at. Stripes are printed in a column ~150 px to
    the left of the first fiducial (in 300 DPI scans)."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return None
    x_center = max(70, int(fid_x) - 150)
    x0 = max(0, x_center - 20)
    x1 = min(gray.shape[1], x_center + 20)
    strip = gray[:, x0:x1]
    inverted = 255.0 - strip.mean(axis=1)
    peaks, _ = find_peaks(inverted, distance=30, prominence=20)
    if len(peaks) < 5:
        return None
    return int(peaks[0]), int(peaks[-1]), (x0 + x1) // 2


def warp_omr_homography(img: np.ndarray):
    """Deterministic warp using a 5-point AFFINE fit:
        4 fiducials + last stripe Y → fixed canonical positions.

    Affine (not perspective) because we only have a bottom-LEFT anchor
    (the timing-stripe column at the form's bottom-left); without a
    bottom-right anchor an unconstrained perspective fit lets that
    corner drift. Affine assumes the form is a parallelogram (scale +
    rotation + shear + translation), which holds for typical scanner
    output — pages aren't tilted in 3D.

    After warp, every scan's bubble positions land at IDENTICAL canonical
    coordinates — no per-scan template refit needed.

    Returns (warped_image, M3x3, rotation). M3x3 is the 3x3 homogeneous
    matrix form of the affine transform (raw → canonical).
    rotation is always None. Returns (None, None, None) on failure.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    fids = _detect_raw_fiducials(gray)
    if fids is None:
        return None, None, None
    stripes = _detect_raw_stripe_endpoints(gray, fids[0][0])
    if stripes is None:
        return None, None, None
    _first_y, last_y, sx = stripes
    src = np.vstack([fids, [[sx, last_y]]]).astype(np.float32)
    dst = np.vstack([FID_CANON, [STRIPE_LAST_CANON]]).astype(np.float32)
    # Full affine (6 DOF): scale_x, scale_y, shear_x, shear_y, tx, ty.
    # estimateAffine2D handles >3 points via RANSAC + least squares.
    M_2x3, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                     ransacReprojThreshold=8.0)
    if M_2x3 is None:
        return None, None, None
    warped = cv2.warpAffine(img, M_2x3, (CANON_W, CANON_H))
    # Promote to 3x3 homogeneous so callers can invert it consistently
    # with the homography path.
    M_3x3 = np.vstack([M_2x3, [0.0, 0.0, 1.0]]).astype(np.float64)
    return warped, M_3x3, None


def warp_omr(img: np.ndarray, return_matrix: bool = False):
    """Warp scan to a canonical 1200x1800 image with the form upright.

    Primary path: 6-point homography (`warp_omr_homography`) anchored on
    the 4 black fiducial squares + first/last timing stripe. This is
    DETERMINISTIC — every scan produces identical canonical coordinates,
    so the template can be hardcoded.

    Fallback path: if fiducial/stripe detection fails (no fiducials
    visible, low contrast, etc.), use the legacy red-mask quad warp +
    4-rotation orientation pick.

    If `return_matrix=True`, also return the perspective matrix M
    (canonical → raw, useful for inverse-mapping annotations onto the
    original page) and the final cardinal rotation applied (None for the
    homography path which doesn't need rotation).
    """
    # Note: a deterministic alternative is available as
    # `warp_omr_homography` (5-point affine fit anchored on the 4
    # fiducials + last timing stripe). It produces identical canonical
    # positions across scans, but the template positions need to be
    # re-tuned to that layout. For now, the legacy quad-based warp +
    # 4-rotation pick is used because the existing template is
    # calibrated against it.
    quad = find_form_quad(img)
    if quad is None:
        return (None, None, None) if return_matrix else None
    base, M_base, ordered_quad = warp_to_canonical(img, quad, return_matrix=True)

    candidates = [
        (base, None, "0"),
        (cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE), cv2.ROTATE_90_CLOCKWISE, "90cw"),
        (cv2.rotate(base, cv2.ROTATE_180), cv2.ROTATE_180, "180"),
        (cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE),
         cv2.ROTATE_90_COUNTERCLOCKWISE, "90ccw"),
    ]
    best_img = base
    best_rot = None
    best_score = -1
    for img_r, rot, _name in candidates:
        if img_r.shape[0] < img_r.shape[1]:
            continue
        s = _stripe_uniformity_score(img_r)
        if s > best_score:
            best_score = s
            best_img = img_r
            best_rot = rot

    if return_matrix:
        return best_img, M_base, best_rot
    return best_img


def canonical_to_original(canonical_pts: np.ndarray, M_canon_to_orig: np.ndarray,
                          rotation: int | None = None,
                          warped_shape: tuple[int, int] = (CANON_H, CANON_W)
                          ) -> np.ndarray:
    """Map an array of (x, y) points in the warped canonical image
    back to the original-image coordinate system.

    Steps:
      1. Undo any final rotation (best_rot from warp_omr) so points sit
         in the un-rotated canonical canvas.
      2. Apply M (the perspectiveTransform matrix, which already maps
         canonical → original because it was built as
         getPerspectiveTransform(orig_quad, canonical_quad)).

    Returns an Nx2 float array of original-image (x, y).
    """
    pts = np.asarray(canonical_pts, dtype=np.float32).reshape(-1, 2)
    H_warp, W_warp = warped_shape

    # Step 1: undo rotation. Each rotation maps a point in the rotated
    # frame back to the un-rotated base frame.
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        # 90 CW rotation: rotated[x', y'] = base[y', H - 1 - x']
        # So a point (x', y') in rotated → (y', W_unrotated - 1 - x') in base
        # where W_unrotated = H_warp (since rotation swaps dims).
        new_pts = np.column_stack([pts[:, 1], W_warp - 1 - pts[:, 0]])
        pts = new_pts
    elif rotation == cv2.ROTATE_180:
        new_pts = np.column_stack([W_warp - 1 - pts[:, 0],
                                    H_warp - 1 - pts[:, 1]])
        pts = new_pts
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        new_pts = np.column_stack([H_warp - 1 - pts[:, 1], pts[:, 0]])
        pts = new_pts

    # Step 2: invert M. M is the perspectiveTransform matrix from
    # `getPerspectiveTransform(quad, canonical_dst)`, which maps original
    # quad corners → canonical (0,0)/(W-1,0)/etc. So to get from canonical
    # back to original, invert M.
    M_inv = np.linalg.inv(M_canon_to_orig)
    pts_3 = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    transformed = (M_inv @ pts_3.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed
