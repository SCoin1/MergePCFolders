
import os
import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# Layout, as fractions of the square output side. Measured from the
# reference images the seller already uses.
MARGIN = 0.11          # empty border around everything
COIN_GAP = 0.004       # tiny gap between the two coin circles


class MergeError(Exception):
    """A problem we can explain to the user in plain words."""


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_image(path):
    """Read any common image (respects phone EXIF rotation). Returns BGR."""
    data = np.fromfile(path, dtype=np.uint8)  # works with non-ASCII paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise MergeError(f"Could not open image: {os.path.basename(path)}")
    return img


def _downscale(img, target=900):
    h, w = img.shape[:2]
    s = min(1.0, target / float(max(h, w)))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img, s


# --------------------------------------------------------------------------- #
#  Coin close-up  ->  circle
# --------------------------------------------------------------------------- #
def _fit_circle(pts):
    """Least-squares circle through points (Kasa fit)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2.0, c[1] / 2.0
    r = np.sqrt(max(c[2] + cx * cx + cy * cy, 1e-6))
    return cx, cy, r


def _hough_candidates(gray):
    h, w = gray.shape
    m = min(h, w)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    out = []
    for p2 in (60, 45, 32, 24):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m * 0.05,
                             param1=110, param2=p2,
                             minRadius=int(m * 0.30), maxRadius=int(m * 0.62))
        if c is not None:
            out.extend([tuple(v) for v in c[0][:8]])
        if len(out) >= 4:
            break
    return out


def _radial_profiles(gray, cx, cy, r_lo, r_hi, n_ang=180, n_r=120):
    """Sample |d/dr| of brightness along rays. Returns (rs, grad[n_ang, n_r],
    valid[n_ang, n_r], angles)."""
    h, w = gray.shape
    rs = np.linspace(r_lo, r_hi, n_r).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False).astype(np.float32)
    xs = cx + np.outer(np.cos(ang), rs)
    ys = cy + np.outer(np.sin(ang), rs)
    valid = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
    vals = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    step = max(1e-6, float(rs[1] - rs[0])) if len(rs) > 1 else 1.0
    # brightness change per PIXEL (independent of how finely we sample)
    grad = np.abs(np.gradient(vals, axis=1)) / step * valid
    return rs, grad, valid, ang


def _coverage(gray, cx, cy, r, band=0.025, n_ang=180):
    """Fraction of rays that have a clear edge within +-band of radius r.
    Rays that leave the photo don't count against the circle."""
    rs, grad, valid, _ = _radial_profiles(gray, cx, cy, r * (1 - band),
                                          r * (1 + band), n_ang, 12)
    inside = valid.all(axis=1)
    if inside.sum() < n_ang * 0.35:
        return 0.0
    peak = grad.max(axis=1)
    return float(((peak > 7) & inside).sum()) / inside.sum() * \
        min(1.0, inside.sum() / (n_ang * 0.6))


def _rim_points(gray, cx, cy, r, band=0.06, n_ang=360):
    """Strongest edge per ray within +-band of r (in image coordinates)."""
    rs, grad, valid, ang = _radial_profiles(gray, cx, cy, r * (1 - band),
                                            r * (1 + band), n_ang, 60)
    i = grad.argmax(axis=1)
    pk = grad[np.arange(n_ang), i]
    ok = (pk > 7) & valid.all(axis=1)
    rr = rs[i][ok]
    return np.column_stack([cx + rr * np.cos(ang[ok]),
                            cy + rr * np.sin(ang[ok])]).astype(np.float64)


def _outer_rim_points(gray, cx, cy, r, lo=0.97, hi=1.12, n_ang=360):
    """Per ray, the OUTERMOST clear edge in [lo*r, hi*r]. The coin's raised
    rim gives an inner edge a few % inside the true outline; we want the
    outline (metal -> holder), so take the last strong edge going out."""
    rs, grad, valid, ang = _radial_profiles(gray, cx, cy, r * lo, r * hi,
                                            n_ang, 90)
    pts = []
    for k in range(n_ang):
        g = grad[k]
        if not valid[k].all():
            g = g * valid[k]
        mx = g.max()
        if mx < 7:
            continue
        # local maxima that are reasonably strong
        pk = np.where((g[1:-1] >= g[:-2]) & (g[1:-1] >= g[2:]) &
                      (g[1:-1] >= max(7, 0.45 * mx)))[0] + 1
        if len(pk) == 0:
            continue
        i = pk[-1]
        pts.append((cx + rs[i] * np.cos(ang[k]), cy + rs[i] * np.sin(ang[k])))
    return np.array(pts, dtype=np.float64)


def _robust_fit(pts):
    for _k in range(4):
        cx, cy, r = _fit_circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = res < max(1.0, np.median(res) * 2.5)
        if keep.sum() < 30 or keep.all():
            break
        pts = pts[keep]
    return _fit_circle(pts)


def _refine(gray, cx, cy, r):
    # 1) jump out to the coin outline
    pts = _outer_rim_points(gray, cx, cy, r)
    if len(pts) >= 60:
        ncx, ncy, nr = _robust_fit(pts)
        if 0.95 * r < nr < 1.13 * r and np.hypot(ncx - cx, ncy - cy) < 0.08 * r:
            cx, cy, r = ncx, ncy, nr
    # 2) tighten on it
    for band in (0.025, 0.012):
        pts = _rim_points(gray, cx, cy, r, band)
        if len(pts) < 40:
            break
        ncx, ncy, nr = _robust_fit(pts)
        if abs(nr - r) > r * band or np.hypot(ncx - cx, ncy - cy) > r * band:
            break
        cx, cy, r = ncx, ncy, nr
    return cx, cy, r


def _backdrop_candidates(small):
    """Round objects lying on a plain backdrop (coin taken out of the
    holder). The backdrop may be unevenly lit, so it is modelled as a
    smooth surface fitted to the photo's border, not a single colour."""
    h, w = small.shape[:2]
    img = small.astype(np.float32)
    bw = max(3, int(min(h, w) * 0.03))
    ring = np.zeros((h, w), bool)
    ring[:bw], ring[-bw:], ring[:, :bw], ring[:, -bw:] = True, True, True, True
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / w - 0.5, yy / h - 0.5
    A_all = np.stack([np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn], -1)
    A = A_all[ring]
    model = np.zeros_like(img)
    keep = np.ones(len(A), bool)
    for _ in range(3):              # robust: ignore things touching the edge
        resid = np.zeros(len(A), np.float32)
        coefs = []
        for ch in range(3):
            v = img[:, :, ch][ring]
            c, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
            coefs.append(c)
            resid = np.maximum(resid, np.abs(A @ c - v))
        mad = float(np.median(resid[keep])) + 1e-3
        keep = resid < max(6.0, 3.0 * mad)
    if keep.mean() < 0.6 or mad > 8:    # border isn't a plain backdrop
        return []
    for ch in range(3):
        model[:, :, ch] = A_all @ coefs[ch]
    diff = np.abs(img - model).max(axis=2)
    fg = (diff > max(16.0, 5.0 * mad)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(c)
        if area < h * w * 0.03:
            break
        hull = cv2.convexHull(c)
        (cx, cy), r = cv2.minEnclosingCircle(hull)
        if cv2.contourArea(hull) / (np.pi * r * r) > 0.6:
            solid = np.zeros_like(fg)          # silhouette without holes
            cv2.drawContours(solid, [c], -1, 255, -1)
            out.append((cx, cy, r, solid))
    return out


def _find_coin_on_backdrop(gray, small):
    """Per ray from the blob centre: walk out until the photo looks like
    backdrop again (silhouette). On the lit side that IS the coin edge;
    on the shadow side it's the end of the shadow, too far. So take the
    nearest crisp edge at or inside the silhouette point, then fit."""
    h, w = gray.shape
    best = None
    for bx, by, br, fg in _backdrop_candidates(small):
        n_ang, n_r = 360, 160
        rs, grad, valid, ang = _radial_profiles(gray, bx, by, br * 0.55,
                                                br * 1.08, n_ang, n_r)
        xs = (bx + np.outer(np.cos(ang), rs)).round().astype(int)
        ys = (by + np.outer(np.sin(ang), rs)).round().astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        f = np.zeros_like(inside)
        f[inside] = fg[ys[inside], xs[inside]] > 0
        pts = []
        for k in range(n_ang):
            if not inside[k].all():
                continue            # ray leaves the photo: skip
            off = np.where(~f[k])[0]
            if len(off) == 0:
                continue
            i_sil = off[0]
            g = grad[k]
            pk = np.where((g[1:-1] >= g[:-2]) & (g[1:-1] >= g[2:]) &
                          (g[1:-1] >= 6))[0] + 1
            pk = pk[pk <= i_sil + 2]
            if len(pk) == 0:
                continue
            i = pk[-1]
            pts.append((bx + rs[i] * np.cos(ang[k]),
                        by + rs[i] * np.sin(ang[k])))
        if len(pts) < 60:
            continue
        cx, cy, r = _robust_fit(np.array(pts, np.float64))
        if not (0.55 * br < r < 1.05 * br):
            continue
        cov = _coverage(gray, cx, cy, r, band=0.015, n_ang=360)
        if best is None or cov > best[0]:
            best = (cov, cx, cy, r)
    if best and best[0] >= 0.4:
        return best[1:]
    return None


def find_coin(img):
    """Return (cx, cy, r) of the coin in a close-up photo, full-res pixels."""
    small, s = _downscale(img, 900)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                            (0, 0), 1.0).astype(np.float32)
    h, w = gray.shape
    m = min(h, w)

    on_backdrop = _find_coin_on_backdrop(gray, small)
    if on_backdrop is not None:
        cx, cy, r = on_backdrop
        return cx / s, cy / s, r / s

    cands = list(_hough_candidates(gray.astype(np.uint8)))
    # Radial sweep around the frame centre: every radius where a ring
    # of edges exists becomes a candidate too.
    for r in np.linspace(m * 0.30, m * 0.62, 70):
        cands.append((w / 2.0, h / 2.0, r))

    scored = []
    for cx, cy, r in cands:
        off = np.hypot(cx - w / 2, cy - h / 2) / m
        if off > 0.25:
            continue
        scored.append((_coverage(gray, cx, cy, r), cx, cy, r))
    if not scored:
        raise MergeError("Could not find the coin in a close-up photo.")
    scored.sort(reverse=True)

    # Refine the best few candidates and keep the one whose outline is
    # most complete.
    results = []
    for cov, cx, cy, r in scored[:6]:
        cx, cy, r = _refine(gray, cx, cy, r)
        results.append((_coverage(gray, cx, cy, r, band=0.012, n_ang=360),
                        cx, cy, r))
    top, cx, cy, r = max(results)
    if top < 0.3:
        raise MergeError("Could not find a clear coin edge in a close-up "
                         "photo.")
    return cx / s, cy / s, r / s


def cut_coin(img, circle, size, trim=0.004, rot=0.0):
    """Cut the coin out as a size x size BGRA image with an anti-aliased
    circular alpha. `trim` shaves a hair off the rim so no background
    or holder prong shows at the edge; `rot` turns the coin (degrees,
    counter-clockwise)."""
    cx, cy, r = circle
    r_in = r * (1.0 - trim)
    scale = (size / 2.0) / r_in
    # rotate + scale about the coin centre, then move it to the tile centre
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot), scale)
    M[0, 2] += size / 2.0 - cx
    M[1, 2] += size / 2.0 - cy
    tile = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA
                          if scale < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    return np.dstack([tile, _circle_alpha(size)])


def _circle_alpha(size, ss=4):
    big = np.zeros((size * ss, size * ss), np.uint8)
    c = size * ss // 2
    cv2.circle(big, (c, c), c - ss, 255, -1, lineType=cv2.LINE_AA)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
#  Slab photo  ->  cut-out slab
# --------------------------------------------------------------------------- #
def _background_mask(small):
    """Pixels that belong to the photo background: similar in colour to the
    image border AND connected to the border."""
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])

    # border may contain 1-2 colours (e.g. a screenshot frame + backdrop)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(1234)            # same photo -> same result, every time
    _, labels, centers = cv2.kmeans(border, 2, None, crit, 3,
                                     cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=2)
    cols = [centers[i] for i in range(2) if counts[i] > len(border) * 0.08]

    near = np.zeros((h, w), bool)
    for c in cols:
        d = np.linalg.norm(lab - c, axis=2)
        near |= d < 14
    near = near.astype(np.uint8)

    # keep only the parts touching the border
    n, cc = cv2.connectedComponents(near, connectivity=4)
    edge_ids = set(np.unique(np.concatenate(
        [cc[0], cc[-1], cc[:, 0], cc[:, -1]])).tolist()) - {0}
    bg = np.isin(cc, list(edge_ids)) & (near > 0)
    return bg


PLASTIC_MIN_ALPHA = 0.22   # how "milky" the clear slab border stays


def find_slab(img, plastic=False):
    """Return (mask, angle, box) for the slab: mask is full-res uint8.

    With plastic=True also returns (band, bg_bgr): band is a full-res bool
    mask of the clear plastic border (inside the slab outline but the same
    colour as the photo backdrop), bg_bgr the backdrop colour. Those pixels
    are made see-through so the chosen background shows through them."""
    small, s = _downscale(img, 1000)
    h, w = small.shape[:2]
    bg = _background_mask(small)

    fg = (~bg).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise MergeError("Could not find the slab in the slab photo.")
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < h * w * 0.08:
        raise MergeError("The slab looks too small in the slab photo - "
                         "please use a photo where the slab fills the frame.")

    # The clear plastic border of a slab can be almost the same colour as
    # a white/black backdrop. Its outline still shows as thin edges, so
    # grow the slab with any edges that touch it.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.0), 25, 70)
    body = np.zeros((h, w), np.uint8)
    cv2.drawContours(body, [big], -1, 255, -1)
    grown = cv2.dilate(edges, np.ones((3, 3), np.uint8)) | body
    n, cc = cv2.connectedComponents(grown, connectivity=8)
    ids = np.unique(cc[body > 0])
    ids = ids[ids > 0]
    allpts = cv2.findNonZero(np.isin(cc, ids).astype(np.uint8))
    hull = cv2.convexHull(allpts)
    # Safety: if edges ran away into a busy backdrop, fall back.
    if cv2.contourArea(hull) > cv2.contourArea(cv2.convexHull(big)) * 1.6 \
            or cv2.contourArea(hull) > h * w * 0.97:
        hull = cv2.convexHull(big)
    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(hull)
    # normalise angle to a small tilt around portrait orientation
    if rw > rh:
        rw, rh = rh, rw
        ang -= 90
    while ang > 45:
        ang -= 90
    while ang < -45:
        ang += 90

    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1, lineType=cv2.LINE_AA)
    H, W = img.shape[:2]
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    k = max(1, int(round(1.0 / s)))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    box = (rcx / s, rcy / s, rw / s, rh / s)
    if not plastic:
        return mask, ang, box

    hull_small = np.zeros((h, w), np.uint8)
    cv2.drawContours(hull_small, [hull], -1, 255, -1)
    band_small = (hull_small > 0) & (body == 0)
    bg_bgr = np.median(small[bg], axis=0) if bg.any() else \
        np.array([255, 255, 255], np.float64)
    band = cv2.resize(band_small.astype(np.uint8) * 255, (W, H),
                      interpolation=cv2.INTER_LINEAR) > 127
    return mask, ang, box, (band, bg_bgr)


def _edge_line(profile, lo, hi, from_start):
    """Outermost strong peak of an edge-strength profile inside [lo, hi)."""
    seg = profile[lo:hi]
    if len(seg) < 5:
        return None
    base = np.median(profile)
    thr = max(base * 4.0, seg.max() * 0.30, 3.0)
    idx = np.where(seg > thr)[0]
    if len(idx) == 0:
        return None
    i = idx[0] if from_start else idx[-1]
    # move to the local peak of that run
    j = i
    step = 1 if from_start else -1
    while 0 <= j + step < len(seg) and seg[j + step] >= seg[j]:
        j += step
    return lo + j


def find_slab_frame(img):
    """For tightly framed photos where the slab nearly fills the picture.
    Finds the slab's long straight outer edges; a side that runs off the
    photo uses the photo border. Returns (x0, y0, x1, y1, open_sides) in
    full-res pixels; open_sides = sides cut by the photo border."""
    small, s = _downscale(img, 1000)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                            (0, 0), 1.2).astype(np.float32)
    h, w = gray.shape
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    # long straight edges: average along the middle 70% of the other axis
    colp = np.median(gx[int(h * 0.15):int(h * 0.85)], axis=0)
    rowp = np.median(gy[:, int(w * 0.15):int(w * 0.85)], axis=1)
    colp = np.convolve(colp, np.ones(3) / 3, "same")
    rowp = np.convolve(rowp, np.ones(3) / 3, "same")
    m = 3                                   # ignore the very border pixels
    x0 = _edge_line(colp, m, int(w * 0.3), True)
    x1 = _edge_line(colp, int(w * 0.7), w - m, False)
    y0 = _edge_line(rowp, m, int(h * 0.3), True)
    y1 = _edge_line(rowp, int(h * 0.7), h - m, False)
    open_sides = []
    if x0 is None:
        x0 = 0; open_sides.append("left")
    if x1 is None:
        x1 = w; open_sides.append("right")
    if y0 is None:
        y0 = 0; open_sides.append("top")
    if y1 is None:
        y1 = h; open_sides.append("bottom")
    if (x1 - x0) < w * 0.45 or (y1 - y0) < h * 0.45:
        raise MergeError("Could not find the slab edges.")
    return (x0 / s, y0 / s, x1 / s, y1 / s, open_sides)


def _rounded_rect_mask(H, W, rect, open_sides=(), radius_frac=0.075):
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    rad = int(round((x1 - x0) * radius_frac))
    ss = 4 if max(H, W) < 2500 else 2          # anti-aliased corners
    small = np.zeros((H // ss + 1, W // ss + 1), np.uint8)
    X0, Y0, X1, Y1, R = x0 // ss, y0 // ss, x1 // ss, y1 // ss, max(1, rad // ss)
    cv2.rectangle(small, (X0 + R, Y0), (X1 - R, Y1), 255, -1)
    cv2.rectangle(small, (X0, Y0 + R), (X1, Y1 - R), 255, -1)
    corners = {("left", "top"): (X0 + R, Y0 + R), ("right", "top"): (X1 - R, Y0 + R),
               ("left", "bottom"): (X0 + R, Y1 - R), ("right", "bottom"): (X1 - R, Y1 - R)}
    for (sx, sy), c in corners.items():
        if sx in open_sides or sy in open_sides:   # cut by the photo: square
            bx = X0 if sx == "left" else X1 - R
            by = Y0 if sy == "top" else Y1 - R
            cv2.rectangle(small, (bx, by), (bx + R, by + R), 255, -1)
        else:
            cv2.circle(small, c, R, 255, -1, lineType=cv2.LINE_AA)
    mask = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    mask[:y0], mask[y1:], mask[:, :x0], mask[:, x1:] = 0, 0, 0, 0
    return mask


def _plastic_alpha(img, mask, rect, bg_bgr, edge_frac=0.055):
    """Clear plastic near the slab's outer edge shows the backdrop through
    it: make those pixels see-through (colour-to-alpha)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    band = np.zeros((H, W), bool)
    e = int(round((x1 - x0) * edge_frac))
    band[max(0, y0):y1, max(0, x0):x1] = True
    band[y0 + e:y1 - e, x0 + e:x1 - e] = False
    px = img[band].astype(np.float32)
    bgc = np.asarray(bg_bgr, np.float32)
    diff = np.abs(px - bgc).max(axis=1)
    near = diff < 40
    a = np.clip(PLASTIC_MIN_ALPHA + diff / 45.0, PLASTIC_MIN_ALPHA, 1.0)
    a[~near] = 1.0
    col = bgc + (px - bgc) / a[:, None]
    img = img.copy()
    img[band] = np.clip(col, 0, 255).astype(np.uint8)
    mask = mask.copy()
    mask[band] = (mask[band].astype(np.float32) * a).astype(np.uint8)
    return img, mask


def detect_slab(img):
    """Pick the best slab outline for this photo. Returns a dict with the
    rectangle (full-res x0, y0, x1, y1), open sides and method used."""
    H, W = img.shape[:2]
    frame = None
    try:
        frame = find_slab_frame(img)
    except MergeError:
        pass
    hull = None
    try:
        hull = find_slab(img)
    except MergeError:
        pass
    if hull is not None:
        hmask = hull[0] > 128
        ys, xs = np.where(hmask)
        hrect = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        harea = (hrect[2] - hrect[0]) * (hrect[3] - hrect[1])
    if frame is not None:
        farea = (frame[2] - frame[0]) * (frame[3] - frame[1])
        # the hull method only finds part of a tightly framed slab
        if hull is None or harea < farea * 0.8:
            return {"rect": [float(v) for v in frame[:4]],
                    "open": list(frame[4]), "method": "frame"}
    if hull is not None:
        return {"rect": [float(v) for v in hrect], "open": [],
                "method": "hull"}
    raise MergeError("Could not find the slab in the holder photo.")


def _backdrop_colour(img, rect):
    """Median colour outside the slab rectangle (None if hardly visible)."""
    H, W = img.shape[:2]
    small, s = _downscale(img, 600)
    x0, y0, x1, y1 = [int(v * s) for v in rect]
    m = np.ones(small.shape[:2], bool)
    m[max(0, y0):y1, max(0, x0):x1] = False
    if m.mean() < 0.01:
        return None
    return np.median(small[m], axis=0)


def cut_slab(img, slab=None):
    """Return the slab as a tight BGRA image. `slab` = result of
    detect_slab() or a manual {"rect": [...], "open": [...]}; detected
    automatically when not given. The clear plastic border is made
    semi-transparent, like real clear plastic."""
    if slab is None:
        slab = detect_slab(img)
    if slab.get("method") == "hull" and not slab.get("manual"):
        mask, ang, box, (band, bg_bgr) = find_slab(img, plastic=True)
        if band.any():
            img = img.copy()
            px = img[band].astype(np.float32)
            bgc = bg_bgr.astype(np.float32)
            diff = np.abs(px - bgc).max(axis=1)
            a = np.clip(PLASTIC_MIN_ALPHA + diff / 45.0, PLASTIC_MIN_ALPHA,
                        1.0)
            col = bgc + (px - bgc) / a[:, None]
            img[band] = np.clip(col, 0, 255).astype(np.uint8)
            mask = mask.copy()
            mask[band] = (mask[band].astype(np.float32) * a).astype(np.uint8)
        return _cut_object(img, (mask, ang, box))
    H, W = img.shape[:2]
    rect = slab["rect"]
    mask = _rounded_rect_mask(H, W, rect, slab.get("open", ()))
    bgc = _backdrop_colour(img, rect)
    if bgc is not None:
        img, mask = _plastic_alpha(img, mask, rect, bgc)
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    out = np.dstack([img[y0:y1, x0:x1], mask[y0:y1, x0:x1]])
    return out


def find_rect_item(img):
    """For rectangular items (stamps, cards, notes): find the item on a
    plain backdrop. Returns find_slab-style result, or None when the item
    fills the photo (no backdrop) - then the whole photo is used."""
    try:
        mask, ang, box = find_slab(img)
    except MergeError:
        return None
    H, W = img.shape[:2]
    frac = (mask > 128).mean()
    _cx, _cy, rw, rh = box
    rectness = (mask > 128).sum() / max(1.0, rw * rh)
    if frac < 0.15 or frac > 0.93 or rectness < 0.85:
        return None
    return mask, ang, box


def cut_rect_item(img):
    found = find_rect_item(img)
    if found is None:
        return np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return _cut_object(img, found)


def _cut_object(img, found):
    mask, ang, (cx, cy, rw, rh) = found
    H, W = img.shape[:2]
    rgba = np.dstack([img, mask])

    if abs(ang) > 0.3 and abs(ang) < 20:
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rgba = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    a = rgba[:, :, 3]
    ys, xs = np.where(a > 24)   # any opacity: clear border counts
    if len(xs) == 0:
        raise MergeError("Could not find the slab in the slab photo.")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = rgba[y0:y1, x0:x1].copy()
    # slightly soften the cut edge
    out[:, :, 3] = cv2.GaussianBlur(out[:, :, 3], (0, 0), 0.7)
    return out


# --------------------------------------------------------------------------- #
#  Composition
# --------------------------------------------------------------------------- #
def _paste(canvas, tile, x, y):
    """Alpha-composite BGRA tile onto float BGRA canvas at (x, y)."""
    th, tw = tile.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + tw), min(H, y + th)
    if x1 <= x0 or y1 <= y0:
        return
    t = tile[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = t[:, :, 3:4] / 255.0
    region = canvas[y0:y1, x0:x1]
    region[:, :, :3] = t[:, :, :3] * a + region[:, :, :3] * (1 - a)
    region[:, :, 3:4] = 255 * a + region[:, :, 3:4] * (1 - a)


def _shadow(tile, blur):
    a = tile[:, :, 3].astype(np.float32)
    pad = int(blur * 3)
    a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    a = cv2.GaussianBlur(a, (0, 0), blur) * 0.35
    sh = np.zeros(a.shape + (4,), np.uint8)
    sh[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    return sh, pad


def _find_coin_named(img, which):
    try:
        return find_coin(img)
    except MergeError:
        raise MergeError(f"Could not find a round edge in the {which} "
                         "photo. Check that the right photo was picked and "
                         "that the whole item is in the picture. (For "
                         "stamps or other square items, set Item shape to "
                         "Rectangle.)")


def _fit(tile, bw, bh):
    """Resize BGRA tile to fit inside bw x bh, keeping its proportions."""
    th, tw = tile.shape[:2]
    sc = min(bw / float(tw), bh / float(th))
    nw, nh = max(1, int(round(tw * sc))), max(1, int(round(th * sc)))
    return cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_AREA
                      if sc < 1 else cv2.INTER_CUBIC)


def compose(slab_img, upper_img, lower_img, size=1600, bg=(255, 255, 255),
            transparent=False, shadow=False, trim=0.004, shape="round",
            upper_circle=None, lower_circle=None, progress=None,
            slab=None, upper_rot=0.0, lower_rot=0.0):
    """Build the final square image. Returns (BGR or BGRA image, info dict).

    shape = "round"  : close-ups are cut as circles (coins, medals, tokens)
            "rect"   : close-ups are kept rectangular (stamps, cards, notes)
    """
    def step(msg):
        if progress:
            progress(msg)

    m = int(round(size * MARGIN))
    content = size - 2 * m
    gap = int(round(size * COIN_GAP))
    r = (content - gap) // 4                 # circle radius
    d = 2 * r                                # each close-up box is d x d

    step("Finding the slab...")
    slab = cut_slab(slab_img, slab)
    sh, sw = slab.shape[:2]
    scale = content / float(sh)
    slab = cv2.resize(slab, (max(1, int(round(sw * scale))), content),
                      interpolation=cv2.INTER_AREA if scale < 1
                      else cv2.INTER_CUBIC)

    info = {}
    tiles = []
    for key, img, given, rot, label in (("upper", upper_img, upper_circle,
                                          upper_rot, "top close-up"),
                                         ("lower", lower_img, lower_circle,
                                          lower_rot, "bottom close-up")):
        step(f"Finding the {label}...")
        if shape == "rect":
            tiles.append(_fit(cut_rect_item(img), d, d))
        else:
            c = given or _find_coin_named(img, label)
            info[key + "_circle"] = [float(v) for v in c]
            tiles.append(cut_coin(img, c, d, trim, rot))

    step("Putting it together...")
    canvas = np.zeros((size, size, 4), np.float32)
    if not transparent:
        canvas[:, :, 0], canvas[:, :, 1], canvas[:, :, 2] = bg
        canvas[:, :, 3] = 255

    _paste(canvas, slab, m, m)
    for i, tile in enumerate(tiles):
        th, tw = tile.shape[:2]
        x = size - m - tw                     # right-aligned
        box_y = m if i == 0 else size - m - d
        y = box_y + (d - th) // 2             # centred in its half
        if shadow:
            s_img, pad = _shadow(tile, size * 0.008)
            off = int(size * 0.006)
            _paste(canvas, s_img, x - pad + off, y - pad + off)
        _paste(canvas, tile, x, y)

    out = np.clip(canvas + 0.5, 0, 255).astype(np.uint8)
    if not transparent:
        out = out[:, :, :3]
    return out, info


def detection_preview(img, circle=None, slab=False, rect=False,
                      max_side=420):
    """Small image with the detected circle / slab outline drawn on it,
    so the user can see what the program found."""
    small, s = _downscale(img, max_side)
    vis = small.copy()
    if circle is not None:
        cx, cy, r = [v * s for v in circle]
        cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)),
                   (0, 200, 0), 2, cv2.LINE_AA)
    found = None
    if slab:
        found = find_slab(img)
    elif rect:
        found = find_rect_item(img)
        if found is None:      # whole photo used
            cv2.rectangle(vis, (1, 1), (vis.shape[1] - 2, vis.shape[0] - 2),
                          (0, 200, 0), 2)
    if found is not None:
        mask = found[0]
        mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]))
        cnts, _ = cv2.findContours((mask > 128).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 0), 2, cv2.LINE_AA)
    return vis


def save_image(img, path):
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise MergeError("Could not save the output image.")
    buf.tofile(path)


# --------------------------------------------------------------------------- #
#  Coin inside a holder photo (cut the coin out of the slab photo itself)
# --------------------------------------------------------------------------- #
def _annulus_contrast(lab, cx, cy, r, sectors=24):
    """How different the ring just INSIDE the circle is from the ring just
    OUTSIDE it, per angular sector (median). A coin edge separates metal
    from plastic: big difference all round. A capsule ring or a stray arc
    has plastic on both sides: small difference."""
    h, w = lab.shape[:2]
    ang = np.linspace(0, 2 * np.pi, sectors * 6, endpoint=False)
    def ring(r0, r1):
        rs = np.linspace(r0, r1, 5)
        xs = (cx + np.outer(np.cos(ang), rs)).clip(0, w - 1).astype(int)
        ys = (cy + np.outer(np.sin(ang), rs)).clip(0, h - 1).astype(int)
        v = lab[ys, xs]                                   # (A, 5, 3)
        return v.reshape(sectors, -1, 3).mean(axis=1)     # per sector
    inner = ring(r * 0.78, r * 0.94)
    outer = ring(r * 1.06, r * 1.22)
    d = np.linalg.norm(inner - outer, axis=1)
    return float(np.median(d))


def find_coin_in_holder(img, debug=False):
    """Return (cx, cy, r) in full-res pixels of the coin seen through a
    holder (slab) photo."""
    small, s = _downscale(img, 1000)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                            (0, 0), 1.0).astype(np.float32)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.5),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = gray.shape
    # Search size bands separately, so a small coin isn't crowded out by
    # big random circles (capsule ring, label art, stickers).
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.uint8)
    cands = []
    bands = [(0.05, 0.09), (0.08, 0.13), (0.12, 0.19), (0.18, 0.27),
             (0.26, 0.36), (0.35, 0.45)]
    for lo, hi in bands:
        rmin, rmax = int(w * lo), int(w * hi)
        got = []
        for p1, p2 in ((90, 40), (60, 28), (40, 20)):
            c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                 minDist=max(8, rmin), param1=p1, param2=p2,
                                 minRadius=rmin, maxRadius=rmax)
            if c is not None:
                got.extend(tuple(v) for v in c[0][:8])
            if len(got) >= 8:
                break
        cands.extend(got[:10])
    # Gold / coloured coins also stand out by colour saturation.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)).astype(np.uint8) * 255
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
        (bx, by), br = cv2.minEnclosingCircle(cv2.convexHull(c))
        if br > w * 0.04 and cv2.contourArea(cv2.convexHull(c)) > 0.6 * np.pi * br * br:
            cands.append((bx, by, br))
    if not cands:
        raise MergeError("Could not find the coin in the holder photo.")

    # horizontal middle of the slab (fall back to the photo's middle)
    mid_x = w / 2.0
    try:
        _m, _a, (bx, _by, bw, _bh) = find_slab(small)
        if abs(bx - w / 2.0) < w * 0.2:
            mid_x = bx
    except MergeError:
        pass

    scored = []
    for cx, cy, r in cands:
        if cx - r < -0.05 * r or cy - r < -0.05 * r or \
                cx + r > w * 1.05 or cy + r > h * 1.05:
            continue                      # circle mostly outside the photo
        pts = _rim_points(gray, cx, cy, r, band=0.05)
        if len(pts) >= 40:
            ncx, ncy, nr = _robust_fit(pts)
            if abs(nr - r) < 0.08 * r and np.hypot(ncx - cx, ncy - cy) < 0.08 * r:
                cx, cy, r = ncx, ncy, nr
        cov = _coverage(gray, cx, cy, r, band=0.015, n_ang=360)
        con = _annulus_contrast(lab, cx, cy, r)
        # holders always hold the coin centred left-to-right; round things
        # off to the side are hologram stickers, logos, label art
        off = (cx - mid_x) / float(w)
        centred = float(np.exp(-(off / 0.12) ** 2))
        scored.append((cov * min(con, 40.0) * centred, cov, con, cx, cy, r))
    if not scored:
        raise MergeError("Could not find the coin in the holder photo.")
    scored.sort(reverse=True)
    if debug:
        for t in scored[:8]:
            print("   score %.1f cov %.2f con %.1f  c=(%d,%d) r=%d" %
                  (t[0], t[1], t[2], t[3] / s, t[4] / s, t[5] / s))
    _, cov, con, cx, cy, r = scored[0]
    cx, cy, r = _expand_to_outline(lab, cx, cy, r)
    return cx / s, cy / s, r / s


def _expand_to_outline(lab, cx, cy, r, n_ang=360):
    """The strongest circle is often the INNER edge of the raised rim. Walk
    out along each ray while pixels still look like the coin's rim metal;
    where they stop is the coin's true outline (rim + reeded edge kept)."""
    h, w = lab.shape[:2]
    rs = np.linspace(r * 0.9, r * 1.14, 72).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False).astype(np.float32)
    xs = cx + np.outer(np.cos(ang), rs)
    ys = cy + np.outer(np.sin(ang), rs)
    valid = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    prof = np.stack([cv2.remap(lab[:, :, c], xs.astype(np.float32),
                               ys.astype(np.float32), cv2.INTER_LINEAR)
                     for c in range(3)], -1)                 # (A, R, 3)
    i_in = rs <= r * 0.97                                    # just inside
    metal = np.median(prof[:, i_in].reshape(-1, 3), axis=0)  # rim colour
    i_out = rs >= r * 1.10
    plastic = np.median(prof[:, i_out].reshape(-1, 3), axis=0)
    gap = np.linalg.norm(metal - plastic)
    if gap < 12:                     # coin and holder look alike: keep r
        return cx, cy, r
    d = np.linalg.norm(prof - metal, axis=2)                 # (A, R)
    thr = gap * 0.5
    pts = []
    start = int(np.searchsorted(rs, r * 0.97))
    for k in range(n_ang):
        if not valid[k].all():
            continue
        far = d[k, start:] > thr
        # first place where it stops looking like metal for 3 samples
        run = np.convolve(far.astype(int), np.ones(3, int), "valid") == 3
        idx = np.argmax(run) if run.any() else -1
        if idx <= 0:
            continue
        rr = rs[start + idx]
        pts.append((cx + rr * np.cos(ang[k]), cy + rr * np.sin(ang[k])))
    if len(pts) < n_ang * 0.4:
        return cx, cy, r
    ncx, ncy, nr = _robust_fit(np.array(pts, np.float64))
    if r * 0.99 <= nr <= r * 1.12 and np.hypot(ncx - cx, ncy - cy) < r * 0.04:
        return ncx, ncy, nr
    return cx, cy, r
