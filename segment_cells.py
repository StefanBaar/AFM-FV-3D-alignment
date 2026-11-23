#!/usr/bin/env python3
"""
Multi-channel cell segmentation of AFM force-volume maps, tracked over time on the registered 3D surface.

Input: the folder written by extract_topo_E.py (one sub-folder per map, one .npy per frame):
    DATA/topo, DATA/E, DATA/alpha, DATA/delta, [DATA/adh_*], [DATA/excluded]
Frame names must end in _HHMMSS (time order). Best with  extract_topo_E.py NPZ --no-mask  (all fitted values).

Method (per frame); all length scales follow --cell-diam D (default 4 um, typical cell diameter)
  0. channels: adhesion force + work of adhesion if present (sharpest junction contrast in this data;
     --channels all uses every map: E, delta, alpha, topo, adhesion), otherwise all present
  1. every channel: log (E), outlier filter (isolated pixel > 4 robust SD beyond its 2nd most extreme neighbour -> neighbour median),
     scan-line offset removal, cubic resampling to a fine isotropic grid h (default min(D/16, px/2, py/2) = 0.25 um),
     resolution equalisation of the slow axis, removal of the large-scale trend (> D)
  2. ridge filter (Sato, 0.08-0.18 D) -> boundary evidence. Polarity: cell junctions are stiffer (E up,
     delta down), more adhesive (adhesion force/work up) and grooves in the topography; undecided
     channels (alpha, pull-off, ...) take the polarity that agrees with the anchor (E)
  3. channels are weighted by their agreement with the other channels (topography capped at 0.2,
     the colloidal probe blurs it) and combined pixel-wise where they have data
  4. seeded watershed on the boundary map smoothed by 0.09 D: seeds = basins at least --depth deep; regions
     smaller than --min-area (0.15 D^2) or without boundary contrast (unstructured areas) are dropped
Tracking: every pixel is placed in the common 3D frame with the chained full-3D rigid registration of
build_timelapse_viewer.py (cached in DATA/registration.json). For linking, frame k is mapped into the x-y scan
plane of the earlier frame with the relative transform (tilt between scans is small) and compared on that
frame's pixel grid; segmentation itself always runs on each frame's own 2D maps. Each pair is then fine-aligned
(shift + rotation, +-4 um / +-4 deg) on the combined cell-boundary maps; cells are matched by overlap
(IoU >= 0.2, Hungarian), also to tracks that ended up to --max-gap (3) frames earlier. Division candidate: an unmatched new cell and the matched successor both lie
>= 50 % inside the same previous cell and their areas add up to 0.6-1.5x the parent; "confirmed" if
daughter and mother both persist >= 3 frames.

Other data sets: give the pixel size (--px/--py or --pixels) and the cell diameter (--cell-diam); everything
else is in robust (dimensionless) units. Check segmentation_overview.png and adjust --depth if cells are split/merged.

Limits: AFM sees the outer surface only -> cells are surface patches (3D surface masks), not volumes.
With non-square pixels (e.g. 1.0 x 0.5 um) boundaries along the slow axis are detected more easily than
across it: cell shapes come out elongated along the fast scan axis (x). Areas / positions are usable,
orientation and aspect ratio are not. Square pixels avoid this.

Output
  DATA/cells/<name>.npy          int32 map (ny, nx): track ID (>=1), 0 = boundary / unsegmented, -1 = no data
  DATA/cells_hr/<name>.npy       int32 track IDs on the fine grid (scale.json: um per fine pixel, x and y)
  DATA/cell_boundary/<name>.npy  float32 combined boundary strength (robust units)
     -> build_timelapse_viewer.py picks both up (field "cells", outline overlay)
  OUT/cells_table.csv            one row per cell per frame (area, 3D centroid, shape, median of every map,
                                 junction vs centre stiffness, ...)
  OUT/tracks.csv                 one row per track (first/last frame, frames observed, longest reliable
                                 stretch, parent / division)
                                 cells_table.csv also flags every link: step_um, area_change, link_ok and
                                 reliable_segment (track split where area changes > 1.8x or the centre jumps > 3.5 um)
  OUT/cells_3d.npz               per frame: registered xyz of every pixel + track ID (+ transforms)
  OUT/ply/<name>.ply             3D surface mesh per frame (registered), vertex colour = track, property cell
  OUT/segmentation_overview.png  outlines over log E for every frame

    python segment_cells.py DATA -o OUT [--cell-diam 4] [--depth 0.1] [--contrast 0.2] [--channels all | E delta ...]

Needs build_timelapse_viewer.py in the same folder (frame loading + registration).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_timelapse_viewer as BV                       # noqa: E402

from skimage.filters import sato                          # noqa: E402
from skimage.segmentation import watershed                # noqa: E402
from skimage.morphology import h_minima                   # noqa: E402
from skimage.measure import regionprops                   # noqa: E402

# ----------------------------------------------------------------------------- channels
# (transform, boundary polarity: +1 junctions bright, -1 dark, 0 = decided against the anchor)
CHANNELS = {
    "E":                   ("log", +1),
    "delta":               ("lin", -1),
    "alpha":               ("lin", 0),
    "topo":                ("lin", -1),
    "adh_force_nN":        ("lin", +1),
    "adh_work_fJ":         ("lin", +1),
    "adh_pulloff_um":      ("lin", 0),
}
ANCHORS = ["E", "adh_force_nN", "adh_work_fJ", "delta"]
TOPO_MAX_W = 0.2
P_YSM = 0.6


def fill_nan(A):
    m = np.isfinite(A)
    if m.all() or not m.any():
        return np.nan_to_num(A)
    idx = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
    return A[tuple(idx)]


def despike(A, ok, k):
    """isolated-outlier filter. r = A - median of the 8 neighbours, s = 1.4826 MAD(r) over the map.
    A pixel is an outlier if it lies more than k*s above its 2nd-highest neighbour (or below its 2nd-lowest):
    it then differs from all but at most one neighbour, so it is a single-pixel spike, whereas a pixel on a thin
    line (cell junction) has >= 2 similar neighbours along the line and is kept. Outliers are replaced by the
    neighbour median. Returns (A, number replaced)."""
    if not k or k <= 0:
        return A, 0
    X = np.where(ok, A, np.nan)
    ny, nx = X.shape
    P = np.pad(X, 1, constant_values=np.nan)
    nb = np.stack([P[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], -1)
    cnt = np.isfinite(nb).sum(-1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN neighbourhoods
        med = np.nanmedian(nb, -1)
    r = X - med
    v = np.abs(r[np.isfinite(r)])
    if v.size < 20:
        return A, 0
    s_ = 1.4826 * np.median(v) + 1e-12
    srt = np.sort(nb, -1)                                   # NaN last
    hi2 = np.take_along_axis(srt, np.clip(cnt - 2, 0, 7)[..., None], -1)[..., 0]
    lo2 = srt[..., 1]
    with np.errstate(invalid="ignore"):
        bad = ok & (cnt >= 4) & (((X - hi2) > k * s_) | ((lo2 - X) > k * s_))
    return np.where(bad, med, A), int(bad.sum())


def destripe(A, valid):
    sm = ndi.median_filter(A, size=(7, 1))
    d = np.where(valid, A - sm, np.nan)
    off = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        r = d[i][np.isfinite(d[i])]
        if r.size:
            off[i] = np.median(r)
    return A - off[:, None]


def robust_z(A, m):
    v = A[m]
    if v.size == 0:
        return A * 0
    med = np.median(v)
    return (A - med) / (np.median(np.abs(v - med)) * 1.4826 + 1e-12)


def to_iso(A, px, py, h, order=1):
    return ndi.zoom(A, (py / h, px / h), order=order, mode="nearest")


def line_shifts(A, ok, px, max_um=1.5, smooth_lines=0):
    """lateral (x) offset of every scan line relative to its neighbours, from the cross-correlation of the
    high-passed line profiles (sub-pixel, um). Line jitter otherwise smears boundaries across lines."""
    X = fill_nan(np.where(ok, A, np.nan))
    X = X - ndi.gaussian_filter1d(X, 4, axis=1)
    ny, nx = X.shape
    up = 8
    xs = np.arange(0, nx - 1 + 1e-9, 1 / up)
    U = np.array([np.interp(xs, np.arange(nx), r) for r in X])
    ref = ndi.uniform_filter1d(U, 5, axis=0)                     # neighbourhood mean of lines
    m = int(round(max_um / px * up))
    sh = np.zeros(ny)
    for i in range(ny):
        r = ref[i] * 5 - U[i]                                    # neighbours only
        a = U[i]
        best, bs = -np.inf, 0
        for s_ in range(-m, m + 1):
            if s_ >= 0:
                c = np.dot(a[s_:], r[:len(r) - s_]) / max(len(r) - s_, 1)
            else:
                c = np.dot(a[:s_], r[-s_:]) / max(len(r) + s_, 1)
            if c > best:
                best, bs = c, s_
        sh[i] = bs / up * px
    sh -= np.median(sh)
    return sh


def apply_line_shifts(A, sh, px):
    ny, nx = A.shape
    out = np.full(A.shape, np.nan)
    x = np.arange(nx)
    for i in range(ny):
        r = A[i]
        m = np.isfinite(r)
        if m.sum() < 2:
            continue
        out[i] = np.interp(x + sh[i] / px, x[m], r[m], left=np.nan, right=np.nan)
    return out


def boundary_map(maps, valid, px, py, h=0.25, cell_um=4.0, ridge_um=(0.32, 0.72), weights=None, dejitter=False,
                 spike_k=4.0):
    if dejitter:     # estimate line jitter on the anchor channel and apply it to every channel
        an = next((a for a in ANCHORS if a in maps and np.isfinite(maps[a]).mean() > 0.3), None)
        if an is not None:
            A = maps[an].astype(float)
            if CHANNELS[an][0] == "log":
                with np.errstate(all="ignore"):
                    A = np.where(A > 0, np.log10(A), np.nan)
            sh = line_shifts(A, np.isfinite(A) & valid, px)
            maps = {k: apply_line_shifts(np.asarray(v, float), sh, px) for k, v in maps.items()}
            valid = apply_line_shifts(valid.astype(float), sh, px) > 0.5
    vi = to_iso(valid.astype(float), px, py, h, order=0) > 0.5
    sig = [r / h for r in ridge_um]
    R, OK, POL = {}, {}, {}
    for name, A in maps.items():
        if name not in CHANNELS:
            continue
        tf, pol = CHANNELS[name]
        X = A.astype(float)
        if tf == "log":
            with np.errstate(all="ignore"):
                X = np.where(X > 0, np.log10(X), np.nan)
        ok = np.isfinite(X) & valid
        if ok.mean() < 0.3:
            continue
        if name == "topo":
            med = ndi.median_filter(fill_nan(np.where(ok, X, np.nan)), 3)
            ok &= np.abs(X - med) < 3.0
        X, _ = despike(X, ok, spike_k)
        X = destripe(fill_nan(np.where(ok, X, np.nan)), ok)
        Xi = to_iso(X, px, py, h, order=3)                 # cubic resampling to the fine isotropic grid
        oki = (to_iso(ok.astype(float), px, py, h, order=1) > 0.99) & vi
        if not ok[valid].all():          # masked holes: do not use this channel within ~1 um of a hole edge
            oki &= ~ndi.binary_dilation(~oki & vi, iterations=max(1, int(round(1.0 / h))))
        Xi = ndi.gaussian_filter(Xi, (max(0.8, P_YSM * px / h), 0.3))   # match y resolution to x sampling
        Xi = Xi - ndi.gaussian_filter(Xi, cell_um / h)
        Xi = robust_z(Xi, oki)
        if pol == 0:
            R[name] = (sato(Xi, sigmas=sig, black_ridges=False, mode="nearest"),
                       sato(Xi, sigmas=sig, black_ridges=True, mode="nearest"))
        else:
            R[name] = (sato(Xi, sigmas=sig, black_ridges=pol < 0, mode="nearest"),)
            POL[name] = pol
        OK[name] = oki
    if not R:
        raise ValueError("no usable channel")
    anchor = next((a for a in ANCHORS if a in R), None)
    ch = {n: robust_z(r[0], OK[n]) for n, r in R.items() if len(r) == 1}
    for n, r in R.items():
        if len(r) == 2:
            p_, q_ = robust_z(r[0], OK[n]), robust_z(r[1], OK[n])
            ref = ch.get(anchor) if anchor and anchor != n else None
            if ref is None:
                ch[n], POL[n] = p_, 1
                continue
            m = OK[n] & OK[anchor]
            cp, cq = np.corrcoef(p_[m], ref[m])[0, 1], np.corrcoef(q_[m], ref[m])[0, 1]
            ch[n], POL[n] = (p_, 1) if cp >= cq else (q_, -1)
    W = {}
    for n in ch:
        if weights and n in weights:
            W[n] = float(weights[n]); continue
        others = [m for m in ch if m != n]
        if not others:
            W[n] = 1.0; continue
        num = sum(np.where(OK[m], ch[m], 0) for m in others)
        den = sum(OK[m].astype(float) for m in others)
        cons = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        mm = OK[n] & np.isfinite(cons)
        W[n] = max(0.0, np.corrcoef(ch[n][mm], cons[mm])[0, 1]) if mm.sum() > 50 else 0.0
    tot = sum(W.values()) or 1.0
    W = {n: w / tot for n, w in W.items()}
    if "topo" in W and W["topo"] > TOPO_MAX_W and len(W) > 1 and not (weights and "topo" in weights):
        s_ = sum(w for n, w in W.items() if n != "topo") or 1.0
        W = {n: (TOPO_MAX_W if n == "topo" else w / s_ * (1 - TOPO_MAX_W)) for n, w in W.items()}
    num = sum(W[n] * np.where(OK[n], ch[n], 0) for n in ch)
    den = sum(W[n] * OK[n] for n in ch)
    B = np.where((den > 0.25) & vi, num / np.maximum(den, 1e-9), np.nan)
    return B, vi & np.isfinite(B), {n: round(w, 3) for n, w in W.items()}, POL


def segment(B, vi, h=0.25, depth=0.1, min_area_um2=2.4, contrast_min=0.2, smooth_um=0.36):
    Bs = ndi.gaussian_filter(fill_nan(B), smooth_um / h)
    seeds = ndi.label(h_minima(Bs, depth) & vi)[0]
    lab = watershed(Bs, seeds, mask=vi, watershed_line=True)
    out = np.zeros_like(lab)
    k = 0
    for rp in regionprops(lab):
        sl = rp.slice
        pad = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
        sub = lab[pad] == rp.label
        if sub.sum() * h * h < min_area_um2:
            continue
        ring = ndi.binary_dilation(sub, iterations=2) & ~sub & vi[pad]
        inner = np.median(Bs[pad][sub])
        edge = np.median(Bs[pad][ring]) if ring.any() else inner
        if edge - inner < contrast_min:
            continue
        k += 1
        out[pad][sub] = k
    return out, Bs


# ----------------------------------------------------------------------------- per frame (worker)
def seg_frame(job):
    name, maps, valid, px, py, P = job
    h = P["h"]
    B, vi, W, POL = boundary_map(maps, valid, px, py, h, P["cell_um"], tuple(P["ridge_um"]), P.get("weights"),
                                 P.get("dejitter", False), P.get("despike", 4.0))
    lab_i, Bs = segment(B, vi, h, P["depth"], P["min_area"], P["contrast"], P["smooth_um"])
    ny, nx = valid.shape
    rr, cc = np.mgrid[0:ny, 0:nx]
    ri = np.clip(np.round(rr * py / h).astype(int), 0, lab_i.shape[0] - 1)
    ci = np.clip(np.round(cc * px / h).astype(int), 0, lab_i.shape[1] - 1)
    lab = lab_i[ri, ci].astype(np.int32)
    lab[~valid] = -1
    Bo = ndi.map_coordinates(np.nan_to_num(Bs), [rr * py / h, cc * px / h], order=1, mode="nearest")
    Bo = np.where(valid, Bo, np.nan).astype(np.float32)
    shape = {}
    for rp in regionprops(lab_i):
        shape[rp.label] = dict(major_um=rp.axis_major_length * h, minor_um=rp.axis_minor_length * h,
                               orientation_deg=float(np.degrees(rp.orientation)), solidity=float(rp.solidity),
                               perimeter_um=float(rp.perimeter * h))
    lab_hr = np.where(vi, lab_i, -1).astype(np.int32)     # fine-grid labels (same local IDs)
    ny_h, nx_h = lab_i.shape
    scale = ((nx - 1) * px / max(nx_h - 1, 1), (ny - 1) * py / max(ny_h - 1, 1))   # um per fine pixel (x, y)
    return name, lab, Bo, W, POL, shape, lab_hr, scale


# ----------------------------------------------------------------------------- tracking
def registered_xyz(f, G):
    T = f["v"]["topo"]
    ny, nx = T.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    P = np.stack([xx * f["px"], yy * f["py"], np.nan_to_num(T), np.ones_like(T)], -1).reshape(-1, 4)
    return (P @ G.T)[:, :3].reshape(ny, nx, 3)


def overlap_counts(la, xa, lb, xb, bin_um):
    """pixel-bin overlap between labels of two frames in registered x, y"""
    def bins(l, x):
        m = l > 0
        b = np.floor(x[m][:, :2] / bin_um).astype(np.int64)
        key = b[:, 0] * 1_000_003 + b[:, 1]
        u, i = np.unique(np.c_[key, l[m]], axis=0, return_index=True)
        return u
    A = bins(la, xa); Bb = bins(lb, xb)
    ca = {k: v for k, v in zip(*np.unique(A[:, 1], return_counts=True))}
    cb = {k: v for k, v in zip(*np.unique(Bb[:, 1], return_counts=True))}
    # join on bin key
    da = {}
    for k, l in A:
        da.setdefault(k, []).append(l)
    inter = {}
    for k, l in Bb:
        for la_ in da.get(k, ()):
            inter[(la_, l)] = inter.get((la_, l), 0) + 1
    return inter, ca, cb


def _grid(xy, val, h):
    """scattered registered points -> regular grid (linear interpolation, NaN outside)"""
    from scipy.interpolate import griddata
    lo = xy.min(0) - 1
    hi = xy.max(0) + 1
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], h), np.arange(lo[1], hi[1], h))
    G = griddata(xy, val, (gx, gy), method="linear")
    return G, lo


def refine_pair(job):
    """local rigid 2D alignment (shift + rotation) of frame B onto frame A using the cell-boundary maps,
    in registered coordinates. Returns (theta, dx, dy, corr_before, corr_after, centre)."""
    from scipy.optimize import minimize
    xyA, bA, xyB, bB, h, smax, rmax = job
    mA = np.isfinite(bA); mB = np.isfinite(bB)
    G, lo = _grid(xyA[mA], bA[mA], h)
    Gf = np.nan_to_num(G)
    Gm = np.isfinite(G).astype(float)
    P = xyB[mB]; v = bB[mB]
    c0 = P.mean(0)
    def corr(par):
        th, dx, dy = par
        c, s_ = np.cos(th), np.sin(th)
        Q = (P - c0) @ np.array([[c, s_], [-s_, c]]) + c0 + [dx, dy]
        cc = [(Q[:, 1] - lo[1]) / h, (Q[:, 0] - lo[0]) / h]
        a = ndi.map_coordinates(Gf, cc, order=1, cval=0)
        m = ndi.map_coordinates(Gm, cc, order=1, cval=0) > 0.99
        if m.sum() < 0.3 * len(v):
            return 0.0
        return np.corrcoef(a[m], v[m])[0, 1]
    c_before = corr([0, 0, 0])
    best = (c_before, [0.0, 0.0, 0.0])
    for th in np.radians(np.arange(-rmax, rmax + 0.1, 1.0)):
        for dx in np.arange(-smax, smax + 0.01, 0.5):
            for dy in np.arange(-smax, smax + 0.01, 0.5):
                c = corr([th, dx, dy])
                if c > best[0]:
                    best = (c, [th, dx, dy])
    r = minimize(lambda p: -corr(p), best[1], method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-4, initial_simplex=np.array(best[1]) + np.array(
                     [[0, 0, 0], [0.01, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])))
    x = r.x if -r.fun >= best[0] else best[1]
    return float(x[0]), float(x[1]), float(x[2]), float(c_before), float(max(-r.fun, best[0])), c0


def apply_refine(xy, rf):
    th, dx, dy, _, _, c0 = rf
    c, s_ = np.cos(th), np.sin(th)
    return (xy - c0) @ np.array([[c, s_], [-s_, c]]) + c0 + [dx, dy]


def overlap_counts(la, xa, lb, xb, bin_um):
    """overlap of labels of two frames on a common grid of bin_um (registered x, y)"""
    bins = np.broadcast_to(np.asarray(bin_um, float), (2,))
    def cells(l, x):
        m = l > 0
        b = np.floor(x[m] / bins + 0.5).astype(np.int64)
        key = b[:, 0] * 1_000_003 + b[:, 1]
        return np.unique(np.c_[key, l[m]], axis=0)
    A = cells(la, xa); Bb = cells(lb, xb)
    ca = dict(zip(*np.unique(A[:, 1], return_counts=True)))
    cb = dict(zip(*np.unique(Bb[:, 1], return_counts=True)))
    da = {}
    for k, l in A:
        da.setdefault(k, []).append(l)
    inter = {}
    for k, l in Bb:
        for la_ in da.get(k, ()):
            inter[(la_, l)] = inter.get((la_, l), 0) + 1
    return inter, ca, cb


def _match(la, xa, lb, xb, iou_min, bin_um):
    inter, ca, cb = overlap_counts(la, xa, lb, xb, bin_um)
    A = sorted(ca); Bl = sorted(cb)
    if not A or not Bl:
        return {}, inter, ca, cb, []
    ia = {a: i for i, a in enumerate(A)}; ib = {b: j for j, b in enumerate(Bl)}
    M = np.zeros((len(A), len(Bl)))
    for (a, b), n in inter.items():
        M[ia[a], ib[b]] = n / (ca[a] + cb[b] - n)
    ri, ci = linear_sum_assignment(-M)
    pairs = {int(A[i]): int(Bl[j]) for i, j in zip(ri, ci) if M[i, j] >= iou_min}
    return pairs, inter, ca, cb, [M[i, j] for i, j in zip(ri, ci) if M[i, j] >= iou_min]


def track(labels, pair_xy, bin_of, iou_min=0.2, max_gap=3):
    """labels: frame-local label maps. pair_xy(a, k) -> (xy_a, xy_k): pixel coordinates of frame a and of frame k
    expressed in the same plane (by default frame a's own scan plane). bin_of(a): comparison grid (um).
    Cells are linked to tracks that ended in one of the previous max_gap+1 frames (nearest frame first),
    by IoU (Hungarian). Returns per-frame {local: track}, tracks {id: first, last, parent, division, frames},
    mean IoU of direct (consecutive-frame) links, number of gap links."""
    K = len(labels)
    tracks, maps, ious = {}, [], []
    nxt, n_gap = 1, 0
    first = {}
    for l in np.unique(labels[0][labels[0] > 0]):
        first[int(l)] = nxt
        tracks[nxt] = dict(first=0, last=0, parent=0, division="", frames=[0])
        nxt += 1
    maps.append(first)
    for k in range(1, K):
        cur = {}
        for g in range(1, max_gap + 2):
            a = k - g
            if a < 0:
                break
            ends = {l: t for l, t in maps[a].items() if tracks[t]["last"] == a}
            if not ends:
                continue
            la = np.where(np.isin(labels[a], list(ends)), labels[a], 0)
            lb = np.where((labels[k] > 0) & ~np.isin(labels[k], list(cur)), labels[k], 0)
            if not (lb > 0).any():
                break
            xa, xk = pair_xy(a, k)
            pairs, inter, ca, cb, st = _match(la, xa, lb, xk, iou_min, bin_of(a))
            for aa, bb in pairs.items():
                t = ends[aa]
                cur[bb] = t
                tracks[t]["last"] = k
                tracks[t]["frames"].append(k)
            if g == 1:
                ious.append(float(np.mean(st)) if st else np.nan)
            else:
                n_gap += len(pairs)
        # new tracks; division candidates from the overlap with frame k-1
        xa, xk = pair_xy(k - 1, k)
        _, inter, ca, cb, _ = _match(labels[k - 1], xa, labels[k], xk, 1.1, bin_of(k - 1))
        succ = {}
        for l_, t in maps[k - 1].items():
            for bb, tt in cur.items():
                if tt == t:
                    succ[l_] = bb
        for b in np.unique(labels[k][labels[k] > 0]):
            b = int(b)
            if b in cur:
                continue
            tracks[nxt] = dict(first=k, last=k, parent=0, division="", frames=[k])
            for a_, b1 in succ.items():
                if b not in cb or b1 not in cb or a_ not in ca:
                    continue
                f2 = inter.get((a_, b), 0) / cb[b]
                f1 = inter.get((a_, b1), 0) / cb[b1]
                tot = (cb[b] + cb[b1]) / ca[a_]
                if f2 >= 0.5 and f1 >= 0.5 and 0.6 <= tot <= 1.5 and 0.2 <= cb[b] / ca[a_] <= 0.8:
                    tracks[nxt]["parent"] = maps[k - 1][a_]
                    break
            cur[b] = nxt
            nxt += 1
        maps.append(cur)
    for t, d in tracks.items():
        if d["parent"]:
            p = tracks.get(d["parent"])
            ok = p is not None and len(d["frames"]) >= 3 and p["last"] - d["first"] >= 2
            d["division"] = "confirmed" if ok else "candidate"
    return maps, tracks, ious, n_gap


# ----------------------------------------------------------------------------- outputs
def palette(t):
    t = np.asarray(t, np.int64)
    h = ((t * 2654435761) % 4294967296) / 4294967296.0
    import colorsys
    return np.array([colorsys.hsv_to_rgb(x, 0.55 + 0.35 * ((i * 7) % 3) / 2, 0.95) for i, x in enumerate(h)])


def write_ply(path, xyz, T, trk, extra):
    ny, nx = T.shape
    ok = np.isfinite(T)
    idx = -np.ones((ny, nx), int)
    idx[ok] = np.arange(ok.sum())
    faces = []
    for r in range(ny - 1):
        for q in range(nx - 1):
            a, b, c, d = idx[r, q], idx[r, q + 1], idx[r + 1, q], idx[r + 1, q + 1]
            if a >= 0 and b >= 0 and d >= 0:
                faces.append((a, b, d))
            if a >= 0 and d >= 0 and c >= 0:
                faces.append((a, d, c))
    v = xyz[ok]
    t = trk[ok]
    col = np.full((len(v), 3), 140, np.uint8)
    m = t > 0
    if m.any():
        col[m] = (palette(t[m]) * 255).astype(np.uint8)
    col[t == 0] = 40
    names = list(extra)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\ncomment AFM surface, registered um; cell = track ID (0 boundary, -1 no data)\n")
        fh.write(f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int cell\n")
        for n in names:
            fh.write(f"property float {n}\n")
        fh.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        ex = [np.asarray(extra[n], float)[ok] for n in names]
        for i in range(len(v)):
            fh.write(f"{v[i,0]:.3f} {v[i,1]:.3f} {v[i,2]:.3f} {col[i,0]} {col[i,1]} {col[i,2]} {int(t[i])}"
                     + "".join(f" {e[i]:.5g}" for e in ex) + "\n")
        for a, b, c in faces:
            fh.write(f"3 {a} {b} {c}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="folder from extract_topo_E.py")
    ap.add_argument("-o", "--out", default=None, help="results folder (default DATA/segmentation)")
    ap.add_argument("--cell-diam", type=float, default=4.0,
                    help="typical cell diameter [um]; sets the defaults of all length scales below (default 4)")
    ap.add_argument("--channels", nargs="*", default=None,
                    help=f"subset of {list(CHANNELS)}, or 'all'. Default: adh_force_nN adh_work_fJ if present "
                         "(sharpest cell junctions in this data), otherwise all present")
    ap.add_argument("--weights", default=None, help='JSON {"E": 0.4, ...} fixed channel weights')
    ap.add_argument("--depth", type=float, default=0.1, help="seed depth (robust units); lower = more cells")
    ap.add_argument("--contrast", type=float, default=0.2, help="min boundary contrast of a cell (robust units)")
    ap.add_argument("--min-area", type=float, default=None, help="min cell area [um^2] (default 0.15 D^2)")
    ap.add_argument("--h", type=float, default=None,
                    help="fine isotropic grid spacing [um] (default min(D/16, px/2, py/2))")
    ap.add_argument("--cell-um", type=float, default=None, help="detrending scale [um] (default D)")
    ap.add_argument("--ridge-um", type=float, nargs=2, default=None, help="ridge filter scales [um] (default 0.08 D, 0.18 D)")
    ap.add_argument("--smooth-um", type=float, default=None, help="smoothing of the boundary map [um] (default 0.09 D)")
    ap.add_argument("--despike", type=float, default=4.0,
                    help="isolated-outlier filter: replace pixels > K robust SD beyond their 2nd most extreme neighbour (0 = off)")
    ap.add_argument("--dejitter", action="store_true",
                    help="re-align scan lines laterally before filtering (estimated on E, applied to all maps)")
    ap.add_argument("--drop-excluded", action="store_true", help="treat pixels flagged in excluded/ as no data")
    ap.add_argument("--iou", type=float, default=0.2, help="min IoU to link a cell to the next frame")
    ap.add_argument("--no-refine", action="store_true", help="skip the pairwise fine alignment for tracking")
    ap.add_argument("--track-plane", choices=["local", "registered"], default="local",
                    help="compare cells in the earlier frame's scan plane (default) or projected onto frame 0")
    ap.add_argument("--max-gap", type=int, default=3, help="max number of missing frames bridged in a track")
    ap.add_argument("--max-area-change", type=float, default=1.8,
                    help="plausibility: max area ratio between consecutive observations of a track")
    ap.add_argument("--max-step", type=float, default=None,
                    help="plausibility: max centroid step [um] (3D distance) between observations (default 0.75 D)")
    ap.add_argument("--refine-shift", type=float, default=None, help="fine-alignment search range [um] (default 0.75 D)")
    ap.add_argument("--refine-rot", type=float, default=4.0, help="fine-alignment rotation range [deg]")
    ap.add_argument("--px", type=float, default=None, help="pixel size x [um]; default from DATA/pixels.json")
    ap.add_argument("--py", type=float, default=None, help="pixel size y [um]; default from DATA/pixels.json")
    ap.add_argument("--pixels", default=None, help='JSON {"name": [px, py]}; default DATA/pixels.json')
    ap.add_argument("--rot-max", type=float, default=24.0, help="registration: rotation search range [deg]")
    ap.add_argument("--shift-max", type=float, default=None, help="registration: shift search range [um] (auto)")
    ap.add_argument("--config", default=None, help="parameter file (TOML); command line wins")
    ap.add_argument("--spike", type=float, default=5.0)
    ap.add_argument("--cores", type=float, default=0.8)
    ap.add_argument("--recompute-registration", action="store_true")
    ap.add_argument("--no-ply", action="store_true")
    chan = lambda v: ["all"] if v == "all" else (list(v) if isinstance(v, (list, tuple)) else [v])
    BV.apply_config(ap, sys.argv[1:], {
        ("segmentation", "cell_diameter_um"): "cell_diam", ("segmentation", "channels"): ("channels", chan),
        ("segmentation", "grid_um"): "h", ("segmentation", "detrend_um"): "cell_um",
        ("segmentation", "ridge_um"): "ridge_um", ("segmentation", "smooth_um"): "smooth_um",
        ("segmentation", "min_area_um2"): "min_area", ("segmentation", "seed_depth"): "depth",
        ("segmentation", "contrast"): "contrast", ("segmentation", "despike_k"): "despike",
        ("segmentation", "drop_excluded"): "drop_excluded",
        ("tracking", "iou_min"): "iou", ("tracking", "max_gap_frames"): "max_gap", ("tracking", "max_step_um"): "max_step",
        ("tracking", "max_area_change"): "max_area_change", ("tracking", "refine_shift_um"): "refine_shift",
        ("tracking", "refine_rot_deg"): "refine_rot", ("tracking", "track_plane"): "track_plane",
        ("registration", "spike_um"): "spike", ("registration", "recompute"): "recompute_registration",
        ("registration", "rot_max_deg"): "rot_max", ("registration", "shift_max_um"): "shift_max",
        ("run", "cores"): "cores"})
    args = ap.parse_args()

    D = args.cell_diam
    args.cell_um = args.cell_um if args.cell_um is not None else D
    args.ridge_um = args.ridge_um if args.ridge_um is not None else [round(0.08 * D, 3), round(0.18 * D, 3)]
    args.min_area = args.min_area if args.min_area is not None else round(0.15 * D * D, 3)
    args.smooth_um = args.smooth_um if args.smooth_um is not None else round(0.09 * D, 3)
    args.max_step = args.max_step if args.max_step is not None else round(0.75 * D, 3)
    args.refine_shift = args.refine_shift if args.refine_shift is not None else round(0.75 * D, 3)
    data = Path(args.data).expanduser()
    out = Path(args.out).expanduser() if args.out else data / "segmentation"
    out.mkdir(parents=True, exist_ok=True)
    args.px, args.py, pixels = BV.resolve_pixels(data, args.px, args.py, args.pixels)
    frames = BV.load_frames(data, args.px, args.py, pixels)
    if not frames:
        sys.exit("no frames")
    for f in frames:
        f["v"]["topo"] = BV.clean_topo(f, args.spike)
    nw = BV.n_workers(args.cores)
    print(f"{len(frames)} frames; registering (cached in {data/'registration.json'}) ...")
    if args.shift_max is None:
        args.shift_max = BV.auto_shift_max(frames)
    pairs = BV.get_registration(frames, data, nw, args.rot_max, args.shift_max,
                                recompute=args.recompute_registration, verbose=False)
    _, G3s = BV.chain(pairs)

    if args.h is None:
        args.h = round(min(D / 16, min(min(f["px"], f["py"]) for f in frames) / 2), 4)
    print(f"cell diameter {D} um -> grid {args.h} um, detrend {args.cell_um} um, ridges {args.ridge_um} um, "
          f"smoothing {args.smooth_um} um, min area {args.min_area} um^2, max step {args.max_step} um, "
          f"fine-alignment range +-{args.refine_shift} um")
    P = dict(h=args.h, smooth_um=args.smooth_um, despike=args.despike, cell_um=args.cell_um, ridge_um=args.ridge_um, depth=args.depth, min_area=args.min_area,
             dejitter=args.dejitter,
             contrast=args.contrast, weights=json.loads(args.weights) if args.weights else None)
    present = [k for k in CHANNELS if all(k in f["v"] and np.isfinite(f["v"][k]).any() for f in frames)]
    if args.channels and args.channels != ["all"]:
        use = [k for k in args.channels if k in present]
    elif args.channels == ["all"]:
        use = present
    else:
        use = [k for k in ("adh_force_nN", "adh_work_fJ") if k in present] or present
    jobs = []
    for f in frames:
        v = f["v"]
        chans = {k: v[k] for k in v if k in CHANNELS and k in use}
        valid = np.isfinite(v["topo"]) & np.any([np.isfinite(a) for k, a in chans.items() if k != "topo"], axis=0)
        x = v.get("_excl")
        if x is not None:
            valid &= (x & 16) == 0
            if args.drop_excluded:
                valid &= x == 0
                chans = {k: (np.where(x == 0, a, np.nan) if k != "topo" else a) for k, a in chans.items()}
        jobs.append((f["name"], chans, valid, f["px"], f["py"], P))
    print(f"segmenting with channels: {sorted(jobs[0][1])}")
    res = {}
    with ProcessPoolExecutor(max_workers=min(nw, len(jobs))) as ex:
        for name, lab, Bo, W, POL, shape, lab_hr, sc in ex.map(seg_frame, jobs):
            res[name] = (lab, Bo, W, POL, shape, lab_hr, sc)
            print(f"  {name[-6:]}: {lab.max()} cells   weights " + " ".join(f"{k}={w:.2f}" for k, w in W.items()))

    labels = [res[f["name"]][0] for f in frames]
    xyz = [registered_xyz(f, G3s[k]) for k, f in enumerate(frames)]
    # Tracking geometry: frame k is mapped into the scan plane of the earlier frame a with the relative rigid
    # transform inv(G_a) G_k (small tilt), so cells are compared in a's own x-y map, not in a projection onto
    # the plane of frame 0 (which foreshortens tilted / steep surfaces). Then a local fine alignment on the
    # boundary maps. --track-plane registered uses the projection onto frame 0 (old behaviour).
    native = []
    for f in frames:
        ny, nx = f["v"]["topo"].shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        native.append(np.stack([xx * f["px"], yy * f["py"], np.nan_to_num(f["v"]["topo"])], -1))
    def mapped(a, k):
        if args.track_plane == "registered":
            return xyz[a][..., :2], xyz[k][..., :2]
        Grel = np.linalg.inv(G3s[a]) @ G3s[k]
        P = native[k].reshape(-1, 3) @ Grel[:3, :3].T + Grel[:3, 3]
        return native[a][..., :2], P[:, :2].reshape(native[k].shape[:2] + (2,))
    pair_list = [(k - g, k) for k in range(1, len(frames)) for g in range(1, args.max_gap + 2) if k - g >= 0]
    base = {pk: mapped(*pk) for pk in pair_list}
    refs = {}
    if not args.no_refine and pair_list:
        rjobs = [(base[(a, k)][0].reshape(-1, 2), res[frames[a]["name"]][1].ravel(),
                  base[(a, k)][1].reshape(-1, 2), res[frames[k]["name"]][1].ravel(),
                  0.5, args.refine_shift, args.refine_rot) for a, k in pair_list]
        with ProcessPoolExecutor(max_workers=min(nw, len(rjobs))) as ex:
            refs = dict(zip(pair_list, ex.map(refine_pair, rjobs)))
        cons = [refs[(k - 1, k)] for k in range(1, len(frames))]
        print(f"tracking in the {'scan plane of the earlier frame' if args.track_plane == 'local' else 'projection onto frame 0'}; "
              f"fine alignment on boundary maps: correlation {np.median([r[3] for r in cons]):.2f} -> "
              f"{np.median([r[4] for r in cons]):.2f}, shift {np.median([np.hypot(r[1], r[2]) for r in cons]):.2f} um, "
              f"rotation {np.median([abs(np.degrees(r[0])) for r in cons]):.1f} deg (medians, consecutive frames)")
    def pair_xy(a, k):
        xa, xk = base[(a, k)]
        if (a, k) in refs:
            xk = apply_refine(xk.reshape(-1, 2), refs[(a, k)]).reshape(xk.shape)
        return xa, xk
    if args.track_plane == "registered":
        bin_of = lambda a: 1.0
    else:
        bin_of = lambda a: (frames[a]["px"], frames[a]["py"])
    maps, tracks, ious, n_gap = track(labels, pair_xy, bin_of, args.iou, max_gap=args.max_gap)
    lens = np.array([len(d["frames"]) for d in tracks.values()])
    print(f"tracking: {len(tracks)} tracks, mean IoU of linked cells {np.nanmean(ious):.2f}, "
          f"observed in >= 5 / 10 / 20 / 30 frames: {np.sum(lens >= 5)} / {np.sum(lens >= 10)} / "
          f"{np.sum(lens >= 20)} / {np.sum(lens >= 30)} tracks (longest {lens.max()}), "
          f"{n_gap} links across gaps, divisions: "
          f"{sum(1 for t in tracks.values() if t['division'] == 'confirmed')} confirmed "
          f"(daughter + mother >= 3 frames), {sum(1 for t in tracks.values() if t['division'] == 'candidate')} candidates")

    (data / "cells").mkdir(exist_ok=True)
    (data / "cell_boundary").mkdir(exist_ok=True)
    (data / "cells_hr").mkdir(exist_ok=True)
    hr_scale = {}
    if not args.no_ply:
        (out / "ply").mkdir(exist_ok=True)
    rows, npz = [], {}
    t0 = frames[0]["sec"]
    chan_keys = [k for k in CHANNELS if k in frames[0]["v"]]
    for k, f in enumerate(frames):
        lab, Bo, W, POL, shape, lab_hr, sc = res[f["name"]]
        trk = np.where(lab > 0, 0, lab).astype(np.int32)
        trk_hr = np.where(lab_hr > 0, 0, lab_hr).astype(np.int32)
        for l, t in maps[k].items():
            trk[lab == l] = t
            trk_hr[lab_hr == l] = t
        np.save(data / "cells" / f"{f['name']}.npy", trk)
        np.save(data / "cells_hr" / f"{f['name']}.npy", trk_hr)
        hr_scale[f["name"]] = [round(sc[0], 6), round(sc[1], 6)]
        np.save(data / "cell_boundary" / f"{f['name']}.npy", Bo)
        npz[f"{f['name']}__xyz"] = xyz[k].astype(np.float32)
        npz[f"{f['name']}__track"] = trk
        npz[f"{f['name']}__G"] = G3s[k]
        # per-cell measurements
        v = f["v"]
        T = v["topo"]
        gy, gx = np.gradient(BV.np.nan_to_num(T), f["py"], f["px"])
        dA = f["px"] * f["py"] * np.sqrt(1 + gx ** 2 + gy ** 2)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        E = v["E"]
        for l, t in maps[k].items():
            m = lab == l
            ring = ndi.binary_dilation(m, iterations=1) & ~m & (lab >= 0)
            c = xyz[k][m].mean(0)
            yy, xx = np.nonzero(m)
            row = dict(frame=k, name=f["name"], time=f["time"], t_min=round((f["sec"] - t0) / 60, 2),
                       track=t, parent=tracks[t]["parent"], division=tracks[t]["division"], n_px=int(m.sum()),
                       area_proj_um2=round(float(m.sum() * f["px"] * f["py"]), 3),
                       area_surf_um2=round(float(dA[m].sum()), 3),
                       eq_diam_um=round(float(np.sqrt(4 * dA[m].sum() / np.pi)), 3),
                       x_um=round(float(c[0]), 3), y_um=round(float(c[1]), 3), z_um=round(float(c[2]), 3),
                       x_raw_um=round(float(xx.mean() * f["px"]), 3), y_raw_um=round(float(yy.mean() * f["py"]), 3),
                       slope_deg=round(float(np.nanmedian(slope[m])), 2),
                       boundary_contrast=round(float(np.nanmedian(Bo[ring]) - np.nanmedian(Bo[m])), 3) if ring.any() else "",
                       E_junction_Pa=round(float(np.nanmedian(E[ring])), 4) if np.isfinite(E[ring]).any() else "",
                       E_centre_Pa=round(float(np.nanmedian(E[m])), 4) if np.isfinite(E[m]).any() else "")
            if row["E_junction_Pa"] != "" and row["E_centre_Pa"] not in ("", 0):
                row["E_junction_over_centre"] = round(row["E_junction_Pa"] / row["E_centre_Pa"], 3)
            else:
                row["E_junction_over_centre"] = ""
            row.update({kk: round(vv, 3) for kk, vv in shape.get(l, {}).items()})
            for ck in chan_keys:
                a = v[ck][m]
                row[f"{ck}_median"] = round(float(np.nanmedian(a)), 5) if np.isfinite(a).any() else ""
            rows.append(row)
        if not args.no_ply:
            extra = {ck: v[ck] for ck in ("E", "alpha", "delta", "adh_force_nN") if ck in v}
            write_ply(out / "ply" / f"{f['name']}.ply", xyz[k], T, trk, extra)
    # link plausibility along every track: area change and centroid step (registered) between observations
    byt = {}
    for r in rows:
        byt.setdefault(r["track"], []).append(r)
    best_rel = {}
    for t, rr in byt.items():
        rr.sort(key=lambda r: r["frame"])
        seg, run, best = 1, 1, 1
        for i, r in enumerate(rr):
            if i == 0:
                r.update(step_um="", area_change="", link_ok="", reliable_segment=f"{t}.1")
                continue
            a = rr[i - 1]
            st = float(np.sqrt((r["x_um"] - a["x_um"]) ** 2 + (r["y_um"] - a["y_um"]) ** 2 + (r["z_um"] - a["z_um"]) ** 2))
            ar = max(r["area_surf_um2"], a["area_surf_um2"]) / max(min(r["area_surf_um2"], a["area_surf_um2"]), 1e-6)
            ok = ar <= args.max_area_change and st <= args.max_step
            if ok:
                run += 1
            else:
                seg += 1; run = 1
            best = max(best, run)
            r.update(step_um=round(st, 3), area_change=round(ar, 3), link_ok=int(ok), reliable_segment=f"{t}.{seg}")
        best_rel[t] = best
    (data / "cells_hr" / "scale.json").write_text(json.dumps(hr_scale))   # um per fine pixel (x, y), per frame
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "cells_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "tracks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "first_frame", "last_frame", "n_observed", "longest_reliable", "span_frames",
                    "first_time", "last_time", "parent", "division", "frames"])
        for t, d in sorted(tracks.items()):
            w.writerow([t, d["first"], d["last"], len(d["frames"]), best_rel.get(t, 1), d["last"] - d["first"] + 1,
                        frames[d["first"]]["time"], frames[d["last"]]["time"], d["parent"], d["division"],
                        " ".join(map(str, d["frames"]))])
    np.savez_compressed(out / "cells_3d.npz", names=np.array([f["name"] for f in frames]),
                        times=np.array([f["time"] for f in frames]), **npz)
    overview(frames, res, maps, out / "segmentation_overview.png")
    n_long = sum(1 for d in tracks.values() if len(d["frames"]) >= 5)
    rel = np.array(list(best_rel.values()))
    print(f"reliable stretches (area change <= {args.max_area_change}x and centroid step <= {args.max_step} um "
          f"between observations): >= 5 / 10 / 15 / 20 frames: {np.sum(rel >= 5)} / {np.sum(rel >= 10)} / "
          f"{np.sum(rel >= 15)} / {np.sum(rel >= 20)} (longest {rel.max() if rel.size else 0})")
    print(f"wrote {data/'cells'}, {data/'cell_boundary'}, {out/'cells_table.csv'} ({len(rows)} rows), "
          f"{out/'tracks.csv'} ({len(tracks)} tracks, {n_long} followed >= 5 frames), cells_3d.npz"
          + ("" if args.no_ply else ", ply/") + ", segmentation_overview.png")


def overview(frames, res, maps, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(frames)
    nc = min(7, n)
    nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.5 * nr), squeeze=False)
    for k, f in enumerate(frames):
        a = ax.ravel()[k]
        lab_hr, (sx, sy) = res[f["name"]][5], res[f["name"]][6]
        key = "adh_force_nN" if "adh_force_nN" in f["v"] else "E"
        A = f["v"][key].astype(float)
        if key == "E":
            with np.errstate(all="ignore"):
                A = np.log10(np.where(A > 0, A, np.nan))
        ok = np.isfinite(A)
        if not ok.any():
            a.axis("off"); continue
        X = ndi.zoom(fill_nan(A), (lab_hr.shape[0] / A.shape[0], lab_hr.shape[1] / A.shape[1]), order=3)
        X = X - ndi.gaussian_filter(X, 4.0 / sx)
        X[lab_hr < 0] = np.nan
        ext = [0, lab_hr.shape[1] * sx, 0, lab_hr.shape[0] * sy]
        a.imshow(X, origin="lower", extent=ext, cmap="gray", aspect="equal",
                 vmin=np.nanpercentile(X, 2), vmax=np.nanpercentile(X, 98))
        trk = np.where(lab_hr > 0, 0, lab_hr)
        for l, t in maps[k].items():
            trk[lab_hr == l] = t
        rgba = np.zeros(lab_hr.shape + (4,))
        m = trk > 0
        if m.any():
            rgba[m, :3] = palette(trk[m])
            rgba[m, 3] = 0.25
        edge = (trk >= 0) & ((np.roll(trk, 1, 0) != trk) | (np.roll(trk, 1, 1) != trk) | (trk == 0))
        rgba[edge] = [0.9, 0.1, 0.1, 0.95]
        a.imshow(rgba, origin="lower", extent=ext, aspect="equal", interpolation="nearest")
        a.set_title(f"{f['time']}  {len(maps[k])} cells", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    for a in ax.ravel()[n:]:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
