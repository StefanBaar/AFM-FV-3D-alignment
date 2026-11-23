"""Re-runs the per-frame segmentation of segment_cells.py step by step and keeps every intermediate array.
The code below is a copy of boundary_map()/segment() with recording added (same parameters)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/seg2')
from scipy import ndimage as ndi
import segment_cells as S, build_timelapse_viewer as BV
from skimage.filters import sato
from skimage.morphology import h_minima
from skimage.segmentation import watershed
from skimage.measure import regionprops
from pathlib import Path

D = 4.0; H = 0.25; CELL = D; RIDGE = (0.32, 0.72); DEPTH = 0.1; MINA = 2.4; CON = 0.2; SM = 0.36; K = 4.0
FR = None
def frames():
    global FR
    if FR is None:
        FR = BV.load_frames(Path('/home/claude/seg2/maps'), 37.5 / 37, 0.5, {})
        for f in FR:
            f['v']['topo'] = BV.clean_topo(f, 5.0)
    return FR

def trace(f, chans=('adh_force_nN', 'adh_work_fJ')):
    v = f['v']; px, py = f['px'], f['py']
    maps = {k: v[k] for k in chans}
    valid = np.isfinite(v['topo']) & np.any([np.isfinite(a) for a in maps.values()], axis=0)
    out = dict(px=px, py=py, valid=valid, ch={})
    vi = S.to_iso(valid.astype(float), px, py, H, order=0) > 0.5
    sig = [r / H for r in RIDGE]
    R, OK = {}, {}
    for name, A in maps.items():
        tf, pol = S.CHANNELS[name]
        X = A.astype(float)
        if tf == 'log':
            X = np.where(X > 0, np.log10(X), np.nan)
        ok = np.isfinite(X) & valid
        rec = dict(raw=np.where(ok, X, np.nan))
        X1, nsp = S.despike(X, ok, K)
        rec['despiked'] = np.where(ok, X1, np.nan); rec['n_spikes'] = nsp
        rec['spike_mask'] = ok & (X1 != X)
        X2 = S.destripe(S.fill_nan(np.where(ok, X1, np.nan)), ok)
        rec['destriped'] = np.where(ok, X2, np.nan)
        rec['line_offset'] = (S.fill_nan(np.where(ok, X1, np.nan)) - X2)[:, 0]
        Xi = S.to_iso(X2, px, py, H, order=3)
        rec['iso'] = np.where(vi, Xi, np.nan)
        oki = (S.to_iso(ok.astype(float), px, py, H, order=1) > 0.99) & vi
        sy = max(0.8, S.P_YSM * px / H)
        Xi = ndi.gaussian_filter(Xi, (sy, 0.3)); rec['sig_y_px'] = sy
        rec['yeq'] = np.where(vi, Xi, np.nan)
        Xi = Xi - ndi.gaussian_filter(Xi, CELL / H)
        Xi = S.robust_z(Xi, oki)
        rec['z'] = np.where(vi, Xi, np.nan)
        r1 = sato(Xi, sigmas=[sig[0]], black_ridges=pol < 0, mode='nearest')
        r2 = sato(Xi, sigmas=[sig[1]], black_ridges=pol < 0, mode='nearest')
        rr = sato(Xi, sigmas=sig, black_ridges=pol < 0, mode='nearest')
        rec['sato_s1'], rec['sato_s2'], rec['sato'] = r1, r2, rr
        R[name] = S.robust_z(rr, oki); OK[name] = oki
        rec['ridge_z'] = np.where(vi, R[name], np.nan)
        out['ch'][name] = rec
    W = {}
    for n in R:
        others = [m for m in R if m != n]
        num = sum(np.where(OK[m], R[m], 0) for m in others); den = sum(OK[m].astype(float) for m in others)
        cons = np.where(den > 0, num / np.maximum(den, 1), np.nan); mm = OK[n] & np.isfinite(cons)
        W[n] = max(0.0, np.corrcoef(R[n][mm], cons[mm])[0, 1]) if mm.sum() > 50 else 0.0
    tot = sum(W.values()) or 1.0
    W = {n: w / tot for n, w in W.items()}
    num = sum(W[n] * np.where(OK[n], R[n], 0) for n in R); den = sum(W[n] * OK[n] for n in R)
    B = np.where((den > 0.25) & vi, num / np.maximum(den, 1e-9), np.nan)
    out.update(W=W, B=B, vi=vi)
    # segment()
    Bs = ndi.gaussian_filter(S.fill_nan(B), SM / H)
    seeds = ndi.label(h_minima(Bs, DEPTH) & vi)[0]
    lab = watershed(Bs, seeds, mask=vi, watershed_line=True)
    keep = np.zeros_like(lab); k = 0; dropped = []
    for rp in regionprops(lab):
        sl = rp.slice; pad = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
        sub = lab[pad] == rp.label
        area = sub.sum() * H * H
        ring = ndi.binary_dilation(sub, iterations=2) & ~sub & vi[pad]
        inner = np.median(Bs[pad][sub]); edge = np.median(Bs[pad][ring]) if ring.any() else inner
        if area < MINA:
            dropped.append((rp.label, 'area', area, edge - inner)); continue
        if edge - inner < CON:
            dropped.append((rp.label, 'contrast', area, edge - inner)); continue
        k += 1; keep[pad][sub] = k
    out.update(Bs=Bs, seeds=seeds, ws=lab, lab=keep, dropped=dropped)
    return out
