#!/usr/bin/env python3
"""
AFM force-volume -> Ting viscoelastic maps (batch pipeline)
===========================================================

Per dataset (folder with config.txt + ForceCurve/ForceCurve_*.lvm):
  1. load all LVM curves (parallel)
  2. hydrodynamic drag correction from the approach/retract baseline offset
  3. Ting fit (approach + retraction, in time) with power-law relaxation
         E(t) = E1 * (t / t0)^(-alpha)
     free parameters: E1 (linear), alpha, contact point
  4. topography from the fitted contact point -> local surface slope
  5. refit along the local surface normal (delta_n = delta_z cos(th), F_n = F_z / cos(th))
  6. shape clustering of the force curves (peak-normalised F(t), PCA + k-means)
  6b. adhesion analysis of the retract curves (force, work, pull-off / detachment length,
      rupture events, tether plateaus, JKR adhesion energy) -> adh_* maps
  7. everything into <out>/<dataset_name>.npz  (+ optional quick-look PNG)

LVM layout (single column): approach defl (num_app) | retract defl (num_ret) |
                            approach Z (num_app)    | retract Z (num_ret)      [V]

Usage
-----
  python afm_ting_pipeline.py DATA_DIR [DATA_DIR ...] -o OUT_DIR
  python afm_ting_pipeline.py ROOT_FOLDER -o OUT_DIR            # finds all datasets below ROOT
  python afm_ting_pipeline.py datasets.txt -o OUT_DIR           # text file, one path per line
  python afm_ting_pipeline.py ... --invols 200 --k 0.09 --radius 3 --png

Run `python afm_ting_pipeline.py -h` for all options.

Units in the npz: lengths um, forces nN, moduli Pa, time s, angles deg.
"""
import os
# one BLAS thread per worker (avoid oversubscription); must happen before numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares, nnls
from scipy.signal import savgol_filter

NRES = 10  # base scalar results per curve returned by fit_one
# extra scalars of the model variants (normal-frame pass only), appended after the NRES base values
VAR_KEYS = ["E1_labcp", "alpha_labcp", "zc_labcp", "R2_labcp",
            "E1_sq", "alpha_sq", "zc_sq", "R2_sq", "h0_sq", "A_sq",
            "E_elastic", "zc_elastic", "R2_elastic", "depth_ratio"]
NTOT = NRES + len(VAR_KEYS)
VARIANTS_ALL = ("labcp", "squeeze", "elastic", "depth")


# ----------------------------------------------------------------------------- config file (shared block)
# identical in afm_ting_pipeline.py, extract_topo_E.py, segment_cells.py, build_timelapse_viewer.py
CONFIG_KEYS = {
    "afm": ["invols_nm_per_V", "spring_constant_N_m", "z_sensitivity_um_per_V", "xy_sensitivity_um_per_V"],
    "probe": ["geometry", "radius_um", "half_angle_deg", "poisson_ratio"],
    "acquisition": ["sample_interval_s", "zigzag"],
    "preprocessing": ["drag_correction", "baseline_samples", "block_average", "z_smoothing_ms"],
    "fit": ["t0_s", "tref_s", "slope_correction", "slope_smooth_um", "variants"],
    "quality": ["r2_min", "slope_max_deg", "max_delta_over_R"],
    "clustering": ["n_clusters"],
    "adhesion": ["enabled", "threshold_min_nN", "noise_factor", "detach_smoothing_points", "rupture_window_points",
                 "rupture_min_se", "tether_smoothing_points", "tether_max_slope_nN_um", "tether_min_length_um",
                 "tether_noise_factor"],
    "export": ["e_map", "no_mask", "recompute_adhesion"],
    "registration": ["rot_max_deg", "shift_max_um", "spike_um", "recompute"],
    "segmentation": ["cell_diameter_um", "channels", "grid_um", "detrend_um", "ridge_um", "smooth_um", "min_area_um2",
                     "seed_depth", "contrast", "despike_k", "drop_excluded"],
    "tracking": ["iou_min", "max_gap_frames", "max_step_um", "max_area_change", "refine_shift_um", "refine_rot_deg",
                 "track_plane"],
    "viewer": ["display_grid_um", "outlier_k", "fill_holes_px", "default_field"],
    "run": ["cores", "name_depth", "png"],
}


def _toml_subset(txt):
    """Minimal TOML reader (used only if tomllib / tomli are missing): [section], key = value with
    numbers, "strings", true/false and [arrays]; '#' comments."""
    import ast
    out, sec = {}, None
    for raw in txt.splitlines():
        line, q = "", None
        for ch in raw:                                   # strip comments outside quotes
            if ch in "\"'":
                q = None if q == ch else (ch if q is None else q)
            if ch == "#" and q is None:
                break
            line += ch
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            sec = out.setdefault(line.strip("[] \t"), {})
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        v = re.sub(r"(?<![\w\"'])true(?![\w\"'])", "True", re.sub(r"(?<![\w\"'])false(?![\w\"'])", "False", v))
        (sec if sec is not None else out)[k] = ast.literal_eval(v)
    return out


def load_config(path):
    txt = Path(path).expanduser().read_text(encoding="utf-8")
    try:
        import tomllib                                   # Python >= 3.11
        cfg = tomllib.loads(txt)
    except ImportError:
        try:
            import tomli as tomllib                      # pip install tomli (Python < 3.11)
            cfg = tomllib.loads(txt)
        except ImportError:
            cfg = _toml_subset(txt)
    for sec, kv in cfg.items():                          # typos would otherwise be ignored silently
        if sec not in CONFIG_KEYS:
            print(f"[config] unknown section [{sec}] in {path}")
        elif isinstance(kv, dict):
            for k in kv:
                if k not in CONFIG_KEYS[sec]:
                    print(f"[config] unknown key {sec}.{k} in {path}")
    return cfg


def apply_config(ap, argv, mapping):
    """Read --config FILE and use its values as defaults of the parser (the command line still wins).
    mapping: {(section, key): dest  or  (dest, converter)}; the value "auto" becomes None (derived later)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    a, _ = pre.parse_known_args(argv)
    if not a.config:
        return None
    cfg = load_config(a.config)
    d = {}
    for (sec, key), dest in mapping.items():
        if isinstance(cfg.get(sec), dict) and key in cfg[sec]:
            v = cfg[sec][key]
            if isinstance(dest, tuple):
                dest, conv = dest
                v = None if v == "auto" else conv(v)
            elif v == "auto":
                v = None
            d[dest] = v
    ap.set_defaults(**d)
    print(f"[config] {a.config}: {len(d)} parameters")
    return cfg



# =============================================================================
#  adhesion analysis of the retract curves
# =============================================================================
"""
Adhesion analysis of AFM retract curves (shared by afm_ting_pipeline.py and extract_topo_E.py).

All quantities from the retract part (after the Z turnaround); force baseline = approach baseline (0);
distances along the tip position  s = Z - F/k  (corrected for cantilever bending), measured from the
zero-force point s0 (first retract sample with F < 0, start of the tensile part; the tip is still in
adhesive contact there). Forces are vertical (as measured).

  adh_force_nN         maximum adhesion force (-min F, >= 0)
  adh_pulloff_um       distance from the zero-force point s0 to the force minimum
  adh_work_fJ          work of adhesion = integral of -F ds over the attractive part  (1 fJ = 1 nN*um)
  adh_length_um        detachment length: s0 -> last point where the smoothed force is < -threshold
  adh_detached         1 = force back to baseline before the end of the retract;
                       0 = still attached at the end (length, work, tether length are lower bounds)
  adh_n_ruptures       discrete rupture steps after the minimum: local fit quadratic + step (+-10 samples),
                       step >= threshold and >= 6 standard errors (smooth unbinding is not counted)
  adh_last_rupture_um  position of the last rupture (NaN if none)
  adh_last_rupture_nN  height of the last rupture step (NaN if none)
  adh_tether_nN        force of the last constant-force plateau (membrane tether: >= 0.8 um long, drift
                       < 0.05 nN/um, force > 3 x noise); 0 if none
  adh_tether_len_um    length of that plateau
  adh_gamma_JKR_mJ_m2  JKR work of adhesion per area  w = 2 F_adh / (3 pi R)  (sphere only; 1 mJ/m^2 = 1 nN/um)
Threshold for "attached" / steps: max(5 x baseline noise, 0.03 nN).
"""
_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy >= 2 / < 2
ADH_DEFAULTS = dict(threshold_min_nN=0.03, noise_factor=5.0, detach_smoothing_points=7, rupture_window_points=10,
                    rupture_min_se=6.0, tether_smoothing_points=31, tether_max_slope_nN_um=0.05,
                    tether_min_length_um=0.8, tether_noise_factor=3.0)
ADH_KEYS = ["adh_force_nN", "adh_pulloff_um", "adh_work_fJ", "adh_length_um", "adh_detached",
            "adh_n_ruptures", "adh_last_rupture_um", "adh_last_rupture_nN",
            "adh_tether_nN", "adh_tether_len_um", "adh_gamma_JKR_mJ_m2"]


def _steps(F, i0, i1, thr, sigma, hw=10, n_se=6.0):
    """Discrete rupture events in F[i0:i1]: in a +-hw window fit  F = a + b x + c x^2 + h*[x > 0]
    and keep positive steps h >= thr that are also >= 6 standard errors (smooth unbinding gives h ~ 0)."""
    n = len(F)
    if i1 - i0 < 3:
        return []
    x = np.arange(-hw, hw + 1, dtype=float)
    X = np.c_[np.ones_like(x), x, x * x, (x > 0).astype(float)]
    XtXi = np.linalg.inv(X.T @ X)
    se_h = np.sqrt(XtXi[3, 3])
    pre = F[np.arange(max(i0, hw), min(i1, n - hw - 1)) + 2].astype(float)  # cheap pre-filter
    ev = []
    for i in range(max(i0, hw), min(i1, n - hw - 1)):
        if F[i + 1:i + 4].mean() - F[i - 2:i + 1].mean() < 0.5 * thr:
            continue
        y = F[i - hw:i + hw + 1]
        c = XtXi @ (X.T @ y)
        res = y - X @ c
        sig = max(np.sqrt(res @ res / (len(y) - 4)), sigma)
        h = c[3]
        if h >= thr and h >= n_se * sig * se_h:
            ev.append((i, float(h)))
    out = []
    for i, st in ev:
        if out and i - out[-1][0] <= hw:
            if st > out[-1][1]:
                out[-1] = (i, st)
        else:
            out.append((i, st))
    return out


def _plateaus(Fs, s, i0, i1, thr, max_slope=0.05, min_len=0.8):
    """Runs where the smoothed force is attractive and nearly constant (|dF/ds| < max_slope nN/um),
    at least min_len um long. Returns [(start, end), ...]."""
    if i1 - i0 < 5:
        return []
    ds = np.gradient(s[i0:i1])
    ds[np.abs(ds) < 1e-9] = 1e-9
    dF = np.gradient(Fs[i0:i1]) / ds
    ok = (Fs[i0:i1] < -thr) & (np.abs(dF) < max_slope)
    runs, j = [], 0
    while j < len(ok):
        if ok[j]:
            k = j
            while k + 1 < len(ok) and ok[k + 1]:
                k += 1
            if s[i0 + k] - s[i0 + j] >= min_len:
                runs.append((i0 + j, i0 + k))
            j = k + 1
        else:
            j += 1
    return runs


def adhesion_curve(F, Z, k_nN_um, noise_nN, thr_min=None, nsig=None, details=False, prm=None):
    q = dict(ADH_DEFAULTS, **(prm or {}))
    thr_min = q["threshold_min_nN"] if thr_min is None else thr_min
    nsig = q["noise_factor"] if nsig is None else nsig
    sgd, sgt = int(q["detach_smoothing_points"]) | 1, int(q["tether_smoothing_points"]) | 1
    out = dict.fromkeys(ADH_KEYS, np.nan)
    if not (np.isfinite(F).all() and np.isfinite(Z).all()) or len(F) < 50:
        return (out, None) if details else out
    ip = int(np.argmax(Z))                                   # Z turnaround
    Fr = F[ip:]
    s = (Z - F / k_nN_um)[ip:]
    s = s[0] - s                                             # tip retract distance, increasing
    thr = max(nsig * (noise_nN if np.isfinite(noise_nN) else 0.0), thr_min)
    neg = np.where(Fr < 0)[0]
    if len(neg) == 0:                                        # no adhesion at all
        out.update(adh_force_nN=0.0, adh_pulloff_um=0.0, adh_work_fJ=0.0, adh_length_um=0.0, adh_detached=1.0,
                   adh_n_ruptures=0.0, adh_tether_nN=0.0, adh_tether_len_um=0.0)
        return (out, None) if details else out
    j0 = neg[0]                                              # zero-force point s0 (start of tensile part)
    jm = j0 + int(np.argmin(Fr[j0:]))                        # force minimum
    Fs = savgol_filter(Fr, sgd, 2) if len(Fr) > sgd + 2 else Fr
    below = np.where(Fs[j0:] < -thr)[0]
    jl = j0 + below[-1] if len(below) else jm                # last attached point
    detached = float(jl < len(Fr) - 6)
    seg = slice(j0, min(jl + 2, len(Fr)))
    work = float(_trapz(np.maximum(-Fr[seg], 0), s[seg]))
    sigma = noise_nN if np.isfinite(noise_nN) else thr / nsig
    hw = int(q["rupture_window_points"])
    ev = _steps(Fr, jm, jl + 6, thr, sigma, hw, q["rupture_min_se"]) if jl > jm else []
    ev = [(i, st) for i, st in ev if Fs[max(i - 3, 0)] < -thr]
    Fw = savgol_filter(Fr, sgt, 2) if len(Fr) > sgt + 2 else Fs
    pl = _plateaus(Fw, s, jm, min(jl + 1, len(Fr)), max(q["tether_noise_factor"] * sigma, thr_min),
                   q["tether_max_slope_nN_um"], q["tether_min_length_um"])
    if pl:
        a, b = pl[-1]
        tether, tlen = float(-np.median(Fr[a:b + 1])), float(s[b] - s[a])
    else:
        tether, tlen = 0.0, 0.0
    out.update(adh_force_nN=float(-Fr[jm]), adh_pulloff_um=float(s[jm] - s[j0]), adh_work_fJ=work,
               adh_length_um=float(s[jl] - s[j0]), adh_detached=detached, adh_n_ruptures=float(len(ev)),
               adh_last_rupture_um=float(s[ev[-1][0]] - s[j0]) if ev else np.nan,
               adh_last_rupture_nN=float(ev[-1][1]) if ev else np.nan,
               adh_tether_nN=tether, adh_tether_len_um=tlen)
    if details:
        return out, dict(s=s, F=Fr, j0=j0, jm=jm, jl=jl, events=ev, plateaus=pl, thr=thr)
    return out


def adhesion_maps(curves_F, curves_Z, k_N_m, noise_map, curve_index_map, radius_um=None, prm=None):
    """{key: map (ny, nx) float32} computed from the stored curves of a pipeline .npz."""
    k = k_N_m * 1e3                                          # nN/um
    idx = np.asarray(curve_index_map, float)
    noise = np.asarray(noise_map, float)
    res = {key: np.full(idx.shape, np.nan, np.float32) for key in ADH_KEYS}
    for (iy, ix), ci in np.ndenumerate(idx):
        if not np.isfinite(ci):
            continue
        r = adhesion_curve(curves_F[int(ci)].astype(float), curves_Z[int(ci)].astype(float), k, noise[iy, ix],
                           prm=prm)
        for key in ADH_KEYS[:-1]:
            res[key][iy, ix] = r[key]
    if radius_um:
        res["adh_gamma_JKR_mJ_m2"] = (2 * res["adh_force_nN"] / (3 * np.pi * radius_um)).astype(np.float32)
    return res



# =============================================================================
#  config / file discovery
# =============================================================================
def read_config(path):
    cfg = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                k, v = line.split(",", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def _cfg_num(cfg, key, default, typ=float):
    try:
        return typ(float(cfg[key]))
    except (KeyError, ValueError):
        return default


def lvm_files(ds):
    d = ds / "ForceCurve" if (ds / "ForceCurve").is_dir() else ds
    fs = list(d.glob("ForceCurve_*.lvm"))
    num = lambda p: int(m.group(1)) if (m := re.search(r"(\d+)$", p.stem)) else -1
    return sorted(fs, key=num)


def is_dataset(p):
    return p.is_dir() and (p / "config.txt").is_file() and len(lvm_files(p)) > 0


def discover(inputs):
    found = []
    for item in inputs:
        p = Path(item).expanduser()
        if _safe(p.is_file) and p.suffix.lower() in (".txt", ".lst", ".list"):
            found += discover(read_list_file(p))
        elif _safe(lambda: is_dataset(p)):
            found.append(p.resolve())
        elif _safe(p.is_dir):
            for cfg in sorted(p.rglob("config.txt")):
                if is_dataset(cfg.parent):
                    found.append(cfg.parent.resolve())
        else:
            print(f"[warn] not a dataset / folder / list file: {item}")
    out, seen = [], set()
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _safe(fn):
    try:
        return fn()
    except (OSError, ValueError):
        return False


def read_list_file(p):
    """Paths one per line, or several per line separated by whitespace / ; / ,
    (a new path must start with '/', '~', '.' or a drive letter, so paths containing
    spaces still work when given one per line). '#' starts a comment."""
    out = []
    for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip().strip('"').strip("'")
        if not line:
            continue
        if _safe(lambda: Path(line).expanduser().exists()):
            out.append(line)
            continue
        toks = re.split(r'[\s;,]+(?=["\']?(?:/|~|\.{1,2}[\\/]|[A-Za-z]:[\\/]|\\\\))', line)
        out += [t.strip().strip('"').strip("'").rstrip(";,") for t in toks if t.strip()]
    return out


def dataset_name(ds, depth):
    parts = [re.sub(r"[^\w\-.]+", "_", x) for x in ds.parts[-depth:]]
    return "_".join(parts)


# =============================================================================
#  contact mechanics + numerical Ting model
# =============================================================================
_U = np.unique(np.r_[np.linspace(0, 0.9, 2000), 1 - np.logspace(-1, -9, 2000)])
_LG = np.log((1 + _U) / (1 - _U))
_SPH_D = _U / 2 * _LG                    # delta / R
_SPH_P = (_U ** 2 + 1) / 2 * _LG - _U    # F (1-nu^2) / (E R^2)


def phi(delta, P):
    """Elastic indentation law F*(1-nu^2)/E  [um^2], delta in um."""
    g = P["geometry"]
    if g == "sphere":                       # exact Sneddon sphere
        R = P["radius"]
        return R * R * np.interp(delta / R, _SPH_D, _SPH_P)
    if g == "paraboloid":                   # Hertz
        return 4.0 / 3.0 * np.sqrt(P["radius"]) * delta ** 1.5
    if g == "cone":                         # Sneddon cone
        return 2.0 * np.tan(np.deg2rad(P["half_angle"])) / np.pi * delta ** 2
    raise ValueError(g)


def kernel_weights(alpha, n, dt, t0):
    """Discretised power-law relaxation kernel (exact integral over each step)."""
    m = np.arange(n + 1.0)
    return (m[1:] ** (1 - alpha) - m[:-1] ** (1 - alpha)) * dt ** (-alpha) * t0 ** alpha / (1 - alpha)


def ting_shape(delta, alpha, dt, P):
    """F/E1 for indentation history delta (uniform time grid, delta[0]=0 at contact).
    Loading: hereditary integral. Unloading: Ting's t1(t) construction."""
    N = len(delta)
    g = np.zeros(N)
    if N < 3:
        return g
    w = kernel_weights(alpha, N, dt, P["t0"])
    dq = np.diff(phi(delta, P))
    dd = np.diff(delta)
    im = int(np.argmax(delta))
    g[1:im + 1] = np.convolve(dq[:im], w[:im])[:im]
    if im < N - 1:
        ii = np.arange(im + 1, N)[:, None]
        jj = np.arange(N - 1)[None, :]
        idx = ii - 1 - jj
        W = np.where(idx >= 0, w[np.clip(idx, 0, N - 1)], 0.0)
        Rm = np.cumsum((W * dd)[:, ::-1], axis=1)[:, ::-1]
        WQ = W * dq
        Q = np.cumsum(WQ, axis=1)
        Ra = Rm[:, :im + 1]
        pos = Ra >= 0
        has = pos.any(1)
        j1 = np.where(has, im - np.argmax(pos[:, ::-1], axis=1), 0)
        r = np.arange(len(j1))
        Rj = Ra[r, j1]
        Rn = Rm[r, np.minimum(j1 + 1, N - 2)]
        frac = np.clip(Rj / np.maximum(Rj - Rn, 1e-30), 0, 1)
        Qm = np.where(j1 > 0, Q[r, np.maximum(j1 - 1, 0)], 0.0)
        g[im + 1:] = np.where(has, np.maximum(Qm + (1 - frac) * WQ[r, j1], 0), 0.0)
    return g / (1 - P["nu"] ** 2)


def ting_model(p, t, s, F, i_end, P, want=False):
    alpha, zc = p
    delta = np.maximum(s - zc, 0)
    ip = int(np.argmax(s))
    below = np.where(s[:ip + 1] <= zc)[0]
    i0 = below[-1] if len(below) else 0
    g = np.zeros_like(s)
    seg = delta[i0:].copy()
    seg[0] = 0
    g[i0:] = ting_shape(seg, alpha, t[1] - t[0], P)
    gg, FF = g[:i_end], F[:i_end]
    E1 = max(gg @ FF / max(gg @ gg, 1e-30), 0.0)      # E1 solved linearly
    if want:
        return E1, E1 * g, delta
    return E1 * gg - FF


def _shape(alpha, zc, t, s, P):
    """model force per unit E1 (E1 = 1 Pa -> nN) for contact point zc"""
    ip = int(np.argmax(s))
    below = np.where(s[:ip + 1] <= zc)[0]
    i0 = below[-1] if len(below) else 0
    g = np.zeros_like(s)
    seg = np.maximum(s - zc, 0)[i0:].copy()
    seg[0] = 0
    g[i0:] = ting_shape(seg, alpha, t[1] - t[0], P)
    return g


def _r2(Fm, F, sl):
    r = (Fm - F)[sl]
    return 1 - np.sum(r ** 2) / max(np.sum((F[sl] - F[sl].mean()) ** 2), 1e-30)


def lab_contact_index(F, zz, ip):
    """Contact point of the lab FV pipeline (calibrate/cp_young_analysis): linear baseline over the first third of
    the approach travel, cubic baseline vs Z up to the last point < 0.2 nN, then the first point after which the
    smoothed force stays > mean + 5 SD of the first 15 % for 10 points, traced back (<= 50 points) to the mean.
    Returns (index, baseline-corrected force) or (None, None)."""
    za = zz[:ip + 1]
    fe = int(np.argmin(np.abs((za - za[0]) - (za[ip] - za[0]) / 3)))
    if fe < 10:
        return None, None
    Fb = F - np.polyval(np.polyfit(zz[:fe], F[:fe], 1), zz)
    lo = np.where(Fb[:ip] < 0.2)[0]
    if len(lo) < 10:
        return None, None
    tc = lo[-1]
    p3 = np.polyfit(zz[:tc], Fb[:tc], 3)
    base = np.polyval(p3, zz)
    base[tc:] = np.polyval(p3, zz[tc])
    Fb = Fb - base
    fs = savgol_filter(Fb, 11, 3)
    fm = int(np.argmax(fs))
    bl = fs[:max(5, int(0.15 * len(fs)))]
    m, sd = bl.mean(), bl.std()
    thr = m + 5 * sd
    for i in np.where(fs[:fm] > thr)[0]:
        if i + 10 < fm and np.all(fs[i:i + 10] > thr) and fs[i:i + 10].mean() >= fs[i]:
            for j in range(i, max(0, i - 50), -1):
                if fs[j] <= m:
                    return j, Fb
            return i, Fb
    return None, None


def fit_variants(t, s, F, zz, sig, c, P, ic, ip, i_end, main_x, which):
    """Model variants on the same curve (surface-normal frame for c < 1). Returns dict with VAR_KEYS."""
    out = dict.fromkeys(VAR_KEYS, np.nan)
    sN, FN = s * c, F / c
    sl = slice(max(ic - 10, 0), i_end)
    # (1) lab contact point: fixed at the start of the toe, alpha and E1 fitted
    if "labcp" in which:
        j, Fb = lab_contact_index(F, zz, ip)
        if j is not None:
            zl = sN[j]
            FbN = Fb / c
            f = lambda p: (lambda g: max(g[:i_end] @ FbN[:i_end] / max(g[:i_end] @ g[:i_end], 1e-30), 0) * g[:i_end]
                           - FbN[:i_end])(_shape(p[0], zl, t, sN, P))
            try:
                r = least_squares(f, [0.4], bounds=([0.0], [0.95]), diff_step=[1e-2])
                g = _shape(r.x[0], zl, t, sN, P)
                E1 = max(g[:i_end] @ FbN[:i_end] / max(g[:i_end] @ g[:i_end], 1e-30), 0)
                out.update(E1_labcp=E1 * 1e3, alpha_labcp=r.x[0], zc_labcp=zl / c,
                           R2_labcp=_r2(E1 * g * c, Fb, sl))
            except Exception:
                pass
    # (2) squeeze film: F = E1 g(alpha, zc) + A v/(h + h0), h = max(zc - s, 0); E1, A >= 0 linear
    if "squeeze" in which:
        v = np.gradient(savgol_filter(sN, 15, 2), t)
        def lin(p):
            g = _shape(p[0], p[1], t, sN, P)
            hy = v / (np.maximum(p[1] - sN, 0) + 10 ** p[2])
            X = np.c_[g, hy][:i_end]
            co, _ = nnls(X, FN[:i_end])
            return co, g, hy
        best = None
        for x0 in ([main_x[0], main_x[1], -1.0], [main_x[0], main_x[1] - 0.3 * c, -0.7], [0.4, sN[ic], -1.3]):
            try:
                r = least_squares(lambda p: (lambda co, g, hy: co[0] * g[:i_end] + co[1] * hy[:i_end] - FN[:i_end])(*lin(p)),
                                  x0, bounds=([0.0, sN[ic] - 4 * c, -2.5], [0.95, sN[ic] + 1.5 * c, 0.0]),
                                  diff_step=[1e-2, 2e-3 / max(abs(x0[1]), 1), 1e-2], x_scale=[0.1, 0.1, 0.3])
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is not None:
            co, g, hy = lin(best.x)
            out.update(E1_sq=co[0] * 1e3, alpha_sq=best.x[0], zc_sq=best.x[1] / c, h0_sq=10 ** best.x[2],
                       A_sq=co[1] * c, R2_sq=_r2((co[0] * g + co[1] * hy) * c, F, sl))
    # (3) elastic (alpha = 0, Hertz/Sneddon), approach only, contact point fitted
    if "elastic" in which:
        sa, Fa = sN[:ip + 1], FN[:ip + 1]
        def fe(p):
            g = phi(np.maximum(sa - p[0], 0), P) / (1 - P["nu"] ** 2)
            return max(g @ Fa / max(g @ g, 1e-30), 0) * g - Fa
        try:
            r = least_squares(fe, [main_x[1]], bounds=([sN[ic] - 4 * c], [sN[ic] + 1.5 * c]), diff_step=[1e-5])
            g = phi(np.maximum(sa - r.x[0], 0), P) / (1 - P["nu"] ** 2)
            Ee = max(g @ Fa / max(g @ g, 1e-30), 0)
            out.update(E_elastic=Ee * 1e3, zc_elastic=r.x[0] / c,
                       R2_elastic=_r2(Ee * g * c, F[:ip + 1], slice(max(ic - 10, 0), ip + 1)))
        except Exception:
            pass
    # (4) depth consistency of the main fit: E1 from the first 40 % of the indentation / E1 from all of it
    if "depth" in which:
        g = _shape(main_x[0], main_x[1], t, sN, P)
        d = np.maximum(sN - main_x[1], 0)
        dm = d[:ip + 1].max()
        Ek = []
        for frac in (0.4, 1.0):
            m = (np.arange(len(sN)) <= ip) & (d > 0) & (d <= frac * dm)
            Ek.append(g[m] @ FN[m] / (g[m] @ g[m]) if m.sum() > 3 and g[m] @ g[m] > 0 else np.nan)
        out["depth_ratio"] = Ek[0] / Ek[1] if Ek[1] else np.nan
    return out


# =============================================================================
#  per-curve processing (runs in worker processes)
# =============================================================================
_W = {}   # worker globals


def _init_worker(cache_path, P, drag_b, cos_th, variants=()):
    _W["var"] = tuple(variants)
    _W["raw"] = np.load(cache_path, mmap_mode="r")
    _W["P"] = P
    _W["b"] = drag_b
    _W["cos"] = cos_th


def prep(a, P, b):
    na, nr, dt = P["num_app"], P["num_ret"], P["dt"]
    d = np.r_[a[:na], a[na:na + nr]].astype(float)
    z = np.r_[a[na + nr:2 * na + nr], a[2 * na + nr:2 * na + 2 * nr]].astype(float)
    win = P["savgol"]
    zs = savgol_filter(z, win, 3) * P["zsens"]
    v = savgol_filter(z, win, 3, deriv=1) * P["zsens"] / dt
    dc = d - b * v
    base = P["baseline"]
    dc = dc - np.median(dc[:base])
    blk = P["block"]
    n = (len(d) // blk) * blk
    B = lambda x: x[:n].reshape(-1, blk).mean(1)
    defl = B(dc) * P["invols"]                         # um
    zz = B(zs)
    t = B(np.arange(len(d)) * dt)
    F = defl * P["k"] * 1e3                            # nN
    sig = np.std(dc[:base]) * P["invols"] * P["k"] * 1e3 / np.sqrt(blk)
    return t, zz - defl, F, zz, sig


def fit_one(a, P, b, c=1.0, want_curves=True, variants=()):
    """c = cos(surface slope). Returns (scalars[NTOT], F, Fmodel, Z) ; F, Fmodel vertical (measured) frame.
    variants: subset of VARIANTS_ALL computed in addition to the main fit (NaN otherwise)."""
    t, s, F, zz, sig = prep(a, P, b)
    sN, FN = s * c, F / c
    ip = int(np.argmax(F))
    Fs = savgol_filter(F, 9, 2)
    thr = max(4 * sig, 0.03 * Fs[ip])
    lo = np.where(Fs[:ip] < thr)[0]
    ic = lo[-1] if len(lo) else 0
    neg = np.where(Fs[ip:] < 0)[0]
    i_end = ip + neg[0] if len(neg) else len(F)       # fit window: up to first F<0 after peak
    best = None
    for dz in (0.0, -0.4):
        x0 = [0.3, sN[ic] + dz * c]
        try:
            r = least_squares(ting_model, x0, args=(t, sN, FN, i_end, P),
                              bounds=([0.0, sN[ic] - 4 * c], [0.95, sN[ic] + 1.5 * c]),
                              diff_step=[1e-2, 2e-3 / max(abs(x0[1]), 1)], x_scale=[0.1, 0.1])
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("fit failed")
    E1, FmN, dn = ting_model(best.x, t, sN, FN, i_end, P, True)
    Fm = FmN * c
    cc = slice(max(ic - 10, 0), i_end)
    res = (Fm - F)[cc]
    r2 = 1 - np.sum(res ** 2) / np.sum((F[cc] - F[cc].mean()) ** 2)
    zc_v = best.x[1] / c
    ipz = int(np.argmax(s))
    z_contact = float(np.interp(zc_v, s[:ipz + 1], zz[:ipz + 1])) if ipz > 2 else np.nan
    adh = float(F[ip:].min())
    vr = fit_variants(t, s, F, zz, sig, c, P, ic, ip, i_end, best.x, variants) if variants else {}
    out = np.array([E1 * 1e3, best.x[0], zc_v, dn.max(), r2, F[ip], z_contact, i_end, sig, adh] +
                   [vr.get(k, np.nan) for k in VAR_KEYS])
    if want_curves:
        return out, F.astype(np.float32), Fm.astype(np.float32), zz.astype(np.float32)
    return out, None, None, None


def _task(i):
    P = _W["P"]
    a = np.asarray(_W["raw"][i])
    c = 1.0 if _W["cos"] is None else float(_W["cos"][i])
    try:
        if not np.isfinite(a).all():
            raise ValueError("bad curve")
        return i, *fit_one(a, P, _W["b"], c, variants=_W.get("var", ()))
    except Exception:
        return i, np.full(NTOT, np.nan), None, None, None


def _read_task(path_and_n):
    path, n = path_and_n
    try:
        a = np.loadtxt(path, dtype=np.float32)
        if a.size < n:
            raise ValueError
        return a[:n]
    except Exception:
        return np.full(n, np.nan, np.float32)


# =============================================================================
#  helpers run in the main process
# =============================================================================
def run_pool(fn, items, ncores, initargs=None, label="", chunksize=None):
    items = list(items)
    n = len(items)
    chunksize = chunksize or max(1, min(50, n // (ncores * 8) or 1))
    out = [None] * n
    t0 = time.time()
    kw = dict(max_workers=ncores)
    if initargs is not None:
        kw.update(initializer=_init_worker, initargs=initargs)
    with ProcessPoolExecutor(**kw) as ex:
        step = max(1, n // 20)
        for k, r in enumerate(ex.map(fn, items, chunksize=chunksize)):
            out[k] = r
            if (k + 1) % step == 0 or k + 1 == n:
                el = time.time() - t0
                print(f"\r    {label}: {k+1}/{n}  {el:6.0f}s  (eta {el/(k+1)*(n-k-1):5.0f}s)",
                      end="", flush=True)
    print()
    return out


def drag_coefficient(raw, P):
    """b [V per um/s]: deflection offset between approach and retract baselines / speed difference."""
    na, nr, dt, base = P["num_app"], P["num_ret"], P["dt"], P["baseline"]
    sub = raw[::max(1, len(raw) // 300)]
    sub = sub[np.isfinite(sub).all(1)]
    d = np.c_[sub[:, :na], sub[:, na:na + nr]].astype(float)
    z = np.c_[sub[:, na + nr:2 * na + nr], sub[:, 2 * na + nr:]].astype(float)
    v = savgol_filter(z, P["savgol"], 3, deriv=1, axis=1) * P["zsens"] / dt
    tail = min(300, nr // 4)
    num = np.median(d[:, :base], 1) - np.median(d[:, -tail:], 1)
    den = np.median(v[:, :base], 1) - np.median(v[:, -tail:], 1)
    bs = num / den
    bs = bs[np.isfinite(bs)]
    vapp = np.median(v[:, na - 200:na])
    vret = np.median(v[:, -tail:])
    return float(np.median(bs)) if len(bs) else 0.0, float(vapp), float(vret)


def to_grid(x, nx, ny, zig):
    full = np.full(nx * ny, np.nan, dtype=float)
    m = min(len(x), nx * ny)
    full[:m] = x[:m]
    g = full.reshape(ny, nx)
    if zig:
        g[1::2] = g[1::2, ::-1]
    return g


def slope_map(topo, px, py, smooth_um):
    t = topo.copy()
    nanm = ~np.isfinite(t)
    if nanm.any():
        t[nanm] = np.nanmedian(t)
    ts = ndi.gaussian_filter(t, (smooth_um / py, smooth_um / px), mode="nearest")
    gy, gx = np.gradient(ts, py, px)
    th = np.arctan(np.hypot(gx, gy))
    th[nanm] = np.nan
    return th


def kmeans(X, k, n_init=10, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(n_init):
        C = X[[rng.integers(len(X))]]
        for _j in range(1, k):                                   # k-means++
            d2 = np.min(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.vstack([C, X[rng.choice(len(X), p=d2 / d2.sum())]])
        for _it in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            Cn = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def shape_clusters(F, valid_fit, k, npc=4):
    """k-means on PCA of peak-normalised force curves (the Z(t) ramp is common to all curves)."""
    lab = np.full(len(F), -1)
    ok = valid_fit & np.isfinite(F).all(1) & (F.max(1) > 0)
    if k < 2 or ok.sum() < 5 * k:
        return lab, None, None
    Fn = F[ok] / F[ok].max(1, keepdims=True)
    act = np.where(np.percentile(np.abs(Fn), 90, axis=0) > 0.02)[0]
    lo, hi = max(act.min() - 20, 0), min(act.max() + 20, F.shape[1])
    X = Fn[:, lo:hi]
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :npc] * S[:npc]
    Z /= Z[:, 0].std()
    l = kmeans(Z, k)
    # order clusters by size (largest = 0)
    order = np.argsort([-np.sum(l == j) for j in range(k)])
    l = np.argsort(order)[l]
    lab[ok] = l
    mean_shapes = np.array([np.median(Fn[l == j], 0) for j in range(k)], dtype=np.float32)
    return lab, mean_shapes, (S[:npc] ** 2 / np.sum(S ** 2))


# =============================================================================
#  one dataset
# =============================================================================
def process_dataset(ds, out_dir, args, ncores):
    name = dataset_name(ds, args.name_depth)
    out_file = out_dir / f"{name}.npz"
    if out_file.exists() and not args.overwrite:
        print(f"[skip] {out_file.name} exists (use --overwrite)")
        return
    t_start = time.time()
    cfg = read_config(ds / "config.txt")
    na = _cfg_num(cfg, "num_app", 3000, int)
    nr = _cfg_num(cfg, "num_ret", 3500, int)
    nx = _cfg_num(cfg, "Xstep", 1, int)
    ny = _cfg_num(cfg, "Ystep", 1, int)
    zig = (cfg.get("Zig", "FALSE").strip().upper() == "TRUE") if args.zigzag is None else bool(args.zigzag)
    dt = args.dt if args.dt else _cfg_num(cfg, "loop_time", 9.0) * 1e-6
    xl = _cfg_num(cfg, "xlength", 0.0) * args.xysens
    yl = _cfg_num(cfg, "ylength", 0.0) * args.xysens
    px = xl / nx if nx else np.nan
    py = yl / ny if ny else np.nan
    win = int(round(args.savgol_ms * 1e-3 / dt)) | 1
    block = args.block if args.block else max(1, int(round((na + nr) / 800)))          # auto: ~800 points
    baseline = args.baseline if args.baseline else min(600, na // 3)                  # auto
    sl_smooth = args.slope_smooth if args.slope_smooth else max(px, py, args.radius / 3 if args.geometry != "cone" else 0)
    P = dict(num_app=na, num_ret=nr, dt=dt, zsens=args.zsens, invols=args.invols * 1e-3,
             k=args.k, geometry=args.geometry, radius=args.radius, half_angle=args.half_angle,
             nu=args.nu, t0=args.t0, block=block, savgol=max(win, 31),
             baseline=min(baseline, na // 3))
    files = lvm_files(ds)
    N = len(files)
    print(f"\n=== {ds}\n    -> {out_file.name}   {N} curves, grid {nx}x{ny} "
          f"({xl:.1f}x{yl:.1f} um), dt={dt*1e6:.2f} us, {ncores} workers")
    if N != nx * ny:
        print(f"    [warn] {N} files but grid is {nx}x{ny}={nx*ny}; missing pixels -> NaN")

    # ---- 1. load ------------------------------------------------------------
    n_tot = 2 * na + 2 * nr
    raws = run_pool(_read_task, [(str(f), n_tot) for f in files], ncores, label="load")
    raw = np.stack(raws).astype(np.float32)
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{name}_raw.npy"
    np.save(cache, raw)

    # ---- 2. drag ------------------------------------------------------------
    b, vapp, vret = (0.0, np.nan, np.nan)
    b_est, vapp, vret = drag_coefficient(raw, P)
    if not args.no_drag:
        b = b_est
    print(f"    Z speed: approach {vapp:.0f} um/s, retract {abs(vret):.0f} um/s ; "
          f"drag {b*P['invols']*P['k']*1e6:.3f} pN s/um {'(off)' if args.no_drag else ''}")

    # ---- 3. vertical Ting fit ----------------------------------------------
    var = tuple(v for v in VARIANTS_ALL if v in args.variants)
    res = run_pool(_task, range(N), ncores, initargs=(str(cache), P, b, None, var if args.no_slope else ()),
                   label="Ting fit (vertical)" + (" + variants" if args.no_slope and var else ""))
    R1 = np.array([r[1] for r in res])
    curves = res

    # ---- 4/5. slope + refit along normal ------------------------------------
    topo_flat = np.nanmax(R1[:, 6]) - R1[:, 6]
    topo = to_grid(topo_flat, nx, ny, zig)
    th_grid = slope_map(topo, px, py, sl_smooth)
    th_flat = th_grid.copy()
    if zig:
        th_flat[1::2] = th_flat[1::2, ::-1]
    th_flat = th_flat.ravel()[:N]
    th_flat = np.where(np.isfinite(th_flat), th_flat, 0.0)
    if args.no_slope:
        R2_ = R1
    else:
        cos_th = np.cos(np.minimum(th_flat, np.deg2rad(85)))
        res = run_pool(_task, range(N), ncores, initargs=(str(cache), P, b, cos_th, var),
                       label="Ting fit (surface normal)" + (" + variants" if var else ""))
        R2_ = np.array([r[1] for r in res])
        curves = res
    if not args.keep_cache:
        try:
            cache.unlink()
            if not any(cache_dir.iterdir()):
                cache_dir.rmdir()
        except OSError:
            pass

    # ---- assemble curves ----------------------------------------------------
    M = max((len(r[2]) for r in curves if r[2] is not None), default=0)
    Fc = np.full((N, M), np.nan, np.float32)
    Fmc = np.full((N, M), np.nan, np.float32)
    Zc = np.full((N, M), np.nan, np.float32)
    for i, r in enumerate(curves):
        if r[2] is not None:
            Fc[i], Fmc[i], Zc[i] = r[2], r[3], r[4]
    t_s = (np.arange(M) * P["block"] + (P["block"] - 1) / 2) * dt

    # ---- validity + derived quantities -------------------------------------
    E1, al, dmax, r2 = R2_[:, 0], R2_[:, 1], R2_[:, 3], R2_[:, 4]
    slope_deg = np.degrees(th_flat)
    valid = (r2 > args.r2_min) & (E1 > 0) & np.isfinite(E1) & (al > 1e-3) & (al < 0.949)
    if args.geometry in ("sphere", "paraboloid"):
        valid &= dmax < args.max_delta_over_R * args.radius
    if not args.no_slope:
        valid &= slope_deg < args.slope_max
    lab, mean_shapes, pc_var = shape_clusters(Fc, np.isfinite(R2_[:, 4]) & (R2_[:, 4] > 0.5), args.clusters)

    G = lambda x: to_grid(np.asarray(x, float), nx, ny, zig).astype(np.float32)
    out = dict(
        # maps (ny, nx); row 0 = first scan line, column 0 = first point of a line
        topo_um=topo, slope_deg=G(slope_deg),
        E1_Pa=G(E1), alpha=G(al),
        E1_vertical_Pa=G(R1[:, 0]), alpha_vertical=G(R1[:, 1]),
        delta_max_um=G(dmax), Fmax_nN=G(R2_[:, 5]), adhesion_nN=G(R2_[:, 9]),
        R2=G(r2), noise_nN=G(R2_[:, 8]), z_contact_um=G(R2_[:, 6]),
        valid=G(valid) > 0.5, cluster=G(lab).astype(np.int16),
        # curves (N, M) in acquisition order, decimated; vertical (measured) frame
        curves_F_nN=Fc, curves_Fmodel_nN=Fmc, curves_Z_um=Zc, curves_t_s=t_s.astype(np.float32),
        curve_index_map=to_grid(np.arange(N, dtype=float), nx, ny, zig),
        # axes / metadata
        x_um=(np.arange(nx) + 0.5) * px, y_um=(np.arange(ny) + 0.5) * py,
        pixel_um=np.array([px, py]),
        v_approach_um_s=vapp, v_retract_um_s=abs(vret),
        drag_V_per_um_s=b, drag_pN_s_per_um=b * P["invols"] * P["k"] * 1e6,
        params_json=json.dumps({**P, "invols_nm_per_V": args.invols, "xysens_um_per_V": args.xysens,
                                "slope_correction": not args.no_slope, "slope_smooth_um": sl_smooth, "zigzag": zig,
                                "adhesion": adh_prm(args),
                                "slope_max_deg": args.slope_max, "r2_min": args.r2_min,
                                "model": "Ting, E(t)=E1*(t/t0)^-alpha"}),
        config_json=json.dumps(cfg), dataset=str(ds),
        files=np.array([f.name for f in files]),
    )
    # adhesion from the retract curves (independent of the Ting fit)
    if not args.no_adhesion:
        A = adhesion_maps(Fc, Zc, args.k, out["noise_nN"], out["curve_index_map"],
                          args.radius if args.geometry == "sphere" else None, adh_prm(args))
        out.update(A)
        print(f"    adhesion: F {np.nanmedian(A['adh_force_nN']):.2f} nN, W {np.nanmedian(A['adh_work_fJ']):.2f} fJ, "
              f"detach length {np.nanmedian(A['adh_length_um']):.2f} um, tethers {100*np.nanmean(A['adh_tether_nN'] > 0):.0f}%, "
              f"ruptures in {100*np.nanmean(A['adh_n_ruptures'] > 0):.0f}% of curves")
    V = {k: R2_[:, NRES + j] for j, k in enumerate(VAR_KEYS)}
    zc_main = R2_[:, 2]
    fam = [("", E1, al), ("_vertical", R1[:, 0], R1[:, 1])]
    if "labcp" in var:
        fam.append(("_labcp", V["E1_labcp"], V["alpha_labcp"]))
        out.update(E1_labcp_Pa=G(V["E1_labcp"]), alpha_labcp=G(V["alpha_labcp"]), R2_labcp=G(V["R2_labcp"]),
                   cp_shift_labcp_um=G(V["zc_labcp"] - zc_main))
    if "squeeze" in var:
        fam.append(("_sq", V["E1_sq"], V["alpha_sq"]))
        eta = V["A_sq"] * 1e-9 / (6 * np.pi * (args.radius * 1e-6) ** 2) * 1e3       # A [nN s] -> eta [mPa s]
        out.update(E1_sq_Pa=G(V["E1_sq"]), alpha_sq=G(V["alpha_sq"]), R2_sq=G(V["R2_sq"]),
                   cp_shift_sq_um=G(V["zc_sq"] - zc_main), h0_sq_um=G(V["h0_sq"]), eta_sq_mPas=G(eta))
    if "elastic" in var:
        out.update(E_elastic_Pa=G(V["E_elastic"]), R2_elastic=G(V["R2_elastic"]),
                   cp_shift_elastic_um=G(V["zc_elastic"] - zc_main))
    if "depth" in var:
        out["E_depth_ratio"] = G(V["depth_ratio"])
    for tr in args.tref:                       # E(t) = E1 (t/t0)^-alpha at the reference times, every variant
        for suf, E_, a_ in fam:
            out[f"E_at_{tr*1e3:g}ms{suf}_Pa"] = G(E_ * (tr / args.t0) ** (-a_))
    if mean_shapes is not None:
        out["cluster_mean_shape"] = mean_shapes
        out["cluster_pca_var"] = pc_var
    np.savez_compressed(out_file, **out)

    # ---- summary -------------------------------------------------------------
    v = valid
    q = lambda x: np.percentile(x[v], [25, 50, 75]) if v.any() else [np.nan] * 3
    print(f"    valid {v.sum()}/{N}   median R2 {np.nanmedian(r2):.3f}")
    print(f"    E1 (t0={args.t0:g}s)  {q(E1)[1]:9.1f} Pa  [IQR {q(E1)[0]:.1f}-{q(E1)[2]:.1f}]")
    for tr in args.tref:
        Ex = E1 * (tr / args.t0) ** (-al)
        print(f"    E({tr*1e3:g} ms)      {q(Ex)[1]:9.1f} Pa  [IQR {q(Ex)[0]:.1f}-{q(Ex)[2]:.1f}]")
    print(f"    alpha          {q(al)[1]:9.3f}     [IQR {q(al)[0]:.3f}-{q(al)[2]:.3f}]")
    for suf, lab_ in (("_labcp", "lab contact point"), ("_sq", "squeeze film")):
        if f"alpha{suf}" in out:
            a_ = out[f"alpha{suf}"]; e_ = out[f"E_at_{args.tref[0]*1e3:g}ms{suf}_Pa"]
            print(f"    variant {lab_:18s} alpha {np.nanmedian(a_):.3f}   E({args.tref[0]*1e3:g} ms) {np.nanmedian(e_):.1f} Pa")
    if "E_elastic_Pa" in out:
        print(f"    variant elastic (Hertz/Sneddon) E {np.nanmedian(out['E_elastic_Pa']):.1f} Pa")
    if "E_depth_ratio" in out:
        print(f"    depth check E(40 %)/E(100 %) median {np.nanmedian(out['E_depth_ratio']):.2f} (1 = consistent contact point)")
    if args.clusters >= 2:
        print("    clusters: " + ", ".join(f"C{j}: n={np.sum(lab==j)}, "
              f"E1={np.median(E1[(lab==j)&v]) if np.any((lab==j)&v) else np.nan:.1f} Pa"
              for j in range(args.clusters)))
    if args.png:
        try:
            quicklook(out, out_dir / f"{name}.png", args, name)
        except Exception as e:
            print(f"    [warn] PNG failed: {e!r}")
    print(f"    saved {out_file}  ({time.time()-t_start:.0f} s)")


def quicklook(o, path, args, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, ListedColormap
    except ImportError:
        print("    [warn] matplotlib missing, no PNG")
        return
    ny, nx = o["topo_um"].shape
    ext = [0, nx * o["pixel_um"][0], 0, ny * o["pixel_um"][1]]
    m = lambda a: np.where(o["valid"], a, np.nan)
    ekey = f"E_at_{args.tref[0]*1e3:g}ms_Pa" if args.tref else "E1_Pa"
    panels = [("Topography (um)", o["topo_um"], "cividis", None),
              (f"{ekey.replace('_Pa','')} (Pa)", m(o[ekey]), "viridis", "log"),
              ("E1 (Pa)", m(o["E1_Pa"]), "viridis", "log"),
              ("alpha", m(o["alpha"]), "magma", None),
              ("slope (deg)", o["slope_deg"], "Greys", None),
              ("cluster", np.where(o["cluster"] >= 0, o["cluster"], np.nan), "tab10", "cat")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for a, (tt, d, cm, sc) in zip(ax.ravel(), panels):
        fin = d[np.isfinite(d)]
        if fin.size == 0:
            a.set_title(tt); continue
        if sc == "log" and np.all(fin > 0):
            kw = dict(norm=LogNorm(*np.percentile(fin, [2, 98])))
        elif sc == "cat":
            cm = ListedColormap(plt.get_cmap("tab10").colors[:max(args.clusters, 1)])
            kw = dict(vmin=-0.5, vmax=max(args.clusters, 1) - 0.5)
        else:
            kw = dict(vmin=np.percentile(fin, 1), vmax=np.percentile(fin, 99))
        h = a.imshow(d, origin="lower", extent=ext, cmap=cm, interpolation="nearest", **kw)
        a.set_title(tt); a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
        fig.colorbar(h, ax=a, shrink=0.85)
    fig.suptitle(f"{title}  |  Ting, {args.geometry}"
                 + (f" R={args.radius:g} um" if args.geometry != "cone" else f" {args.half_angle:g} deg")
                 + ("" if args.no_slope else ", slope-corrected"))
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
#  CLI
# =============================================================================
def n_workers(spec):
    n = os.cpu_count() or 1
    w = max(1, int(n * spec)) if spec <= 1 else max(1, min(int(spec), n))
    if sys.platform.startswith("win"):
        w = min(w, 61)           # ProcessPoolExecutor limit on Windows
    return w


def adh_prm(args):
    return {k: getattr(args, "adh_" + k) for k in ADH_DEFAULTS}


CONFIG_MAP = {
    ("afm", "invols_nm_per_V"): "invols", ("afm", "spring_constant_N_m"): "k",
    ("afm", "z_sensitivity_um_per_V"): "zsens", ("afm", "xy_sensitivity_um_per_V"): "xysens",
    ("probe", "geometry"): "geometry", ("probe", "radius_um"): "radius", ("probe", "half_angle_deg"): "half_angle",
    ("probe", "poisson_ratio"): "nu",
    ("acquisition", "sample_interval_s"): "dt", ("acquisition", "zigzag"): "zigzag",
    ("preprocessing", "drag_correction"): ("no_drag", lambda v: not v),
    ("preprocessing", "baseline_samples"): "baseline", ("preprocessing", "block_average"): "block",
    ("preprocessing", "z_smoothing_ms"): "savgol_ms",
    ("fit", "t0_s"): "t0", ("fit", "tref_s"): "tref", ("fit", "slope_correction"): ("no_slope", lambda v: not v),
    ("fit", "slope_smooth_um"): "slope_smooth", ("fit", "variants"): "variants",
    ("quality", "r2_min"): "r2_min", ("quality", "slope_max_deg"): "slope_max",
    ("quality", "max_delta_over_R"): "max_delta_over_R",
    ("clustering", "n_clusters"): "clusters",
    ("adhesion", "enabled"): ("no_adhesion", lambda v: not v),
    **{("adhesion", k): "adh_" + k for k in ADH_DEFAULTS},
    ("run", "cores"): "cores", ("run", "name_depth"): "name_depth", ("run", "png"): "png",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        description="Batch Ting viscoelastic analysis of AFM force-volume data (LVM + config.txt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="dataset folders, root folders (searched recursively) and/or .txt list files")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--config", default=None, help="parameter file (TOML, e.g. afm_pipeline.toml); command line wins")
    ap.add_argument("--cores", type=float, default=0.8, help="fraction of CPU cores (<=1) or number of workers (>1)")
    g = ap.add_argument_group("calibration")
    g.add_argument("--invols", type=float, default=200.0, help="cantilever InvOLS [nm/V]")
    g.add_argument("--k", type=float, default=0.09, help="spring constant [N/m]")
    g.add_argument("--zsens", type=float, default=25.0, help="Z sensor sensitivity [um/V]")
    g.add_argument("--xysens", type=float, default=25.0, help="X/Y scanner sensitivity [um/V]")
    g.add_argument("--dt", type=float, default=None, help="sample interval [s]; default = loop_time [us] from config")
    g.add_argument("--zigzag", type=lambda v: str(v).lower() in ("1", "true", "yes"), default=None,
                   help="every 2nd scan line reversed (true/false); default = Zig from config.txt")
    g = ap.add_argument_group("contact model")
    g.add_argument("--geometry", choices=["sphere", "paraboloid", "cone"], default="sphere")
    g.add_argument("--radius", type=float, default=3.0, help="probe radius [um] (sphere/paraboloid)")
    g.add_argument("--half-angle", type=float, default=17.5, help="cone half angle [deg]")
    g.add_argument("--nu", type=float, default=0.5, help="Poisson ratio")
    g.add_argument("--t0", type=float, default=1.0, help="reference time of E1 [s]")
    g.add_argument("--variants", nargs="*", default=list(VARIANTS_ALL),
                   help="model variants in addition to the main fit: labcp squeeze elastic depth (none: --variants)")
    g.add_argument("--tref", type=float, nargs="*", default=[0.001, 0.002, 0.01],
                   help="extra times [s] at which E(t) maps are stored")
    g = ap.add_argument_group("processing")
    g.add_argument("--no-slope", action="store_true", help="skip surface-slope correction")
    g.add_argument("--slope-smooth", type=float, default=None, help="topography smoothing for slope [um]; default max(pixel, R/3)")
    g.add_argument("--slope-max", type=float, default=60.0, help="pixels steeper than this are invalid [deg]")
    g.add_argument("--no-drag", action="store_true", help="skip hydrodynamic drag correction")
    g.add_argument("--block", type=int, default=None, help="block-average factor; default round((num_app+num_ret)/800)")
    g.add_argument("--baseline", type=int, default=None, help="baseline samples at start of approach; default min(600, num_app/3)")
    g.add_argument("--savgol-ms", type=float, default=2.7, help="Z smoothing window [ms]")
    g.add_argument("--r2-min", type=float, default=0.9)
    g.add_argument("--max-delta-over-R", type=float, default=2.0, help="invalid if delta_max > this * R")
    g.add_argument("--clusters", type=int, default=3, help="number of shape clusters (0 = off)")
    g.add_argument("--no-adhesion", action="store_true", help="skip the adhesion analysis of the retract curves")
    g = ap.add_argument_group("adhesion analysis (see [adhesion] in the config file)")
    for k_, v_ in ADH_DEFAULTS.items():
        g.add_argument("--adh-" + k_.replace("_", "-"), dest="adh_" + k_, type=type(v_), default=v_)
    g = ap.add_argument_group("output")
    g.add_argument("--png", action="store_true", help="also write a quick-look PNG per dataset")
    g.add_argument("--name-depth", type=int, default=3, help="path components used for the output name")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--keep-cache", action="store_true", help="keep raw-data cache (.npy) in OUT/_cache")
    apply_config(ap, argv, CONFIG_MAP)
    args = ap.parse_args(argv)
    if args.radius is None and args.geometry != "cone":
        sys.exit("probe radius needed (--radius or [probe] radius_um)")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = discover(args.inputs)
    if not datasets:
        sys.exit("no datasets found (need config.txt + ForceCurve/ForceCurve_*.lvm)")
    ncores = n_workers(args.cores)
    print(f"{len(datasets)} dataset(s), {ncores}/{os.cpu_count()} cores, output -> {out_dir}")
    failed = []
    for ds in datasets:
        try:
            process_dataset(ds, out_dir, args, ncores)
        except Exception as e:
            print(f"[error] {ds}: {e!r}")
            failed.append(str(ds))
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()          # Windows frozen executables
    main()
