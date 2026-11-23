#!/usr/bin/env python3
"""
Extract topography, E, alpha and delta maps from all pipeline .npz files in a folder.

    python extract_topo_E.py NPZ_DIR [-o OUT_DIR] [--e-key E1_Pa] [--no-mask] [-r]
                             [--r2-min 0.9] [--slope-max 60] [--max-delta-over-R 2]

Writes one .npy per dataset (shape ny x nx) into
    OUT_DIR/topo/<name>.npy      topography (um)
    OUT_DIR/E/<name>.npy         E (Pa)                      -- key set by --e-key
    OUT_DIR/alpha/<name>.npy     power-law exponent alpha
    OUT_DIR/delta/<name>.npy     max indentation delta (um)
    OUT_DIR/Fmax_nN/<name>.npy   peak (loading) force; OUT_DIR/adh_ratio/ = F_adh / F_max (load-normalised adhesion)
    OUT_DIR/adh_*/<name>.npy     adhesion maps (see below), computed from the retract curves stored
                                 in the .npz (no refitting); never masked (independent of the Ting fit)
    OUT_DIR/pixels.json          pixel size [px, py] (um) of every map (used by segment_cells.py and the viewer)
    OUT_DIR/<key>/<name>.npy     every model version present in the npz: E1_*_Pa, E_at_*ms*_Pa, alpha_*,
                                 E_elastic_Pa, E_depth_ratio, R2*, eta_sq_mPas, h0_sq_um, cp_shift_*_um
    OUT_DIR/maps_info.json       which npz key the E/ folder holds
    OUT_DIR/excluded/<name>.npy  uint8 bitmask, why a pixel is excluded (0 = valid):
                                   1 R2 < r2-min        2 slope >= slope-max
                                   4 delta > max*R      8 alpha at fit bound / E <= 0
                                  16 no fit (cannot be recovered)

Pixels are excluded with the thresholds above, re-evaluated from the values stored in the
.npz (no refitting). Defaults = the thresholds used in the pipeline run.
Without --no-mask, E/alpha/delta are NaN at excluded pixels (as before).
With    --no-mask, all fitted values are kept; use the excluded/ maps to filter later
(the time-lapse viewer then lets you switch excluded pixels on and off per reason).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

R2_LOW, STEEP, DEEP, BOUND, NOFIT = 1, 2, 4, 8, 16


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




def reasons(z, r2_min, slope_max, dR, E):
    p = json.loads(str(z["params_json"])) if "params_json" in z.files else {}
    r2_min = p.get("r2_min", 0.9) if r2_min is None else r2_min
    slope_max = p.get("slope_max_deg", 60.0) if slope_max is None else slope_max
    geom, R = p.get("geometry", "sphere"), p.get("radius", np.inf)
    a, d, r2 = z["alpha"], z["delta_max_um"], z["R2"]
    m = np.zeros(E.shape, np.uint8)
    fin = np.isfinite(E) & np.isfinite(a)
    with np.errstate(invalid="ignore"):
        m[fin & ~(r2 > r2_min)] |= R2_LOW
        if "slope_deg" in z.files and p.get("slope_correction", True):
            m[fin & (z["slope_deg"] >= slope_max)] |= STEEP
        if geom in ("sphere", "paraboloid"):
            m[fin & (d >= (2.0 if dR is None else dR) * R)] |= DEEP
        m[fin & ((E <= 0) | (a <= 1e-3) | (a >= 0.949))] |= BOUND
    m[~fin] = NOFIT
    return m, (r2_min, slope_max)


CONFIG_MAP = {
    ("export", "e_map"): "e_key", ("export", "no_mask"): "no_mask", ("export", "recompute_adhesion"): "recompute_adhesion",
    ("quality", "r2_min"): "r2_min", ("quality", "slope_max_deg"): "slope_max",
    ("quality", "max_delta_over_R"): "max_delta_over_R",
    ("adhesion", "enabled"): ("no_adhesion", lambda v: not v),
    **{("adhesion", k): "adh_" + k for k in ADH_DEFAULTS},
    ("run", "cores"): "cores",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_dir")
    ap.add_argument("--config", default=None, help="parameter file (TOML); command line wins")
    ap.add_argument("--recompute-adhesion", action="store_true",
                    help="recompute the adhesion maps from the stored curves with the [adhesion] parameters")
    for k_, v_ in ADH_DEFAULTS.items():
        ap.add_argument("--adh-" + k_.replace("_", "-"), dest="adh_" + k_, type=type(v_), default=v_,
                        help=argparse.SUPPRESS)
    ap.add_argument("-o", "--out", default=None, help="output folder (default: NPZ_DIR)")
    ap.add_argument("--e-key", default="E_at_1ms_Pa",
                    help="E map to export, e.g. E1_Pa, E_at_2ms_Pa, E_at_10ms_Pa, E1_vertical_Pa")
    ap.add_argument("--no-mask", action="store_true", help="keep values at excluded pixels")
    ap.add_argument("--r2-min", type=float, default=None, help="default: value used in the pipeline (0.9)")
    ap.add_argument("--slope-max", type=float, default=None, help="deg; default: pipeline value (60)")
    ap.add_argument("--max-delta-over-R", type=float, default=None, help="default 2")
    ap.add_argument("-r", "--recursive", action="store_true", help="also search subfolders")
    ap.add_argument("--no-adhesion", action="store_true", help="skip the adhesion maps")
    ap.add_argument("--cores", type=float, default=0.8, help="fraction of cores (<=1) or number (>1)")
    apply_config(ap, sys.argv[1:], CONFIG_MAP)
    args = ap.parse_args()

    src = Path(args.npz_dir).expanduser()
    out = Path(args.out).expanduser() if args.out else src
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.rglob("*.npz") if args.recursive else src.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {src}")
    n = os.cpu_count() or 1
    nw = max(1, int(n * args.cores)) if args.cores <= 1 else max(1, min(int(args.cores), n))
    if sys.platform.startswith("win"):
        nw = min(nw, 61)
    jobs = [(str(f), str(out), vars(args)) for f in files]
    n_ok, pixels = 0, {}
    with ProcessPoolExecutor(max_workers=min(nw, len(files))) as ex:
        for ok, msg, pix in ex.map(export_one, jobs):
            print(msg)
            n_ok += ok
            pixels.update(pix)
    (out / "maps_info.json").write_text(json.dumps({"E": args.e_key}))   # which npz key the E/ folder holds
    if pixels:      # pixel size per map, read by segment_cells.py and build_timelapse_viewer.py
        pf = out / "pixels.json"
        old = json.loads(pf.read_text()) if pf.is_file() else {}
        old.update(pixels)
        pf.write_text(json.dumps(old, indent=1))
    print(f"\n{n_ok}/{len(files)} exported -> {out}/{{topo,E,alpha,delta,excluded"
          + ("" if args.no_adhesion else ",adh_*") + "}"
          + ("  (all fitted values kept)" if args.no_mask else ""))


# every model version of E and alpha (and its quality / contact-point maps) is exported under its npz key
MODEL_KEYS = re.compile(r"^(Fmax_nN|E1_.*Pa|E_at_.*Pa|alpha_.+|E_elastic_Pa|E_depth_ratio|R2(_.+)?|eta_sq_mPas|h0_sq_um|"
                        r"cp_shift_.+_um)$")
MODEL_MASKED = re.compile(r"^(E1_.*|E_at_.*|alpha_.+|E_elastic_Pa)$")     # masked like E unless --no-mask


def export_one(job):
    f, out, a = job
    f, out = Path(f), Path(out)
    try:
        with np.load(f) as z:
            ek = a["e_key"] if a["e_key"] in z.files else "E1_Pa"     # older results: no E_at_1ms_Pa
            maps = {"topo": ("topo_um", False), "E": (ek, True),
                    "alpha": ("alpha", True), "delta": ("delta_max_um", True)}
            miss = [k for k, _ in maps.values() if k not in z.files] + \
                   [k for k in ("R2", "alpha", "delta_max_um") if k not in z.files]
            if miss:
                return 0, f"[skip] {f.name}: missing {sorted(set(miss))}", {}
            E = np.asarray(z[ek], np.float32)
            bad, (r2m, slm) = reasons(z, a["r2_min"], a["slope_max"], a["max_delta_over_R"], E)
            (out / "excluded").mkdir(parents=True, exist_ok=True)
            np.save(out / "excluded" / f"{f.stem}.npy", bad)
            shape = np.shape(z["topo_um"])
            for key in z.files:
                if MODEL_KEYS.match(key) and np.ndim(z[key]) == 2 and np.shape(z[key]) == shape:
                    maps.setdefault(key, (key, bool(MODEL_MASKED.match(key))))
            for d, (key, masked) in maps.items():
                arr = np.asarray(z[key], np.float32)
                if masked and not a["no_mask"]:
                    arr = np.where(bad == 0, arr, np.nan).astype(np.float32)
                (out / d).mkdir(parents=True, exist_ok=True)
                np.save(out / d / f"{f.stem}.npy", arr)
            adh_msg = ""
            if not a["no_adhesion"]:
                prm = {k: a["adh_" + k] for k in ADH_DEFAULTS}
                if all(k in z.files for k in ADH_KEYS) and not a["recompute_adhesion"]:
                    A = {k: np.asarray(z[k], np.float32) for k in ADH_KEYS}
                    src_ = "stored"
                elif all(k in z.files for k in ("curves_F_nN", "curves_Z_um", "noise_nN", "curve_index_map")):
                    p = json.loads(str(z["params_json"])) if "params_json" in z.files else {}
                    R = p.get("radius") if p.get("geometry", "sphere") == "sphere" else None
                    A = adhesion_maps(z["curves_F_nN"], z["curves_Z_um"], p.get("k", 0.09), z["noise_nN"],
                                      z["curve_index_map"], R, prm)
                    src_ = "from curves"
                else:
                    A = None
                    adh_msg = "  | adhesion: no curves in npz"
                if A is not None:
                    for k, m in A.items():
                        (out / k).mkdir(parents=True, exist_ok=True)
                        np.save(out / k / f"{f.stem}.npy", m)
                    if "Fmax_nN" in z.files and np.shape(z["Fmax_nN"]) == np.shape(A["adh_force_nN"]):
                        # adhesion scales with the peak (loading) force; the ratio is the load-normalised adhesion
                        fm = np.asarray(z["Fmax_nN"], np.float32)
                        (out / "adh_ratio").mkdir(parents=True, exist_ok=True)
                        np.save(out / "adh_ratio" / f"{f.stem}.npy",
                                np.where(fm > 0, A["adh_force_nN"] / np.where(fm > 0, fm, 1), np.nan).astype(np.float32))
                    fa = A["adh_force_nN"]
                    adh_msg = (f"  | adhesion ({src_}): F {np.nanmedian(fa):.2f} nN, "
                               f"W {np.nanmedian(A['adh_work_fJ']):.2f} fJ, "
                               f"tethers {100*np.nanmean(A['adh_tether_nN'] > 0):.0f}%")
        cnt = lambda b: int(np.sum((bad & b) > 0))
        with np.load(f) as z:
            pix = {f.stem: [float(v) for v in z["pixel_um"]]} if "pixel_um" in z.files else {}
        return 1, (f"{f.name}: {bad.shape}  valid {100*np.mean(bad==0):.1f}%  | excluded: "
                   f"R2<{r2m:g} {cnt(R2_LOW)}, slope>={slm:g} {cnt(STEEP)}, deep {cnt(DEEP)}, "
                   f"bound {cnt(BOUND)}, no fit {cnt(NOFIT)}" + adh_msg), pix
    except Exception as e:
        return 0, f"[error] {f.name}: {e!r}", {}


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
