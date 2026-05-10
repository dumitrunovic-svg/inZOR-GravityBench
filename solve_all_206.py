"""
Gravity-Bench-v1 — independent reference solver for all 206 tasks.

Reproduces the PhD-level reference algorithms described in Appendix B of
Koblischke et al. 2025 (ICML 2025, arXiv:2501.18411).

No-leakage protocol: the solver only consumes data that an agent receives at
evaluation time:
  - `simulation_csv_content` (star positions x, y, z vs. time)
  - `scenario_name`            (used for task dispatch only)

It does NOT read `variation_name`, which would expose ground-truth masses,
the unit system, and OOD labels ("Modified Gravity", "Drag", "Unbound").

Task families:
  A) Orbital geometry (apoastron, periastron, SMA, eccentricity, period, ...)
  B) Kinematics (velocities, accelerations, angular velocities)
  C) Stellar masses (acceleration method: M_other = a_self * d^2 / G)
  D) Conservation laws (areal velocity, specific angular momentum)
  E) Boolean tasks (Kepler 3, virial theorem, is_bound) — detected from data
  F) Out-of-distribution: modified gravity (log-log fit on a vs r),
     linear drag (exponential fit on SMA(t))

Unit handling (SI / cgs / year-AU-Msun) is automatic: we try each unit set
and accept the one for which the estimated stellar mass falls in
(1e28, 1e34) kg.

Run: `python3 solve_all_206.py`
"""

from __future__ import annotations
import json, sys, time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_G = 6.674e-11
_M_SUN = 1.989e30
_OUT_DIR = Path(__file__).resolve().parent / "results"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utilitare de baza
# ══════════════════════════════════════════════════════════════════════════════

_AU = 1.495978707e11   # m
_YR = 3.15576e7        # s
_CM = 1e-2             # m per cm


def _detect_units_from_data(df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Detecteaza unitatile pur din date prin verificarea stelaritatii masei
    estimate prin acceleration method (M = a*r^2/G).

    Strategie: incercam fiecare set de unitati (SI / yrAUMsun / cgs) si alegem
    pe cel care da o masa stelara plauzibila (10^28..10^33 kg).
    """
    pos_cols = [c for c in df.columns if c != "time"]
    candidates = [
        ("SI",       1.0,  1.0),
        ("yrAUMsun", _AU,  _YR),
        ("cgs",      _CM,  1.0),
    ]
    best_label = "SI"; best_fpos = 1.0; best_ftime = 1.0
    best_score = -1.0

    for label, fpos, ftime in candidates:
        # converteste o copie temporara
        t = df["time"].values.astype(float) * ftime
        x1 = df["star1_x"].values.astype(float) * fpos
        y1 = df["star1_y"].values.astype(float) * fpos
        x2 = df["star2_x"].values.astype(float) * fpos
        y2 = df["star2_y"].values.astype(float) * fpos
        if len(t) < 50:
            continue
        dt = float(np.median(np.diff(t[:200])))
        if dt <= 0:
            continue
        rr = np.hypot(x2 - x1, y2 - y1)
        ax2 = (x2[2:] - 2 * x2[1:-1] + x2[:-2]) / dt**2
        ay2 = (y2[2:] - 2 * y2[1:-1] + y2[:-2]) / dt**2
        a2 = np.hypot(ax2, ay2)
        rr_mid = rr[1:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            M_est = float(np.median(a2 * rr_mid**2 / _G))
        # scor: 1 daca M e stelara (10^28-10^33 kg), 0 altfel
        if 1e27 < M_est < 1e34:
            # mai mare = mai aproape de 10^30 (Msun)
            score = 1.0 - abs(np.log10(M_est) - 30.0) / 4.0
            if score > best_score:
                best_score = score
                best_label, best_fpos, best_ftime = label, fpos, ftime

    return best_label, best_fpos, best_ftime


def _load_df(row) -> pd.DataFrame:
    """Incarca CSV si normalizeaza la SI (m, s) bazat pe detectie din date."""
    df = pd.read_csv(StringIO(row["simulation_csv_content"]))
    pos_cols = [c for c in df.columns if c != "time"]

    label, fpos, ftime = _detect_units_from_data(df)
    if fpos != 1.0:
        for c in pos_cols:
            df[c] = df[c] * fpos
    if ftime != 1.0:
        df["time"] = df["time"] * ftime
    return df


def _rel(df: pd.DataFrame):
    """Coordonate relative r2 - r1."""
    rx = (df["star2_x"] - df["star1_x"]).values
    ry = (df["star2_y"] - df["star1_y"]).values
    r  = np.hypot(rx, ry)
    return rx, ry, r


def _velocities(pos: np.ndarray, dt: float) -> np.ndarray:
    """Viteza prin diferente centrale, capete prin diferente unilaterale."""
    v = np.empty_like(pos)
    v[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    v[0]    = (pos[1]  - pos[0])   / dt
    v[-1]   = (pos[-1] - pos[-2])  / dt
    return v


def _accelerations(pos: np.ndarray, dt: float) -> np.ndarray:
    """Acceleratie prin diferente de ordinul 2."""
    a = np.empty_like(pos)
    a[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) / dt**2
    a[0]    = a[1]
    a[-1]   = a[-2]
    return a


def _dt(df: pd.DataFrame) -> float:
    diffs = np.diff(df["time"].values[:200])
    return float(np.median(diffs))


def _detrend(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Elimina tendinta liniara (miscarea COM)."""
    p = np.polyfit(t, arr, 1)
    return arr - np.polyval(p, t)


def _detect_period(r: np.ndarray, t: np.ndarray) -> float | None:
    """Detecteaza perioada orbitala din minimele separarii relative."""
    neg_r = -r
    # cauta minime proeminente (periastron)
    prom = (r.max() - r.min()) * 0.2
    peaks, _ = find_peaks(neg_r, prominence=prom, distance=max(5, len(r) // 100))
    if len(peaks) >= 2:
        return float(np.median(np.diff(t[peaks])))
    # fallback: maxime (apoastron)
    peaks2, _ = find_peaks(r, prominence=prom, distance=max(5, len(r) // 100))
    if len(peaks2) >= 2:
        return float(np.median(np.diff(t[peaks2])))
    return None


def _estimate_masses(df: pd.DataFrame):
    """
    Estimeaza m1, m2 din datele orbitale folosind metoda PhD-level din paper:
    M_other = a_self * d^2 / G  (din F = G*m1*m2/d^2 = m_self * a_self).
    Functioneaza pentru orice sistem (bound, unbound, drag, modified gravity).
    Returneaza (m1, m2, M_tot, q=m1/m2).
    """
    dt = _dt(df)
    rx, ry, r = _rel(df)
    # Acceleratii din diferente finite ordinul 2
    ax1 = _accelerations(df["star1_x"].values.astype(float), dt)
    ay1 = _accelerations(df["star1_y"].values.astype(float), dt)
    ax2 = _accelerations(df["star2_x"].values.astype(float), dt)
    ay2 = _accelerations(df["star2_y"].values.astype(float), dt)
    a1 = np.hypot(ax1, ay1)
    a2 = np.hypot(ax2, ay2)

    # Filtreaza valori bune (eliminam capete unde diferentele finite sunt corupte)
    n = len(r)
    pad = max(5, n // 100)
    sl = slice(pad, n - pad)
    r_s, a1_s, a2_s = r[sl], a1[sl], a2[sl]

    # M_other = a_self * d^2 / G; folosim mediana pt rezistenta la outlieri
    # Pentru gravitatie inversa-patratica standard: F = GM_otherM_self/d^2 → a_self = GM_other/d^2
    M2 = float(np.median(a1_s * r_s**2 / _G))   # star1 trasa de star2 (m2)
    M1 = float(np.median(a2_s * r_s**2 / _G))   # star2 trasa de star1 (m1)
    M_tot = M1 + M2
    q = M1 / max(M2, 1e-30)
    return float(M1), float(M2), float(M_tot), float(q)


def _com_pos(df: pd.DataFrame, m1: float, m2: float):
    """Pozitia centru de masa (x, y) in timp."""
    M = m1 + m2
    xc = (m1 * df["star1_x"].values + m2 * df["star2_x"].values) / M
    yc = (m1 * df["star1_y"].values + m2 * df["star2_y"].values) / M
    return xc.astype(float), yc.astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# Solvere per scenariu
# ══════════════════════════════════════════════════════════════════════════════

def _solve_apoastron(df):
    _, _, r = _rel(df)
    return float(r.max())


def _solve_periastron(df):
    _, _, r = _rel(df)
    return float(r.min())


def _solve_eccentricity(df):
    _, _, r = _rel(df)
    apo = r.max(); peri = r.min()
    return float((apo - peri) / (apo + peri))


def _solve_semi_major_axis(df):
    _, _, r = _rel(df)
    return float((r.max() + r.min()) / 2.0)


def _solve_semi_minor_axis(df):
    a = _solve_semi_major_axis(df)
    e = _solve_eccentricity(df)
    return float(a * np.sqrt(max(0.0, 1.0 - e**2)))


def _solve_period(df):
    _, _, r = _rel(df)
    t = df["time"].values.astype(float)
    T = _detect_period(r, t)
    if T is None:
        T = float(t[-1] - t[0])
    return float(T)


def _detect_modified_gravity(df) -> bool:
    """
    Detecteaza fizica modificata: |slope(log a, log r)| - 2 > 0.005.
    In fizica standard slope = -2 exact (machine precision).
    """
    alpha = _solve_modified_gravity_signed(df)
    return abs(alpha) > 0.005


def _detect_drag(df) -> bool:
    """
    Detecteaza drag: SMA(t) scade monoton, slope(SMA vs t) negativ semnificativ.
    """
    _, _, r = _rel(df)
    t = df["time"].values.astype(float)
    prom = (r.max() - r.min()) * 0.05
    dist = max(5, len(r) // 200)
    apo_idx, _  = find_peaks(r, prominence=prom, distance=dist)
    peri_idx, _ = find_peaks(-r, prominence=prom, distance=dist)
    n_orb = min(len(apo_idx), len(peri_idx))
    if n_orb < 4:
        return False
    apo_idx = apo_idx[:n_orb]; peri_idx = peri_idx[:n_orb]
    sma = (r[apo_idx] + r[peri_idx]) / 2.0
    t_orbit = (t[apo_idx] + t[peri_idx]) / 2.0
    # decay > 1% dupa toate orbitele observate
    decay = (sma[0] - sma[-1]) / sma[0]
    return bool(decay > 0.01)


def _solve_kepler_3rd_law(df) -> bool:
    """
    Kepler 3 valid daca fizica e Newton standard (no modified gravity, no drag).
    """
    if _detect_modified_gravity(df):
        return False
    if _detect_drag(df):
        return False
    return True


def _solve_virial_theorem(df) -> bool:
    """
    Virial 2K + U = 0 valid in sisteme bound conservative.
    Eshueaza pentru drag (energie scade) si unbound (E > 0).
    """
    if _detect_drag(df):
        return False
    return _solve_is_bound(df)


def _solve_is_bound(df) -> bool:
    """
    Sistem legat: K + U < 0 (energia negativa).
    Verificam cu masele estimate din date.
    """
    m1, m2, _, _ = _estimate_masses(df)
    _, _, v1 = _star_velocities(df, 1)
    _, _, v2 = _star_velocities(df, 2)
    _, _, r  = _rel(df)
    n = len(r)
    pad = max(5, n // 100)
    sl  = slice(pad, n - pad)
    K = 0.5 * m1 * v1[sl]**2 + 0.5 * m2 * v2[sl]**2
    U = -_G * m1 * m2 / r[sl]
    E = K + U
    return bool(np.median(E) < 0)


def _star_orbit_amplitude(df, star: int, component: str, m1: float, m2: float):
    """Semi-axa orbitei stea star in jurul COM (metoda: amplitudine maxima)."""
    xc, yc = _com_pos(df, m1, m2)
    t = df["time"].values.astype(float)
    if star == 1:
        dx = df["star1_x"].values.astype(float) - xc
        dy = df["star1_y"].values.astype(float) - yc
    else:
        dx = df["star2_x"].values.astype(float) - xc
        dy = df["star2_y"].values.astype(float) - yc
    r_star = np.hypot(dx, dy)
    return float(r_star.max())


def _solve_semi_major_axis_star(df, star: int):
    m1, m2, _, _ = _estimate_masses(df)
    xc, yc = _com_pos(df, m1, m2)
    t = df["time"].values.astype(float)
    col_x = f"star{star}_x"; col_y = f"star{star}_y"
    dx = df[col_x].values.astype(float) - xc
    dy = df[col_y].values.astype(float) - yc
    r_star = np.hypot(dx, dy)
    apo = r_star.max(); peri = r_star.min()
    return float((apo + peri) / 2.0)


def _solve_semi_minor_axis_star(df, star: int):
    m1, m2, _, _ = _estimate_masses(df)
    e = _solve_eccentricity(df)
    a = _solve_semi_major_axis_star(df, star)
    return float(a * np.sqrt(max(0.0, 1.0 - e**2)))


def _solve_orbital_area_star(df, star: int):
    a = _solve_semi_major_axis_star(df, star)
    b = _solve_semi_minor_axis_star(df, star)
    return float(np.pi * a * b)


def _star_velocities(df, star: int):
    dt = _dt(df)
    x = df[f"star{star}_x"].values.astype(float)
    y = df[f"star{star}_y"].values.astype(float)
    vx = _velocities(x, dt)
    vy = _velocities(y, dt)
    v  = np.hypot(vx, vy)
    return vx, vy, v


def _solve_max_velocity_star(df, star: int):
    _, _, v = _star_velocities(df, star)
    return float(v.max())


def _solve_min_velocity_star(df, star: int):
    _, _, v = _star_velocities(df, star)
    return float(v[v > 0].min() if (v > 0).any() else v.min())


def _solve_max_acceleration_star(df, star: int):
    dt = _dt(df)
    x = df[f"star{star}_x"].values.astype(float)
    y = df[f"star{star}_y"].values.astype(float)
    ax = _accelerations(x, dt)
    ay = _accelerations(y, dt)
    a  = np.hypot(ax, ay)
    return float(a.max())


def _solve_min_acceleration_star(df, star: int):
    dt = _dt(df)
    x = df[f"star{star}_x"].values.astype(float)
    y = df[f"star{star}_y"].values.astype(float)
    ax = _accelerations(x, dt)
    ay = _accelerations(y, dt)
    a  = np.hypot(ax, ay)
    return float(a[a > 0].min() if (a > 0).any() else a.min())


def _solve_max_angular_velocity_star(df, star: int):
    """omega = |r × v| / r^2 pentru fiecare stea fata de COM."""
    m1, m2, _, _ = _estimate_masses(df)
    dt = _dt(df)
    xc, yc = _com_pos(df, m1, m2)
    x = df[f"star{star}_x"].values.astype(float) - xc
    y = df[f"star{star}_y"].values.astype(float) - yc
    vx = _velocities(x, dt)
    vy = _velocities(y, dt)
    r2 = x**2 + y**2
    r2 = np.maximum(r2, 1e10)
    omega = np.abs(x * vy - y * vx) / r2
    return float(omega.max())


def _solve_min_angular_velocity_star(df, star: int):
    m1, m2, _, _ = _estimate_masses(df)
    dt = _dt(df)
    xc, yc = _com_pos(df, m1, m2)
    x = df[f"star{star}_x"].values.astype(float) - xc
    y = df[f"star{star}_y"].values.astype(float) - yc
    vx = _velocities(x, dt)
    vy = _velocities(y, dt)
    r2 = x**2 + y**2
    r2 = np.maximum(r2, 1e10)
    omega = np.abs(x * vy - y * vx) / r2
    return float(omega[omega > 0].min() if (omega > 0).any() else omega.min())


def _solve_max_momentum_star(df, star: int):
    m1, m2, _, _ = _estimate_masses(df)
    m = m1 if star == 1 else m2
    _, _, v = _star_velocities(df, star)
    return float(m * v.max())


def _solve_min_momentum_star(df, star: int):
    m1, m2, _, _ = _estimate_masses(df)
    m = m1 if star == 1 else m2
    _, _, v = _star_velocities(df, star)
    return float(m * (v[v > 0].min() if (v > 0).any() else v.min()))


def _solve_mass_star(df, star: int):
    m1, m2, _, _ = _estimate_masses(df)
    return float(m1 if star == 1 else m2)


def _solve_total_mass(df):
    _, _, M, _ = _estimate_masses(df)
    return float(M)


def _solve_mass_ratio(df) -> float:
    """Raportul de masa m1/m2 (poate fi < 1 daca m1 < m2)."""
    _, _, _, q = _estimate_masses(df)
    return float(q)  # q = sigma2/sigma1 = m1/m2


def _solve_mass_largest_star(df):
    m1, m2, _, _ = _estimate_masses(df)
    return float(max(m1, m2))


def _solve_reduced_mass(df):
    m1, m2, M, _ = _estimate_masses(df)
    return float(m1 * m2 / M)


def _solve_multiply_mass_period(df):
    """
    Prompt: "factor X by which central mass (star1) should be multiplied
    for the orbital period to be 21 days."
    Din Kepler T ∝ 1/sqrt(M): X = (T_curent / T_tinta)^2, T_tinta = 21 zile.
    """
    T_current = _solve_period(df)
    T_target   = 21.0 * 86400.0   # 21 zile in secunde
    return float((T_current / T_target) ** 2)


def _angular_momentum_rel(df: pd.DataFrame):
    """Moment cinetic specific al orbitei relative (m²/s)."""
    dt = _dt(df)
    rx, ry, _ = _rel(df)
    vrx = _velocities(rx, dt)
    vry = _velocities(ry, dt)
    L = rx * vry - ry * vrx  # specific angular momentum z-component
    return L


def _solve_area_swept(df, at: str):
    """
    Rata de arie matuita la apoastron / periastron.
    dA/dt = |L_specific| / 2 unde L_specific = r × v (vectorul orbital).
    """
    _, _, r = _rel(df)
    L = _angular_momentum_rel(df)
    dA_dt = np.abs(L) / 2.0

    if at == "apo":
        idx = int(np.argmax(r))
    else:
        idx = int(np.argmin(r))

    # Medie in jurul punctului
    w = max(1, len(r) // 200)
    lo = max(0, idx - w); hi = min(len(r), idx + w)
    return float(np.mean(np.abs(dA_dt[lo:hi])))


def _solve_specific_angular_momentum(df):
    """
    h = |r_rel × v_rel| (m²/s), constant pe orbita.
    """
    L = _angular_momentum_rel(df)
    return float(np.median(np.abs(L)))


def _solve_avg_distance_COM_star(df, star: int):
    """Distanta medie a stelei fata de COM pe o singura orbita."""
    m1, m2, _, _ = _estimate_masses(df)
    xc, yc = _com_pos(df, m1, m2)
    col_x = f"star{star}_x"; col_y = f"star{star}_y"
    dx = df[col_x].values.astype(float) - xc
    dy = df[col_y].values.astype(float) - yc
    r  = np.hypot(dx, dy)
    t  = df["time"].values.astype(float)
    T  = _detect_period(r, t)
    if T is None:
        return float(np.mean(r))
    # ia doar o singura orbita (primele T secunde)
    mask = (t - t[0]) <= T
    return float(np.mean(r[mask]))


def _solve_time_fraction_accel_below_mean(df):
    dt = _dt(df)
    x1 = df["star1_x"].values.astype(float)
    y1 = df["star1_y"].values.astype(float)
    ax = _accelerations(x1, dt)
    ay = _accelerations(y1, dt)
    a  = np.hypot(ax, ay)
    return float(np.mean(a < np.mean(a)))


def _solve_travel_time_pct(df, pct: float):
    """
    Timp pentru a parcurge pct% din lungimea orbitala relative, masurat
    INCEPAND DE LA PERIAPSIS (prima trecere prin punctul de minima separare).
    """
    rx, ry, r = _rel(df)
    t = df["time"].values.astype(float)
    dx = np.diff(rx); dy = np.diff(ry)
    ds = np.hypot(dx, dy)
    s  = np.concatenate([[0.0], np.cumsum(ds)])

    T = _detect_period(r, t)
    if T is None:
        s_total = s[-1]
    else:
        idx_T = int(np.searchsorted(t - t[0], T))
        s_total = float(s[min(idx_T, len(s) - 1)])

    # Gaseste prima periapsis (minim de separare)
    prom = (r.max() - r.min()) * 0.2
    dist = max(5, len(r) // 100)
    peri_peaks, _ = find_peaks(-r, prominence=prom, distance=dist)
    if len(peri_peaks) == 0:
        peri_idx = int(np.argmin(r))
    else:
        peri_idx = int(peri_peaks[0])

    # Lungimea de la periapsis
    s_from_peri = s - s[peri_idx]
    target = pct * s_total
    idx = int(np.searchsorted(s_from_peri, target))
    idx = min(idx, len(t) - 1)
    return float(t[idx] - t[peri_idx])


def _solve_KU(df):
    """
    Energia mecanica totala K + U.
    Foloseste masele estimate pur din date (acceleration method).
    """
    m1, m2, _, _ = _estimate_masses(df)
    _, _, v1 = _star_velocities(df, 1)
    _, _, v2 = _star_velocities(df, 2)
    _, _, r  = _rel(df)
    K = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    U = -_G * m1 * m2 / r
    E = K + U
    return float(np.median(E))


def _solve_roche_lobe_radius(df):
    """
    Raza lobului Roche al lui star1 (Eggleton 1983).
    q = m1/m2 (raportul de masa cu star1 ca primar).
    """
    m1, m2, _, _ = _estimate_masses(df)
    a  = _solve_semi_major_axis(df)
    q  = m1 / m2   # raport m1/m2 (nu min/max)
    q23 = q ** (2.0 / 3.0)
    R_L = a * 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q ** (1.0 / 3.0)))
    return float(R_L)


# ── OOD: modified_gravity_power_law si linear_drag ────────────────────────────
# Metodele PhD-level din paper Appendix B, fara variation_name (no leakage).

def _solve_modified_gravity_alpha(df) -> float:
    """
    PhD-level (Appendix B): log(a) vs log(r) trebuie sa aiba slope -(2+α).
    F ∝ r^(-2-α) → a ∝ r^(-2-α) → log(a) = -(2+α)*log(r) + const.
    Folosim acceleratia totala a uneia din stele si separatia.
    """
    dt = _dt(df)
    _, _, r = _rel(df)
    ax2 = _accelerations(df["star2_x"].values.astype(float), dt)
    ay2 = _accelerations(df["star2_y"].values.astype(float), dt)
    a2  = np.hypot(ax2, ay2)
    n = len(r)
    pad = max(5, n // 100)
    sl = slice(pad, n - pad)
    r_s, a_s = r[sl], a2[sl]
    # filtru: r > median si a > 0 (eliminam outlieri MAD)
    med_r = np.median(r_s)
    med_a = np.median(a_s)
    mask = (r_s > med_r) & (a_s > 0.1 * med_a) & (a_s < 10.0 * med_a)
    if mask.sum() < 10:
        mask = (r_s > 0) & (a_s > 0)
    log_r = np.log(r_s[mask])
    log_a = np.log(a_s[mask])
    slope, _ = np.polyfit(log_r, log_a, 1)
    alpha = abs(slope) - 2.0
    return float(alpha)


def _solve_modified_gravity_signed(df) -> float:
    """
    Adaugam semn la α: pozitiv (slope < -2) sau negativ (slope > -2).
    Folosim acceleratia ca semn al deviatiei de la r^-2.
    """
    dt = _dt(df)
    _, _, r = _rel(df)
    ax2 = _accelerations(df["star2_x"].values.astype(float), dt)
    ay2 = _accelerations(df["star2_y"].values.astype(float), dt)
    a2  = np.hypot(ax2, ay2)
    n = len(r)
    pad = max(5, n // 100)
    sl = slice(pad, n - pad)
    r_s, a_s = r[sl], a2[sl]
    med_r = np.median(r_s); med_a = np.median(a_s)
    mask = (r_s > med_r) & (a_s > 0.1 * med_a) & (a_s < 10.0 * med_a)
    if mask.sum() < 10:
        mask = (r_s > 0) & (a_s > 0)
    log_r = np.log(r_s[mask])
    log_a = np.log(a_s[mask])
    slope, _ = np.polyfit(log_r, log_a, 1)
    # slope = -(2+α), deci α = -slope - 2
    return float(-slope - 2.0)


def _solve_linear_drag_tau(df) -> float:
    """
    PhD-level (Appendix B): SMA(t) decade ca a_0 * exp(-2t/τ).
    Detectam apoastron + periastron in fiecare orbita si fit exponential.
    """
    from scipy.optimize import curve_fit
    rx, ry, r = _rel(df)
    t = df["time"].values.astype(float)

    prom = (r.max() - r.min()) * 0.05
    dist = max(5, len(r) // 200)
    apo_idx, _  = find_peaks(r, prominence=prom, distance=dist)
    peri_idx, _ = find_peaks(-r, prominence=prom, distance=dist)

    # match peaks: pereche fiecare apo cu cel mai apropiat peri
    n_orb = min(len(apo_idx), len(peri_idx))
    if n_orb < 3:
        # fallback: presupunem orbita aproape circulara, fit r(t) direct
        def model_r(tt, r0, tau):
            return r0 * np.exp(-tt / tau)
        try:
            p0 = [float(np.mean(r)), 1e9]
            popt, _ = curve_fit(model_r, t, r, p0=p0, maxfev=2000)
            return float(popt[1])
        except Exception:
            return 1e9

    apo_idx  = apo_idx[:n_orb]
    peri_idx = peri_idx[:n_orb]
    sma     = (r[apo_idx] + r[peri_idx]) / 2.0
    t_orbit = (t[apo_idx] + t[peri_idx]) / 2.0

    def model(tt, a0, tau):
        return a0 * np.exp(-2.0 * tt / tau)

    try:
        p0 = [float(sma[0]), 1e9]
        popt, _ = curve_fit(model, t_orbit, sma, p0=p0, maxfev=5000)
        return float(popt[1])
    except Exception:
        return float(np.median(sma))


# ══════════════════════════════════════════════════════════════════════════════
# Router principal
# ══════════════════════════════════════════════════════════════════════════════

def solve_task(row, _unused: dict | None = None) -> float | str | None:
    """Rezolva un task si returneaza valoarea estimata."""
    sc = row["scenario_name"]
    df = _load_df(row)

    if sc == "apoastron":
        return _solve_apoastron(df)
    if sc == "periastron":
        return _solve_periastron(df)
    if sc == "eccentricity":
        return _solve_eccentricity(df)
    if sc == "semi_major_axis":
        return _solve_semi_major_axis(df)
    if sc == "semi_minor_axis":
        return _solve_semi_minor_axis(df)
    if sc == "period":
        return _solve_period(df)
    if sc == "kepler_3rd_law":
        return _solve_kepler_3rd_law(df)
    if sc == "virial_theorem":
        return _solve_virial_theorem(df)
    if sc == "is_bound":
        return _solve_is_bound(df)

    if sc == "semi_major_axis_star1":
        return _solve_semi_major_axis_star(df, 1)
    if sc == "semi_major_axis_star2":
        return _solve_semi_major_axis_star(df, 2)
    if sc == "semi_minor_axis_star1":
        return _solve_semi_minor_axis_star(df, 1)
    if sc == "semi_minor_axis_star2":
        return _solve_semi_minor_axis_star(df, 2)
    if sc == "orbital_area_star1":
        return _solve_orbital_area_star(df, 1)
    if sc == "orbital_area_star2":
        return _solve_orbital_area_star(df, 2)

    if sc == "max_velocity_star1":
        return _solve_max_velocity_star(df, 1)
    if sc == "max_velocity_star2":
        return _solve_max_velocity_star(df, 2)
    if sc == "min_velocity_star1":
        return _solve_min_velocity_star(df, 1)
    if sc == "min_velocity_star2":
        return _solve_min_velocity_star(df, 2)

    if sc == "max_acceleration_star1":
        return _solve_max_acceleration_star(df, 1)
    if sc == "max_acceleration_star2":
        return _solve_max_acceleration_star(df, 2)
    if sc == "min_acceleration_star1":
        return _solve_min_acceleration_star(df, 1)
    if sc == "min_acceleration_star2":
        return _solve_min_acceleration_star(df, 2)

    if sc == "max_angular_velocity_star1":
        return _solve_max_angular_velocity_star(df, 1)
    if sc == "max_angular_velocity_star2":
        return _solve_max_angular_velocity_star(df, 2)
    if sc == "min_angular_velocity_star1":
        return _solve_min_angular_velocity_star(df, 1)
    if sc == "min_angular_velocity_star2":
        return _solve_min_angular_velocity_star(df, 2)

    if sc == "max_momentum_star1":
        return _solve_max_momentum_star(df, 1)
    if sc == "max_momentum_star2":
        return _solve_max_momentum_star(df, 2)
    if sc == "min_momentum_star1":
        return _solve_min_momentum_star(df, 1)
    if sc == "min_momentum_star2":
        return _solve_min_momentum_star(df, 2)

    if sc == "mass_star1":
        return _solve_mass_star(df, 1)
    if sc == "mass_star2":
        return _solve_mass_star(df, 2)
    if sc == "total_mass":
        return _solve_total_mass(df)
    if sc == "mass_ratio":
        return _solve_mass_ratio(df)
    if sc == "mass_largest_star":
        return _solve_mass_largest_star(df)
    if sc == "reduced_mass":
        return _solve_reduced_mass(df)

    if sc == "area_swept_over_time_apo":
        return _solve_area_swept(df, "apo")
    if sc == "area_swept_over_time_peri":
        return _solve_area_swept(df, "peri")
    if sc == "specific_angular_momentum":
        return _solve_specific_angular_momentum(df)
    if sc == "avg_distance_COM_star1":
        return _solve_avg_distance_COM_star(df, 1)
    if sc == "avg_distance_COM_star2":
        return _solve_avg_distance_COM_star(df, 2)
    if sc == "time_fraction_acceleraton_below_mean":
        return _solve_time_fraction_accel_below_mean(df)
    if sc == "travel_time_orbital_20per_path":
        return _solve_travel_time_pct(df, 0.20)
    if sc == "travel_time_orbital_70per_path":
        return _solve_travel_time_pct(df, 0.70)
    if sc == "K+U":
        return _solve_KU(df)
    if sc == "roche_lobe_radius":
        return _solve_roche_lobe_radius(df)
    if sc == "multiply_mass_period":
        return _solve_multiply_mass_period(df)

    if sc == "modified_gravity_power_law":
        return _solve_modified_gravity_signed(df)
    if sc == "linear_drag":
        return _solve_linear_drag_tau(df)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Evaluare eroare
# ══════════════════════════════════════════════════════════════════════════════

def _rel_error(pred, true_val) -> float | None:
    try:
        p = float(pred); t = float(true_val)
        if abs(t) < 1e-30:
            return abs(p - t)
        return abs(p - t) / abs(t) * 100.0
    except Exception:
        return None


def _bool_correct(pred, true_val) -> bool:
    try:
        ta_str = str(true_val).strip().lower()
        ta = ta_str == "true"
        if isinstance(pred, bool):
            return pred == ta
        return str(pred).strip().lower() == ta_str
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import datasets as _ds
    d = _ds.load_dataset(
        "GravityBench/GravityBench",
        cache_dir="/tmp/gravitybench_cache"
    )["test"]
    rows = list(d)
    print(f"\nGravityBench — {len(rows)} task-uri\n")

    results = []
    pass_full = 0; pass_budget = 0; total = 0

    for i, row in enumerate(rows):
        sc  = row["scenario_name"]
        var = row.get("variation_name", "")[:40]
        ta  = row["true_answer"]
        ft  = float(row.get("full_obs_threshold_percent", 5.0))
        bt  = float(row.get("budget_obs_threshold_percent", ft))

        t0 = time.time()
        try:
            pred = solve_task(row)
        except Exception as ex:
            pred = None
            print(f"  [{i:3d}] {sc:<42} ERROR: {ex}", flush=True)

        elapsed = time.time() - t0

        # Evalueaza
        is_bool = str(ta).strip().lower() in ("true", "false")
        if is_bool:
            correct = _bool_correct(pred, ta)
            err_pct = 0.0 if correct else 100.0
            full_ok = correct; budget_ok = correct
        elif pred is None:
            err_pct = None
            full_ok = False; budget_ok = False
        else:
            err_pct = _rel_error(pred, ta)
            full_ok   = (err_pct is not None and err_pct <= ft)
            budget_ok = (err_pct is not None and err_pct <= bt)

        total += 1
        if full_ok:   pass_full   += 1
        if budget_ok: pass_budget += 1

        status = "✓" if full_ok else "✗"
        err_str = f"{err_pct:6.2f}%" if err_pct is not None else "  N/A  "
        print(f"  [{i:3d}] {status} {sc:<42} {err_str}  [{elapsed:.1f}s]  {var[:25]}",
              flush=True)

        results.append({
            "idx": i, "scenario": sc, "variation": var,
            "true": str(ta), "pred": str(pred),
            "err_pct": float(err_pct) if err_pct is not None else None,
            "full_thr": ft, "budget_thr": bt,
            "full_pass": bool(full_ok), "budget_pass": bool(budget_ok),
            "elapsed_s": elapsed,
        })

    # Salveaza rezultate
    out_path = _OUT_DIR / "all206_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results,
                   "summary": {
                       "total": total,
                       "pass_full": pass_full,
                       "pass_budget": pass_budget,
                       "pct_full": pass_full / total * 100,
                       "pct_budget": pass_budget / total * 100,
                   }}, f, indent=2)

    print("\n" + "=" * 90)
    print(f"TOTAL: {total} task-uri")
    print(f"  PASS full_obs  ({5:.0f}%):  {pass_full}/{total} = {pass_full/total*100:.1f}%")
    print(f"  PASS budget    (variabil): {pass_budget}/{total} = {pass_budget/total*100:.1f}%")
    print()
    print(f"  Baseline paper ICML 2025 (top model o4-mini-high):")
    print(f"    Full observation:           74%")
    print(f"    Budget-limited (100 obs):   49%")
    print(f"\n  Rezultate salvate: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
