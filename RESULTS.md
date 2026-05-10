# Gravity-Bench-v1 — Results report

Companion to `solve_all_206.py`. Reproduces the PhD-level reference solver
of Koblischke et al. (ICML 2025) under a strict no-leakage protocol.

## 1. Headline numbers

| Mode               | Our solver | Paper PhD ref. | o4-mini-high | Claude 3.5 S | GPT-4o |
|--------------------|-----------:|---------------:|-------------:|-------------:|-------:|
| full-obs (5% strict)         | **100.0%** | 100.0%  | —    | —    | —    |
| full-obs (task threshold)    | **100.0%** | 100.0%  | 74%  | 39.5%| 36.1%|
| budget-obs-100 (5% strict)   |  73.8%     | —       | —    | —    | —    |
| budget-obs-100 (task thr.)   | **86.4%**  | 82.5%   | 49%  | 21.5%| 15.5%|

LLM and PhD numbers are taken from Table 1 of the paper.

## 2. Setup and constraints

* Dataset: HuggingFace `GravityBench/GravityBench`, split `test`, 206 tasks.
* Hardware: single 14-core x86 CPU. No GPU.
* Time: ~25 s end-to-end for `full-obs`, ~22 s for `budget-obs-100`.
* No external services, no API calls, no LLM.

The solver consumes only what an agent receives at evaluation time: the
positions in `simulation_csv_content` and `scenario_name` for dispatch.
It does not read `variation_name`, which encodes ground-truth masses
("21.3 M, 3.1 M"), unit system (`yrAUMsun`, `cgs`), and OOD labels
("Modified Gravity 1.97", "Drag tau = 1.7e9", "Unbound"). We treat that
field as benchmark-builder metadata.

Automatic unit detection: we try (SI), (cgs), (yr-AU-Msun) and accept the
one for which `M = a · d² / G` lies in the stellar range (10²⁸..10³³ kg).

## 3. Error distribution (full-obs, numeric tasks only)

* median: **0.0001 %**
* mean:   **0.045 %**
* max:    **2.73 %** (`linear_drag` τ = 8.3e8)
* tasks > 1 %:  3 (all in `linear_drag`)
* tasks > 4 %:  0
* boolean tasks (kepler_3rd_law, virial_theorem, is_bound): all correct

## 4. Methods (one line per task family)

| Family | Method |
|---|---|
| Geometry (apo/peri/SMA/ecc/period) | min/max of \|r₂−r₁\|, peak detection on r(t) |
| Kinematics (v, a, ω) | central finite differences on positions |
| Stellar masses | M_other = a_self · d² / G (paper Appendix B) |
| Energies (K+U) | estimated masses + finite-difference velocities |
| Conservation (h, dA/dt) | r × v from relative coordinates |
| Roche lobe | Eggleton (1983) with q = m₁/m₂ |
| Kepler 3rd law (bool) | False if log–log slope ≠ −2, or if SMA decays |
| Virial (bool) | False if SMA decays or if K+U ≥ 0 |
| is_bound (bool) | sign of median(K+U) |
| Unit handling | try {SI, cgs, yr-AU-Msun}, accept stellar mass |
| `modified_gravity_power_law` | linear fit of log(a) vs log(r); slope = −(2+α) |
| `linear_drag` | apo/peri detection, fit SMA(t) = a₀·exp(−2t/τ) |

## 5. Where budget-obs-100 fails (28/206)

| Scenario | Failures | Cause (consistent with paper) |
|---|---:|---|
| `max_angular_velocity_starN` | 8 | peaks missed by uniform sampling |
| `max_velocity_star1`, `max_momentum_star1`, `max_acceleration_star1` | 6 | same peak-miss issue |
| `area_swept_over_time_peri` | 3 | finite differences noisy at periastron |
| `kepler_3rd_law` | 3 | log-log slope noisy → false negative |
| `modified_gravity_power_law` | 3 | only 100 obs → log-log fit drifts |
| `travel_time_orbital_20per_path` | 3 | path integration sensitive |
| `eccentricity`, `periastron` | 2 | one extreme orbit (Elliptical, Single) |

These failure modes match the discussion in §4.4 of the paper: uniform
sampling at 100 points is fundamentally unable to localize narrow peaks
or to fit a precise log–log slope.

## 6. Honest disclaimer

This work re-implements the published PhD-level reference solver. It is
**not a new agent**: there is no observation planning, no natural-language
reasoning, and no incremental uncertainty reduction. The benchmark is
designed to evaluate agentic capabilities, which we do not address.

What this work does provide is (1) an end-to-end runnable reference
solver, (2) cross-validation that the paper's full-obs PhD score of
100% is reachable under a strict no-leakage protocol, and (3) a slightly
higher budget-obs-100 score (86.4 % vs. 82.5 %) due to automatic unit
detection and a data-driven detection of OOD scenarios.

## 7. Citation

Please cite the original benchmark:

```
@article{koblischke2025gravitybench,
  title  = {Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents},
  author = {Koblischke, Nolan and Jang, Hyunseok and Menou, Kristen and Ali-Dib, Mohamad},
  journal= {ICML 2025},
  year   = {2025},
  eprint = {2501.18411},
  archivePrefix = {arXiv},
}
```

This reproduction can be cited via the Zenodo DOI listed in the README.
