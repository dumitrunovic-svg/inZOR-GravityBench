# Gravity-Bench-v1 — Independent reference solver (no metadata leakage)

Open-source Python implementation of the PhD-level reference algorithms
described in Appendix B of:

> Koblischke, N., Jang, H., Menou, K., and Ali-Dib, M. (2025).
> *Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents*.
> ICML 2025. [arXiv:2501.18411](https://arxiv.org/abs/2501.18411).

## Headline results

| Protocol                       | This solver | Paper PhD ref. | Top LLM |
|--------------------------------|------------:|---------------:|--------:|
| `full-obs` (5% strict)         | **100.0%**  |        100.0%  |   —     |
| `full-obs` (task threshold)    | **100.0%**  |        100.0%  |  74%    |
| `budget-obs-100` (5% strict)   |  73.8%      |              — |   —     |
| `budget-obs-100` (task thr.)   | **86.4%**   |         82.5%  |  49%    |

Total wall time: ~25 s per protocol on a 14-core CPU. No GPU, no API,
no LLM in the loop.

* Median error on `full-obs` numeric tasks: **0.0001%**
* Mean error: **0.045%**
* Max error: **2.73%** (a `linear_drag` variant)

## Why this might be useful

1. A runnable reference solver makes it easy to reproduce the numbers
   in Table 1 of the paper.
2. It can be used to cross-check future agent runs without re-running
   the LLM baselines.
3. The unit-detection trick may be of independent interest for any
   benchmark that mixes SI / cgs / yr-AU-Msun representations.

## Honest disclaimer

This is **not a new agent**. It does not plan observations, does not reason
in natural language, and does not reduce uncertainty incrementally. The
whole point of GravityBench is to evaluate agentic capabilities, which this
work does not address. We simply re-implement the PhD-level reference
solver from Appendix B and verify that it indeed reaches 100% on `full-obs`
when no metadata leakage is allowed.

## No-leakage protocol

The solver consumes only:

* `simulation_csv_content` — star positions x, y, z and time
* `scenario_name` — used solely for task dispatch

It does **not** read `variation_name`, which exposes ground-truth masses
("21.3 M, 3.1 M"), the unit system (`yrAUMsun`, `cgs`), and OOD labels
("Modified Gravity 1.97", "Drag tau = 1.7e9", "Unbound"). We treat that
field as benchmark-builder metadata, not agent input.

Unit detection is automatic: we try (SI), (cgs), and (yr-AU-Msun) and
pick the one for which `M = a · d² / G` falls in the stellar range
(10²⁸ … 10³³ kg).

## Files

* `solve_all_206.py` — full-obs solver (≈900 lines, depends on
  numpy / scipy / pandas / datasets).
* `solve_all_206_budget.py` — wrapper that uniform-subsamples each
  trajectory to 100 points and calls the same solver.
* `results/all206_results.json` — per-task results on `full-obs`.
* `results/all206_budget100_results.json` — per-task results on
  `budget-obs-100`.
* `RESULTS.md` — extended report with method details.

## Run

```bash
pip install numpy scipy pandas datasets
python3 solve_all_206.py
python3 solve_all_206_budget.py
```

Expected output ends with `PASS full_obs (5%): 206/206 = 100.0%`.

## Note on optional alternative discovery

For the six out-of-distribution tasks (`modified_gravity_power_law` and
`linear_drag`), the published reference uses dedicated estimators
(log-log slope and exponential SMA decay). We separately verified, in a
private codebase, that an *evolutionary search* over the parameter space
(without any prior knowledge of the underlying functional form) also
recovers the parameters within the published thresholds. That work is
not included here because it depends on a proprietary search framework;
the open-source numbers above use only the algorithms in this repository.

## License

MIT. Please cite both the original benchmark paper and (optionally) this
repository.
