"""
GravityBench — solver pentru protocolul "budget-obs-100".

In acest mod, agentul vede DOAR 100 observatii (uniform-spatiate in timp,
fara planificare). Aceasta replica baseline-ul "human-ref-100" din paper.

Scopul: sa producem rezultate comparabile cu Tabelul 1 din paper:
  - Full-obs (100% PhD reference, 74% top LLM)
  - Budget-obs-100 (82.5% PhD reference, 49% top LLM)
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solve_all_206 import solve_task, _OUT_DIR


def _subsample_uniform(row: dict, n_obs: int = 100) -> dict:
    """
    Returneaza o copie a row-ului dataset cu CSV redus la n_obs puncte
    spatiate uniform in timp (cum face human-ref-100 din paper).
    """
    import pandas as pd
    from io import StringIO
    df = pd.read_csv(StringIO(row["simulation_csv_content"]))
    n = len(df)
    if n <= n_obs:
        return row
    # uniform sampling in indici (timp uniform-spatiat)
    idx = np.linspace(0, n - 1, n_obs).astype(int)
    df_sub = df.iloc[idx].reset_index(drop=True)
    csv_sub = df_sub.to_csv(index=False)
    return {**row, "simulation_csv_content": csv_sub}


def _rel_error(pred, true_val):
    try:
        p = float(pred); t = float(true_val)
        if abs(t) < 1e-30:
            return abs(p - t)
        return abs(p - t) / abs(t) * 100.0
    except Exception:
        return None


def main():
    import datasets as _ds
    d = _ds.load_dataset(
        "GravityBench/GravityBench",
        cache_dir="/tmp/gravitybench_cache",
    )["test"]
    rows = list(d)
    print(f"\nGravityBench BUDGET-OBS-100 — {len(rows)} task-uri")
    print("Subsample 100 observatii uniform spatiate (no planning).\n")

    pass_full = 0; pass_budget = 0; total = 0
    results = []

    for i, row in enumerate(rows):
        sc = row["scenario_name"]
        var = row.get("variation_name", "")[:30]
        ta = row["true_answer"]
        ft = float(row.get("full_obs_threshold_percent", 5.0))
        bt = float(row.get("budget_obs_threshold_percent", ft))

        sub = _subsample_uniform(row, n_obs=100)

        t0 = time.time()
        try:
            pred = solve_task(sub)
        except Exception as ex:
            pred = None
        dt = time.time() - t0

        is_bool = str(ta).strip().lower() in ("true", "false")
        if is_bool:
            ok = str(pred).strip().lower() == str(ta).strip().lower()
            err_pct = 0.0 if ok else 100.0
            full_ok = ok; budget_ok = ok
        elif pred is None:
            err_pct = None
            full_ok = False; budget_ok = False
        else:
            err_pct = _rel_error(pred, ta)
            full_ok   = err_pct is not None and err_pct <= ft
            budget_ok = err_pct is not None and err_pct <= bt

        total += 1
        if full_ok:   pass_full   += 1
        if budget_ok: pass_budget += 1

        status = "✓" if budget_ok else "✗"
        err_str = f"{err_pct:6.2f}%" if err_pct is not None else "  N/A "
        print(f"  [{i:3d}] {status} {sc:<42} {err_str}  bt={bt:.0f}%  {var[:20]}",
              flush=True)

        results.append({
            "idx": i, "scenario": sc, "variation": var,
            "true": str(ta), "pred": str(pred),
            "err_pct": float(err_pct) if err_pct is not None else None,
            "full_thr": ft, "budget_thr": bt,
            "full_pass": bool(full_ok), "budget_pass": bool(budget_ok),
            "elapsed_s": dt,
        })

    out_path = _OUT_DIR / "all206_budget100_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "total": total,
                "pass_full_5pct":   pass_full,
                "pass_budget_thr":  pass_budget,
                "pct_full_5pct":    pass_full / total * 100,
                "pct_budget_thr":   pass_budget / total * 100,
            },
        }, f, indent=2)

    print("\n" + "=" * 90)
    print(f"BUDGET-OBS-100 — TOTAL: {total} task-uri")
    print(f"  PASS full_obs criterion (5% strict):    {pass_full}/{total} = {pass_full/total*100:.1f}%")
    print(f"  PASS budget criterion (task-specific):  {pass_budget}/{total} = {pass_budget/total*100:.1f}%")
    print()
    print("  Comparatie cu Tabelul 1 din paper (budget-obs-100):")
    print("    PhD-level human ref: 82.5%")
    print("    o4-mini-high LLM:    49%")
    print("    Claude 3.5 Sonnet:   21.5%")
    print("    GPT-4o:              15.5%")
    print()
    print(f"  Rezultate: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
