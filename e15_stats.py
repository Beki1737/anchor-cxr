#!/usr/bin/env python3
"""E15: statistical hardening of the TEST comparisons (any env with pandas; CPU). Aggregates only.
  - patient-cluster bootstrap (test patients can have several studies) next to the study bootstrap
  - Holm-Bonferroni adjustment over the family of pre-specified comparisons (each system vs B0, CRIMSON and GREEN)
  - win / tie / loss per comparison and the normal / abnormal strata (E1 strata)"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import study_strata  # noqa: E402

PAIRS = [("e8anchor", "b0_medgemma4b"), ("e8edit", "b0_medgemma4b"), ("e8multi", "b0_medgemma4b"), ("e8vlmfree", "b0_medgemma4b"),
         ("e3rag_medgemma4b_k3", "b0_medgemma4b"), ("b0_medgemma4b", "normal_template"), ("e8anchor", "normal_template")]


def cluster_boot(d, groups, n=2000, seed=0):
    g = pd.Series(d).groupby(groups.to_numpy())
    s, c = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(seed); out = np.empty(n)
    for b in range(n):
        i = rng.integers(0, len(s), len(s))
        out[b] = s[i].sum() / c[i].sum()
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "test"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key")
    st = study_strata(HOME / f"work/corpus_v2/{split}")
    res, pvals = {}, {}
    for judge in ["crimson", "green"]:
        f = HOME / f"work/judge/{tag}/{judge}_scores.parquet"
        if not f.exists():
            continue
        w = pd.read_parquet(f).pivot_table(index="key", columns="cond", values="score")
        pat = man.patient.astype(str).reindex(w.index)
        print(f"\n{judge.upper()} ({tag}): {len(w)} studies from {pat.nunique()} patients "
              f"({(pat.value_counts() > 1).sum()} patients with >1 study)")
        for a, b in PAIRS:
            if a not in w or b not in w:
                continue
            x = w[[a, b]].dropna(); d = (x[a] - x[b]).to_numpy(); p_ = pat.reindex(x.index)
            rng = np.random.default_rng(0)
            sb = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            cb = cluster_boot(d, p_)
            p = float(min(1.0, 2 * min((cb <= 0).mean(), (cb >= 0).mean())))
            p = max(p, 1 / 2000)
            r = {"n": len(d), "diff": round(float(d.mean()), 4), "ci_study": [round(float(v), 4) for v in np.percentile(sb, [2.5, 97.5])],
                 "ci_patient_cluster": [round(float(v), 4) for v in np.percentile(cb, [2.5, 97.5])], "p_cluster": p,
                 "win_tie_loss": [round(float(v), 3) for v in ((d > 1e-9).mean(), (np.abs(d) <= 1e-9).mean(), (d < -1e-9).mean())]}
            for nm, sel in [("normal", ~st.abn.reindex(x.index).fillna(False).astype(bool)), ("abnormal", st.abn.reindex(x.index).fillna(False).astype(bool))]:
                dd = d[sel.to_numpy()]
                cbs = cluster_boot(dd, p_[sel.to_numpy()])
                r[nm] = {"n": int(sel.sum()), "diff": round(float(dd.mean()), 4), "ci_patient_cluster": [round(float(v), 4) for v in np.percentile(cbs, [2.5, 97.5])]}
            res[f"{judge}|{a} vs {b}"] = r; pvals[f"{judge}|{a} vs {b}"] = p
    # Holm
    order = sorted(pvals, key=pvals.get); m = len(order); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        res[k]["p_holm"] = round(run, 4)
    print(f"\n{'comparison':<46}{'diff':>8}  {'95% CI study':<18}{'95% CI patient':<18}{'Holm p':>8}  normal | abnormal")
    for k, r in res.items():
        print(f"{k:<46}{r['diff']:>+8.4f}  {str(r['ci_study']):<18}{str(r['ci_patient_cluster']):<18}{r['p_holm']:>8.4f}  "
              f"{r['normal']['diff']:+.3f} {r['normal']['ci_patient_cluster']} | {r['abnormal']['diff']:+.3f} {r['abnormal']['ci_patient_cluster']}")
    (HOME / "work/e15").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HOME / f"work/e15/stats_{tag}.json", "w"), indent=1)
    print("E15_DONE")


if __name__ == "__main__":
    main()
