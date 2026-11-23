#!/usr/bin/env python3
"""E17: the frozen guarantee breaks on MIMIC (ANCHOR capped-FP risk 0.249 vs alpha 0.10; B0 0.314). Pre-registered
remedy: RECALIBRATE, DON'T RETRAIN. Scorers stay frozen (ReXGradient); only the CRC threshold is re-estimated on a
small labelled MIMIC calibration sample, evaluated on the remaining MIMIC studies. (~/venvs/anchor, small GPU for probe)

  n_cal in {25, 50, 100, 200, 400, 800} studies, 300 random patient-disjoint splits each
  report: mean realized risk on the held-out remainder (CRC promises <= alpha in expectation), share of splits above
          alpha, recall, precision, claims per report; frozen (no recalibration) row for reference
  systems: ANCHOR (freeze.json scorer) and FUS (freeze2.json scorer, E16), plus view-Mondrian recalibration (AP / PA)
Claim truth = lenient match to the extractor reference (same as calibration on ReXGradient). Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e8_final as F  # noqa: E402
import e16_fusion as FU  # noqa: E402

ALPHA = 0.10
NCAL = [25, 50, 100, 200, 400, 800]
NSPLIT = 300


def matrices(d, keys, ref, grid):
    """Per study x threshold: false claims, true claims, all claims, covered reference findings (exact, per reference)."""
    pos = {k: i for i, k in enumerate(keys)}; S, Gn = len(keys), len(grid)
    gi = np.searchsorted(grid, d.score.to_numpy(), side="right")          # claim admitted for grid index < gi
    si = d.key.map(pos).to_numpy()
    FP, TP = np.zeros((S, Gn + 1)), np.zeros((S, Gn + 1))
    lab = d.label.to_numpy().astype(bool)
    np.add.at(FP, (si[~lab], gi[~lab]), 1); np.add.at(TP, (si[lab], gi[lab]), 1)
    FP = np.cumsum(FP[:, ::-1], 1)[:, ::-1][:, 1:]; TP = np.cumsum(TP[:, ::-1], 1)[:, ::-1][:, 1:]
    COV = np.zeros((S, Gn + 1)); NREF = np.zeros(S)
    by = {k: g for k, g in d.groupby("key")}
    for k in keys:
        R = list(ref.get(k, {})); NREF[pos[k]] = len(R)
        g = by.get(k)
        for r in R:
            if g is None:
                continue
            m = g[(g.finding == r[0]).to_numpy() & np.array([E4.side_ok(l, r[1]) for l in g.lat], bool)]
            if len(m):
                COV[pos[k], np.searchsorted(grid, m.score.max(), side="right")] += 1
    COV = np.cumsum(COV[:, ::-1], 1)[:, ::-1][:, 1:]
    z = np.zeros((S, 1))                                                    # extra index Gn = admit nothing
    return np.hstack([FP, z]), np.hstack([TP, z]), np.hstack([COV, z]), NREF


def crc_index(FPc, alpha):
    """grid ascending; most permissive index j such that every stricter index also satisfies the adjusted bound."""
    n = FPc.shape[0]
    r = (np.minimum(FPc, 3) / 3).mean(0)
    ok = n / (n + 1) * r + 1 / (n + 1) <= alpha
    bad = np.where(~ok)[0]
    return (bad.max() + 1) if len(bad) else 0                       # first index above the last violation (strictest side = high index)


def evalm(M, rows, j):
    FP, TP, COV, NREF = M
    fp, tp = FP[rows, j], TP[rows, j]
    return float((np.minimum(fp, 3) / 3).mean()), COV[rows, j].sum() / max(NREF[rows].sum(), 1), tp.sum() / max((tp + fp).sum(), 1), (tp + fp).mean()


def main():
    fz = json.load(open(F.OUT / "freeze.json")); fz2 = json.load(open(FU.OUT / "freeze2.json"))
    C = F.load_cands("mimic"); ref = E4.load_claims(HOME / "work/corpus_v2/mimic")
    keys = sorted(C.key.unique())
    man = pd.read_parquet(HOME / "work/manifest/rex_mimic.parquet").set_index("key").reindex(keys)
    pat = man.patient.astype(str).to_numpy(); view = man.frontal_view.astype(str).to_numpy()
    probe = F.probe_arrays("mimic", keys)
    ex, fu = FU.build("mimic", C, probe, ref, fz["tau"])
    systems = {"ANCHOR": (C.assign(score=F.apply_scorer(C, fz["scorer_full"], fz["delta"])), fz["t_full"]),
               "FUS": (fu.assign(score=F.apply_scorer(fu, fz2["scorer_fus"], fz2["delta"])), fz2["t_fus"])}
    print(f"MIMIC studies {len(keys)} from {len(set(pat))} patients | views {dict(pd.Series(view).value_counts())}")
    up = np.array(sorted(set(pat))); rng = np.random.default_rng(0); pc = pd.Series(pat).value_counts()
    res = {}
    for nm, (d, t_frozen) in systems.items():
        grid = np.unique(np.quantile(d.score, np.linspace(0, 1, 800)))
        M = matrices(d, keys, ref, grid)
        jf = int(np.searchsorted(grid, t_frozen, side="left"))
        r0 = evalm(M, np.arange(len(keys)), jf)
        print(f"\n{nm}: frozen ReXGradient threshold -> risk {r0[0]:.3f} recall {r0[1]:.3f} precision {r0[2]:.3f} claims/rpt {r0[3]:.2f}")
        res[nm] = {"frozen": [round(x, 4) for x in r0]}
        print(f"  {'n_cal':>6} {'risk mean':>10} {'P(risk>a)':>10} {'risk p90':>9} {'recall':>8} {'precision':>10} {'claims':>7} | view-Mondrian: risk, P>a, recall, worst-view risk")
        for n in NCAL:
            out, outm = [], []
            for _ in range(NSPLIT):
                perm = rng.permutation(up); cal_p, cnt = [], 0
                for p in perm:
                    cal_p.append(p); cnt += pc[p]
                    if cnt >= n:
                        break
                cm = np.isin(pat, cal_p); cal, ev = np.where(cm)[0], np.where(~cm)[0]
                j = crc_index(M[0][cal][:, :-1], ALPHA)
                out.append(evalm(M, ev, j))
                # view-Mondrian: separate threshold per view from the same calibration sample
                js = {v: crc_index(M[0][cal[view[cal] == v]][:, :-1], ALPHA) if (view[cal] == v).sum() >= 10 else j for v in set(view)}
                jj = np.array([js[v] for v in view])
                fp = M[0][ev, jj[ev]]; tp = M[1][ev, jj[ev]]; cov = M[2][ev, jj[ev]]
                wv = max(float((np.minimum(fp[view[ev] == v], 3) / 3).mean()) for v in set(view[ev]))
                outm.append((float((np.minimum(fp, 3) / 3).mean()), cov.sum() / max(M[3][ev].sum(), 1), wv))
            O, Om = np.array(out, float), np.array(outm, float)
            row = {"risk_mean": round(float(O[:, 0].mean()), 4), "p_over": round(float((O[:, 0] > ALPHA).mean()), 3),
                   "risk_p90": round(float(np.quantile(O[:, 0], 0.9)), 4), "recall": round(float(O[:, 1].mean()), 4),
                   "precision": round(float(np.nanmean(O[:, 2])), 4), "claims": round(float(O[:, 3].mean()), 2),
                   "mondrian_risk": round(float(Om[:, 0].mean()), 4), "mondrian_p_over": round(float((Om[:, 0] > ALPHA).mean()), 3),
                   "mondrian_recall": round(float(Om[:, 1].mean()), 4), "mondrian_worst_view": round(float(Om[:, 2].mean()), 4)}
            res[nm][str(n)] = row
            print(f"  {n:>6} {row['risk_mean']:>10.4f} {row['p_over']:>10.3f} {row['risk_p90']:>9.4f} {row['recall']:>8.3f} {row['precision']:>10.3f} "
                  f"{row['claims']:>7.2f} | {row['mondrian_risk']:.4f}, {row['mondrian_p_over']:.3f}, {row['mondrian_recall']:.3f}, {row['mondrian_worst_view']:.4f}", flush=True)
    (HOME / "work/e17").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HOME / "work/e17/recal_mimic.json", "w"), indent=1)
    print("E17_DONE")


if __name__ == "__main__":
    main()
