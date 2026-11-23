#!/usr/bin/env python3
"""E17b: E17 showed that recalibrated CRC holds the MIMIC risk IN EXPECTATION (mean 0.095 to 0.101) but individual
calibration draws exceed alpha 37 to 49% of the time. A deployment wants a high-probability statement. Here the same
recalibration uses Learn-then-Test with the Hoeffding-Bentkus upper confidence bound (Bates et al. 2021; Angelopoulos
et al. 2025): P(risk > alpha) <= delta, delta = 0.10, fixed-sequence testing from strict to permissive thresholds.
Scorers frozen, threshold only, 300 patient-disjoint splits per n_cal. (~/venvs/anchor, small GPU for the probe)
Aggregates only."""
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e8_final as F  # noqa: E402
import e16_fusion as FU  # noqa: E402
import e17_recal as E  # noqa: E402

ALPHA, DELTA = 0.10, 0.10
NCAL = [100, 200, 400, 800]
_K = {}


def kmax(n):
    """largest k with e * P(Bin(n, alpha) <= k) <= delta (Bentkus); -1 if none."""
    if n not in _K:
        c, best = 0.0, -1
        for k in range(n + 1):
            c += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(ALPHA) + (n - k) * math.log(1 - ALPHA))
            if math.e * c <= DELTA:
                best = k
            else:
                break
        _K[n] = best
    return _K[n]


def pac_index(FPc):
    n = FPc.shape[0]
    r = (np.minimum(FPc, 3) / 3).mean(0)
    ok = (r <= ALPHA - math.sqrt(math.log(1 / DELTA) / (2 * n))) | (np.ceil(n * r - 1e-9) <= kmax(n))
    bad = np.where(~ok)[0]
    return (bad.max() + 1) if len(bad) else 0


def main():
    fz = json.load(open(F.OUT / "freeze.json")); fz2 = json.load(open(FU.OUT / "freeze2.json"))
    C = F.load_cands("mimic"); ref = E4.load_claims(HOME / "work/corpus_v2/mimic")
    keys = sorted(C.key.unique())
    man = pd.read_parquet(HOME / "work/manifest/rex_mimic.parquet").set_index("key").reindex(keys)
    pat = man.patient.astype(str).to_numpy()
    probe = F.probe_arrays("mimic", keys)
    ex, fu = FU.build("mimic", C, probe, ref, fz["tau"])
    systems = {"ANCHOR": C.assign(score=F.apply_scorer(C, fz["scorer_full"], fz["delta"])),
               "FUS": fu.assign(score=F.apply_scorer(fu, fz2["scorer_fus"], fz2["delta"]))}
    up = np.array(sorted(set(pat))); pc = pd.Series(pat).value_counts(); rng = np.random.default_rng(1)
    res = {}
    print(f"Learn-then-Test (Hoeffding-Bentkus), alpha {ALPHA}, delta {DELTA}; MIMIC {len(keys)} studies / {len(up)} patients")
    for nm, d in systems.items():
        grid = np.unique(np.quantile(d.score, np.linspace(0, 1, 800)))
        M = E.matrices(d, keys, ref, grid)
        print(f"\n{nm}: {'n_cal':>6} {'risk mean':>10} {'P(risk>a)':>10} {'risk p90':>9} {'recall':>8} {'precision':>10} {'claims':>7}   (CRC row for contrast)")
        res[nm] = {}
        for n in NCAL:
            out, outc = [], []
            for _ in range(E.NSPLIT):
                cal_p, cnt = [], 0
                for p in rng.permutation(up):
                    cal_p.append(p); cnt += pc[p]
                    if cnt >= n:
                        break
                cm = np.isin(pat, cal_p); cal, ev = np.where(cm)[0], np.where(~cm)[0]
                out.append(E.evalm(M, ev, pac_index(M[0][cal][:, :-1])))
                outc.append(E.evalm(M, ev, E.crc_index(M[0][cal][:, :-1], ALPHA)))
            for tag, O in [("LTT", np.array(out, float)), ("CRC", np.array(outc, float))]:
                row = {"risk_mean": round(float(O[:, 0].mean()), 4), "p_over": round(float((O[:, 0] > ALPHA).mean()), 3),
                       "risk_p90": round(float(np.quantile(O[:, 0], 0.9)), 4), "recall": round(float(O[:, 1].mean()), 4),
                       "precision": round(float(np.nanmean(O[:, 2])), 4), "claims": round(float(O[:, 3].mean()), 2)}
                res[nm][f"{tag}|{n}"] = row
                print(f"  {tag:<4} {n:>6} {row['risk_mean']:>10.4f} {row['p_over']:>10.3f} {row['risk_p90']:>9.4f} {row['recall']:>8.3f} "
                      f"{row['precision']:>10.3f} {row['claims']:>7.2f}", flush=True)
    (HOME / "work/e17").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HOME / "work/e17/pac_mimic.json", "w"), indent=1)
    print("E17B_DONE")


if __name__ == "__main__":
    main()
