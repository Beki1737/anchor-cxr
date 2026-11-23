#!/usr/bin/env python3
"""E19 (round 44): ANYTIME-VALID MONITORING of the certified risk after deployment, then recalibration on alarm.
Validation folds 2-3 (in-distribution) and MIMIC (shift) only; never the ReXGradient test split. CPU only, reads the
work/e18 cache. Aggregates only.

Why: E17 showed the frozen guarantee fails at a new hospital (MIMIC risk ~0.25 vs alpha 0.10) and that 100-200
labelled local studies repair it. A site still needs to KNOW, during use, that it must recalibrate. In a drafting
workflow labels are free: the radiologist signs every report, so each signed study yields its loss
L_t = min(FP_t, 3)/3 a posteriori. We test H0: E[L_t | past] <= alpha for all t with a betting test martingale
    K_t = prod_{i<=t} (1 + b_i (L_i - alpha)),  b_i predictable in [0, 0.5/alpha]  (aGRAPA-style plug-in),
alarm when K_t >= 1/delta_m (delta_m = 0.05). Under H0, K is a nonnegative supermartingale, so by Ville's inequality
P(any false alarm, ever) <= delta_m, with no i.i.d. assumption and optional stopping allowed.
Change-point variant (deployment that starts in-distribution, then drifts): e-CUSUM M_t = (1 + b_t(L_t - alpha)) *
max(M_{t-1}, 1) with windowed bets, alarm at A = 1000 (average run length >= A under H0; Shin, Ramdas, Rinaldo).
After an alarm: collect the NEXT n_cal = 200 signed studies (whole patients), recalibrate the threshold with LTT
(Hoeffding-Bentkus, delta 0.10, same as E17b), deploy on the rest and keep monitoring.

PRE-STATED (round 44, before any run):
 H19a  in-distribution streams (valid folds 2-3, 1000 orders): false-alarm share <= 0.05 for the test martingale.
 H19b  MIMIC streams (frozen threshold): median detection <= 50 signed studies, >= 95% of orders alarm.
 H19c  monitor -> recalibrate: after recalibration, P(risk > alpha) <= 0.10 and post-recalibration alarm share <= 0.05.
 Disproof: false-alarm share > 0.05 (implementation or dependence problem), or median detection > 200 studies."""
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e17_recal as E  # noqa: E402
import e17b_pac as PAC  # noqa: E402

D18 = HOME / "work/e18"
ALPHA, DM, A_CUSUM = 0.10, 0.05, 1000.0
NPERM = 1000
SYSTEMS = {"ANCHOR": ("anchor", "t_anchor_multi"), "FUS": ("fus", "t_fus_multi")}


def frozen_loss(d, st, col):
    th = st.set_index("key")[col]
    adm = d[d.score.to_numpy() >= d.key.map(th).to_numpy()]
    fp = adm[~adm.label.astype(bool)].groupby("key").size().reindex(st.key).fillna(0).to_numpy()
    return np.minimum(fp, 3) / 3


def bets(L, window=None, m0=10.0, v0=0.05, cmax=0.5):
    """predictable plug-in bet b_t from L_1..L_{t-1} (optionally last `window` only), prior mean alpha with weight m0."""
    c1 = np.concatenate([[0.0], np.cumsum(L)])[:-1]; c2 = np.concatenate([[0.0], np.cumsum(L ** 2)])[:-1]
    n = np.arange(len(L), dtype=float)
    if window:
        s1 = c1 - np.concatenate([np.zeros(window), c1[:-window]]) if len(L) > window else c1
        s2 = c2 - np.concatenate([np.zeros(window), c2[:-window]]) if len(L) > window else c2
        n = np.minimum(n, window); c1, c2 = s1, s2
    mu = (m0 * ALPHA + c1) / (m0 + n)
    ex2 = (m0 * (v0 + ALPHA ** 2) + c2) / (m0 + n)
    var = np.maximum(ex2 - mu ** 2, 1e-4)
    g = mu - ALPHA
    return np.clip(g / (var + g ** 2), 0.0, cmax / ALPHA)


def alarm_tm(L):
    """first index (1-based count of studies) at which the test martingale crosses 1/DM; None if never."""
    lk = np.cumsum(np.log1p(bets(L) * (L - ALPHA)))
    hit = np.where(lk >= math.log(1 / DM))[0]
    return int(hit[0]) + 1 if len(hit) else None


def alarm_cusum(L, window=100):
    b = bets(L, window=window); m, out = 0.0, None
    for t in range(len(L)):
        m = (1 + b[t] * (L[t] - ALPHA)) * max(m, 1.0)
        if m >= A_CUSUM:
            return t + 1
    return out


def orders(pat, rng, block):
    if not block:
        return rng.permutation(len(pat))
    up = np.array(sorted(set(pat))); by = pd.Series(np.arange(len(pat))).groupby(pat).apply(list).to_dict()
    return np.array([i for p in rng.permutation(up) for i in by[p]])


def summ(x, n):
    a = np.array([v for v in x if v is not None], float)
    return {"alarm_share": len(a) / len(x), "median": float(np.median(a)) if len(a) else None,
            "p90": float(np.quantile(a, 0.9)) if len(a) else None, "stream_len": n}


def main():
    rng = np.random.default_rng(19); out = {}
    refm = E4.load_claims(HOME / "work/corpus_v2/mimic")
    for nm, (pool, col) in SYSTEMS.items():
        dv = pd.read_parquet(D18 / f"{pool}_valid.parquet"); sv = pd.read_parquet(D18 / "studies_valid.parquet")
        dm = pd.read_parquet(D18 / f"{pool}_mimic.parquet"); sm = pd.read_parquet(D18 / "studies_mimic.parquet")
        Lv_all = frozen_loss(dv, sv, col); ev = (sv.fold >= 2).to_numpy()
        Lv, pv = Lv_all[ev], sv.patient.to_numpy()[ev]
        Lm, pm = frozen_loss(dm, sm, col), sm.patient.to_numpy()
        print(f"\n==== {nm}-multi (frozen) ====\nrealised risk: valid folds 2-3 {Lv.mean():.4f} (n {len(Lv)}) | MIMIC {Lm.mean():.4f} (n {len(Lm)})")
        res = {"risk_valid_eval": float(Lv.mean()), "risk_mimic": float(Lm.mean())}
        if Lm.mean() > ALPHA > Lv.mean():                          # boundary null: i.i.d. mixture with mean exactly alpha
            q = (ALPHA - Lv.mean()) / (Lm.mean() - Lv.mean()); fb = []
            for _ in range(NPERM):
                z = rng.random(2000) < q
                fb.append(alarm_tm(np.where(z, rng.choice(Lm, 2000), rng.choice(Lv, 2000))))
            res["boundary_null_tm"] = summ(fb, 2000)
            print(f"boundary null (mixture with mean exactly {ALPHA}, 2000 studies, {NPERM} streams): false-alarm share "
                  f"{res['boundary_null_tm']['alarm_share']:.3f} (Ville bound {DM})", flush=True)
        for block in [False, True]:
            tag = "patient-block" if block else "study-random"
            fa = [alarm_tm(Lv[orders(pv, rng, block)]) for _ in range(NPERM)]
            fac = [alarm_cusum(Lv[orders(pv, rng, block)]) for _ in range(200)]
            dt = [alarm_tm(Lm[orders(pm, rng, block)]) for _ in range(NPERM)]
            dc = []
            for _ in range(200):                                   # change point: 1000 in-distribution studies, then MIMIC
                pre = Lv[rng.choice(len(Lv), 1000, replace=False)]
                a = alarm_cusum(np.concatenate([pre, Lm[orders(pm, rng, block)]]))
                dc.append(None if a is None else a - 1000)
            r = {"null_tm": summ(fa, len(Lv)), "null_cusum": summ(fac, len(Lv)), "mimic_tm": summ(dt, len(Lm)),
                 "change_cusum_delay": summ([x for x in dc if x is None or x > 0], len(Lm)),
                 "change_cusum_alarm_before_change": float(np.mean([x is not None and x <= 0 for x in dc]))}
            res[tag] = r
            print(f"[{tag}] in-distribution false-alarm share: test martingale {r['null_tm']['alarm_share']:.3f} ({NPERM} orders), "
                  f"e-CUSUM {r['null_cusum']['alarm_share']:.3f} (200 orders)")
            print(f"[{tag}] MIMIC test martingale: alarm share {r['mimic_tm']['alarm_share']:.3f}, detection after median "
                  f"{r['mimic_tm']['median']} / p90 {r['mimic_tm']['p90']} signed studies")
            print(f"[{tag}] change point (1000 valid then MIMIC), e-CUSUM A={A_CUSUM:.0f}: alarm before change "
                  f"{r['change_cusum_alarm_before_change']:.3f}, delay median {r['change_cusum_delay']['median']} / p90 "
                  f"{r['change_cusum_delay']['p90']}, detected share {r['change_cusum_delay']['alarm_share']:.3f}", flush=True)
        # monitor -> recalibrate (LTT, Hoeffding-Bentkus) -> keep monitoring, on MIMIC patient-block orders
        keys = list(sm.key)
        grid = np.unique(np.quantile(dm.score, np.linspace(0, 1, 800)))
        M = E.matrices(dm, keys, refm, grid)
        rec = []
        for _ in range(300):
            o = orders(pm, rng, True); a = alarm_tm(Lm[o])
            if a is None:
                continue
            seen = set(pm[o[:a]]); rest = o[a:]
            cal_idx, cnt, pats = [], 0, set()
            for i in rest:                                        # next whole patients until >= 200 studies
                if cnt >= 200 and pm[i] not in pats:
                    break
                cal_idx.append(i); cnt += 1; pats.add(pm[i])
            dep = np.array([i for i in rest[len(cal_idx):] if pm[i] not in pats and pm[i] not in seen])
            if len(dep) < 200:
                continue
            j = PAC.pac_index(M[0][np.array(cal_idx)][:, :-1])
            Lr = np.minimum(M[0][dep, j], 3) / 3
            rec.append({"alarm_at": a, "labels_used": a + len(cal_idx), "risk_after": float(Lr.mean()),
                        "recall_after": float(M[2][dep, j].sum() / max(M[3][dep].sum(), 1)),
                        "post_alarm": alarm_tm(Lr) is not None, "n_deployed": len(dep)})
        R = pd.DataFrame(rec)
        res["monitor_recal"] = {"runs": len(R), "labels_median": float(R.labels_used.median()), "risk_after_mean": float(R.risk_after.mean()),
                                "p_risk_over": float((R.risk_after > ALPHA).mean()), "recall_after": float(R.recall_after.mean()),
                                "post_recal_alarm_share": float(R.post_alarm.mean()), "deployed_median": float(R.n_deployed.median())}
        mr = res["monitor_recal"]
        print(f"[monitor->recalibrate, {mr['runs']} patient-block orders] labels used median {mr['labels_median']:.0f} (alarm + 200) | after "
              f"recalibration: risk mean {mr['risk_after_mean']:.4f}, P(risk > 0.10) {mr['p_risk_over']:.3f}, recall {mr['recall_after']:.3f}, "
              f"post-recalibration alarm share {mr['post_recal_alarm_share']:.3f} (deployed on median {mr['deployed_median']:.0f} studies)", flush=True)
        out[nm] = res
    json.dump(out, open(D18 / "e19_monitor.json", "w"), indent=1, default=float)
    print("E19_DONE")


if __name__ == "__main__":
    main()
