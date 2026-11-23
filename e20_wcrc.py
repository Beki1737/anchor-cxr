#!/usr/bin/env python3
"""E20 (round 44): can the new-site failure be FIXED or at least PREDICTED without labels? Covariate-shift weighted CRC.
Validation fold 1 (source calibration) and MIMIC (target) only; never the ReXGradient test split. CPU only, reads the
work/e18 cache. Aggregates only.

Method (Tibshirani et al. 2019 weighted conformal, applied to risk control): a domain classifier separates source
calibration studies (valid fold 1) from UNLABELLED target studies (MIMIC half A, patient-disjoint), cross-fitted 5-fold,
logistic regression on (a) PCA-32 of the RAD-DINO CLS embedding, (b) logits of the 30 global-probe probabilities
(case-mix), (c) both; plus view. Likelihood ratio w(x) = p/(1-p) * n_s/n_t, clipped at the 99th source percentile.
Per test point x: most permissive threshold with (sum_i w_i L_i(lam) + w(x) * 1) / (sum_i w_i + w(x)) <= alpha for
every stricter lam (the standard weighted-CRC bound with B = 1); if none, admit nothing.
Evaluated on MIMIC half B with the true labels. Also reported: the LABEL-FREE risk estimate of the frozen system,
sum_i w_i L_i(frozen) / sum_i w_i, against the true MIMIC risk, and the weight effective sample size.
Reference rows: unweighted CRC (w = 1, = frozen), oracle CRC calibrated on LABELLED MIMIC half A. 20 random halvings.

PRE-STATED (round 44, before any run):
 H20   weighted CRC brings the MIMIC half-B risk to <= 0.15 without target labels, with ESS >= 100.
 H20b  the label-free estimate of the frozen risk is within 0.05 of the true MIMIC risk (a usable pre-deployment alarm).
 Disproof: weighted risk >= 0.20 or ESS < 50 (shift is not covariate-only, e.g. reporting-style / label shift, so
       labels are required, which supports the recalibration recipe); estimate off by > 0.10."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e17_recal as E  # noqa: E402

D18 = HOME / "work/e18"
ALPHA = 0.10
SYSTEMS = {"ANCHOR": ("anchor", "t_anchor_multi"), "FUS": ("fus", "t_fus_multi")}
NREP = 20


def logreg(X, y, l2=1.0, iters=300):
    """L2 logistic regression by Newton steps (numpy)."""
    X1 = np.hstack([X, np.ones((len(X), 1))]); w = np.zeros(X1.shape[1]); R = l2 * np.eye(X1.shape[1]); R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) + R @ w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + R
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.clip(np.hstack([X, np.ones((len(X), 1))]) @ w, -30, 30)))


def auroc(s, y):
    r = pd.Series(s).rank().to_numpy(); y = y.astype(bool)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def features(F, keys, kind, pca, views):
    pos = {k: i for i, k in enumerate(F["keys"])}; ix = [pos[k] for k in keys]
    v = np.asarray(views).astype(str)
    parts = [np.stack([(v == "PA"), (v == "AP")], 1).astype(np.float64)]
    if kind in ("cls", "both"):
        X = F["cls"][ix].astype(np.float64)
        mu, Vt = pca
        parts.append((X - mu) @ Vt[:32].T)
    if kind in ("probe", "both"):
        p = np.clip(F["pg"][ix].astype(np.float64), 1e-4, 1 - 1e-4); parts.append(np.log(p / (1 - p)))
    return np.hstack(parts)


def domain_weights(Xs, Xa, Xb, rng):
    """cross-fitted P(target | x); returns likelihood ratios for source rows and target-B rows, and out-of-fold AUROC."""
    X = np.vstack([Xs, Xa]); y = np.r_[np.zeros(len(Xs)), np.ones(len(Xa))]
    mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd; Xb = (Xb - mu) / sd
    f = rng.integers(0, 5, len(X)); oof = np.zeros(len(X)); pb = np.zeros(len(Xb))
    for k in range(5):
        w = logreg(X[f != k], y[f != k]); oof[f == k] = predict(w, X[f == k]); pb += predict(w, Xb) / 5
    ratio = len(Xs) / len(Xa)
    lr = lambda p: np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1) * ratio  # noqa: E731
    ws, wb = lr(oof[:len(Xs)]), lr(pb)
    cap = np.quantile(ws, 0.99)
    return np.minimum(ws, cap), np.minimum(wb, cap), auroc(oof, y)


def wcrc_index(FPs, ws, wx):
    """per test point: most permissive grid index j with (sum w_i L_i(j') + w_x) / (W + w_x) <= alpha for all j' >= j."""
    Ls = np.minimum(FPs, 3) / 3                                            # source rows x (grid + admit-nothing)
    Rw = ws @ Ls                                                           # weighted loss sum per index
    Rmax = np.maximum.accumulate(Rw[::-1])[::-1]                           # max over stricter indices
    budget = ALPHA * ws.sum() - (1 - ALPHA) * wx                           # need Rmax[j] <= budget
    G = len(Rw) - 1                                                        # last index = admit nothing (loss 0)
    # Rmax is non-increasing in j: first j with Rmax[j] <= budget
    j = np.searchsorted(-Rmax, -budget, side="left")
    return np.where(budget < 0, G, np.minimum(j, G))


def main():
    rng = np.random.default_rng(20); out = {}
    # our own cache (written by e18_cache.py); keys were saved as an object array, so pickle is needed to read them
    Fv, Fm = (dict(np.load(D18 / f"feat_{s}.npz", allow_pickle=True)) for s in ("valid", "mimic"))
    for Fx in (Fv, Fm):
        Fx["keys"] = np.array([str(k) for k in Fx["keys"]])
    refv, refm = E4.load_claims(HOME / "work/corpus_v2/valid"), E4.load_claims(HOME / "work/corpus_v2/mimic")
    sv, sm = pd.read_parquet(D18 / "studies_valid.parquet"), pd.read_parquet(D18 / "studies_mimic.parquet")
    src = sv[sv.fold == 1].reset_index(drop=True)
    for nm, (pool, col) in SYSTEMS.items():
        dv = pd.read_parquet(D18 / f"{pool}_valid.parquet"); dm = pd.read_parquet(D18 / f"{pool}_mimic.parquet")
        dv = dv[dv.key.isin(set(src.key))]
        grid = np.unique(np.quantile(np.r_[dv.score, dm.score], np.linspace(0, 1, 600)))
        Ms = E.matrices(dv, list(src.key), refv, grid); Mm = E.matrices(dm, list(sm.key), refm, grid)
        th_s = src[col].to_numpy(); th_m = sm[col].to_numpy()
        js, jm = np.searchsorted(grid, th_s, side="left"), np.searchsorted(grid, th_m, side="left")
        Lfz_s = np.minimum(Ms[0][np.arange(len(src)), js], 3) / 3          # frozen loss on source (grid-rounded)
        Lfz_m = np.minimum(Mm[0][np.arange(len(sm)), jm], 3) / 3
        print(f"\n==== {nm} ====  source fold 1 n {len(src)} frozen risk {Lfz_s.mean():.4f} | MIMIC n {len(sm)} frozen risk {Lfz_m.mean():.4f}")
        res = {}
        up = np.array(sorted(set(sm.patient)))
        posv = {k: i for i, k in enumerate(Fv["keys"])}
        cls_all = np.vstack([Fv["cls"][[posv[k] for k in src.key]], Fm["cls"]]).astype(np.float64)
        mu = cls_all.mean(0); Vt = np.linalg.svd(cls_all - mu, full_matrices=False)[2]
        for kind in ["cls", "probe", "both"]:
            rows = []
            Xs = features(Fv, list(src.key), kind, (mu, Vt), src.view); Xm = features(Fm, list(sm.key), kind, (mu, Vt), sm.view)
            for rep in range(NREP):
                pa = set(rng.permutation(up)[:len(up) // 2]); a = sm.patient.isin(pa).to_numpy(); b = ~a
                ws, wb, auc = domain_weights(Xs, Xm[a], Xm[b], rng)
                jb = wcrc_index(Ms[0], ws, wb)
                Lw = np.minimum(Mm[0][np.where(b)[0], jb], 3) / 3
                rec_w = Mm[2][np.where(b)[0], jb].sum() / max(Mm[3][b].sum(), 1)
                j_or = E.crc_index(Mm[0][a][:, :-1], ALPHA)                    # oracle: labelled MIMIC half A
                Lo = np.minimum(Mm[0][b, j_or], 3) / 3
                ju = int(wcrc_index(Ms[0], np.ones(len(src)), np.ones(1))[0])            # sanity: w = 1 must equal plain CRC
                assert ju == E.crc_index(Ms[0][:, :-1], ALPHA), (ju, E.crc_index(Ms[0][:, :-1], ALPHA))
                Lu = np.minimum(Mm[0][b, ju], 3) / 3
                ess = ws.sum() ** 2 / (ws ** 2).sum()
                est = float((ws * Lfz_s).sum() / ws.sum())
                rows.append({"domain_auc": auc, "ess": ess, "risk_weighted": float(Lw.mean()), "recall_weighted": float(rec_w),
                             "abstain_share": float((jb == len(grid)).mean()), "risk_frozen_B": float(Lfz_m[b].mean()), "risk_unweighted_crc": float(Lu.mean()),
                             "risk_oracle": float(Lo.mean()), "recall_oracle": float(Mm[2][b, j_or].sum() / max(Mm[3][b].sum(), 1)),
                             "labelfree_estimate": est, "true_frozen": float(Lfz_m.mean())})
            R = pd.DataFrame(rows); res[kind] = R.mean().to_dict(); res[kind]["sd_risk_weighted"] = float(R.risk_weighted.std())
            r = res[kind]
            print(f"[{kind:>5}] domain AUROC {r['domain_auc']:.3f} | ESS {r['ess']:.0f} of {len(src)} | MIMIC-B risk: frozen-multi {r['risk_frozen_B']:.4f}, unweighted CRC {r['risk_unweighted_crc']:.4f}, "
                  f"weighted {r['risk_weighted']:.4f} (sd {r['sd_risk_weighted']:.4f}, recall {r['recall_weighted']:.3f}, abstain {r['abstain_share']:.3f}), "
                  f"oracle-labelled {r['risk_oracle']:.4f} (recall {r['recall_oracle']:.3f}) | label-free estimate of frozen risk "
                  f"{r['labelfree_estimate']:.4f} vs true {r['true_frozen']:.4f}", flush=True)
        out[nm] = res
    json.dump(out, open(D18 / "e20_wcrc.json", "w"), indent=1, default=float)
    print("E20_DONE")


if __name__ == "__main__":
    main()
