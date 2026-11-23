#!/usr/bin/env python3
"""Shared helpers for rounds 48+ (E23 to E26): final-claim tables, frozen admission sets, per-finding counts,
patient-cluster bootstrap. Aggregates only."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import e4a_pool as E4  # noqa: E402
import e5_anchor as A5  # noqa: E402
import e6_deep as D  # noqa: E402
import e8_final as F  # noqa: E402

CRIT = sorted(F.CRIT)
KEYF = ["pneumothorax", "nodule", "mass", "pleural effusion", "consolidation", "pulmonary edema", "fracture", "airspace opacity"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def load_cands_dir(dirname, split):
    """F.load_cands for an arbitrary candidate directory (work/<dirname>)."""
    d = HOME / "work" / dirname
    df = pd.read_parquet(d / "candidates.parquet")
    pr = pd.read_parquet(d / "verify_probe.parquet")
    df["probe_region"] = pr.probe_region.fillna(0.5).to_numpy(); df["probe_global"] = pr.probe_global.fillna(0.5).to_numpy()
    df["inv_region_rank"] = 1 / df.region_rank; df["inv_global_rank"] = 1 / df.global_rank
    df["from_vlm"] = df.from_vlm.astype(float)
    df["fold"] = [int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 4 for p in df.patient] if split == "valid" else 9
    return df


def final_claims(c, probe, tau, ref):
    """Resolve every candidate's side (threshold-free), relabel against the reference, and collapse candidates that map
    to the same stated claim (key, finding, side) with score = max over them. The admitted final set at threshold t is
    then exactly {final claims with score >= t}, identical to 'admit, then resolve', so a threshold calibrated on these
    labels certifies the findings that are actually stated."""
    c = c[["key", "finding", "lat", "score", "fold"] + [x for x in ["patient"] if x in c.columns]].copy()
    c["lat0"] = c["lat"]
    c["lat"] = [D.resolve(probe, k, f, l, tau) for k, f, l in zip(c.key, c.finding, c.lat)]
    c["label"] = [E4.matches((f, l), set(ref.get(k, {}))) for k, f, l in zip(c.key, c.finding, c.lat)]
    c = c.sort_values("score", ascending=False).drop_duplicates(["key", "finding", "lat"])
    return c.reset_index(drop=True)


def raw_claims(c, ref):
    """claims as stated by a source without side resolution (e.g. a generator's own draft), labelled."""
    c = c.drop_duplicates(["key", "finding", "lat"]).copy()
    c["label"] = [E4.matches((f, l), set(ref.get(k, {}))) for k, f, l in zip(c.key, c.finding, c.lat)]
    return c


def multi_thresholds(split, keys, gthr, t, tops):
    G = F.groups_of(split, keys, tops)
    th = pd.Series(-np.inf, index=keys)
    for gname, g in G.items():
        th = np.maximum(th, g.map(lambda v: gthr.get(f"{gname}={v}", t)).reindex(keys).fillna(t))
    return th


def calibrate(fin_cal, grid, split, keys_all, tops, alpha=F.ALPHA):
    """marginal CRC threshold + multi-group thresholds (n >= 40) on a calibration table of final claims."""
    t = float(A5.crc_threshold(fin_cal, F.fp_loss, alpha, grid))
    G = F.groups_of(split, keys_all, tops); g_thr = {}
    for gname, g in G.items():
        for val in g.unique():
            c = fin_cal[fin_cal.key.isin(set(g[g == val].index))]
            g_thr[f"{gname}={val}"] = float(A5.crc_threshold(c, F.fp_loss, alpha, grid)) if c.key.nunique() >= 40 else t
    return t, g_thr


def admit(fin, keys, th):
    e = fin[fin.key.isin(set(keys))]
    if np.isscalar(th):
        return e[e.score >= th]
    return e[e.score.to_numpy() >= e.key.map(th).to_numpy()]


def claim_summary(tag, adm, ref, keys, split=None, tops=None):
    ml, ms = D.metrics(adm, ref, keys), D.metrics(adm, ref, keys, strict=True)
    fp = pd.Series(A5.fp_per_study(adm.assign(score=1.0), 0.5, list(keys)), index=list(keys))
    loss = np.minimum(fp, 3) / 3
    out = {"recall": ml["recall"], "precision": ml["precision"], "strict_recall": ms["recall"], "strict_precision": ms["precision"],
           "false_per_report": ml["false_per_report"], "claims_per_report": ml["claims_per_report"], "risk": round(float(loss.mean()), 4)}
    if split and tops:
        G = F.groups_of(split, list(keys), tops)
        out["worst_subgroup"] = round(max(float(loss[g[g == v].index].mean()) for g in G.values() for v in g.unique() if (g == v).sum() >= 40), 4)
    print(f"  {tag:<26} recall {out['recall']:.3f} (strict {out['strict_recall']:.3f}) precision {out['precision']:.3f} "
          f"(strict {out['strict_precision']:.3f}) false/rpt {out['false_per_report']:.3f} claims/rpt {out['claims_per_report']:.2f} "
          f"risk {out['risk']:.4f}" + (f" worst-subgroup {out['worst_subgroup']:.4f}" if "worst_subgroup" in out else ""), flush=True)
    return out


def per_study_finding(adm, ref, keys, f):
    """per study: TP claims, FP claims, covered refs, ref count for finding f (lenient side matching)."""
    by = {k: [(a, l) for a, l in zip(g.finding, g.lat) if a == f] for k, g in adm[adm.finding == f].groupby("key")}
    tp, fp, cov, n = (np.zeros(len(keys)) for _ in range(4))
    for i, k in enumerate(keys):
        R = [r for r in ref.get(k, {}) if r[0] == f]; A = by.get(k, [])
        t = sum(E4.matches(a, R) for a in A)
        tp[i], fp[i] = t, len(A) - t
        cov[i] = sum(E4.matches(r, A) for r in R); n[i] = len(R)
    return tp, fp, cov, n


def patients(split, keys):
    man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key").reindex(list(keys))
    return man.patient.astype(str).to_numpy()


def cluster_ratio_diff(numa, numb, den_a, den_b, pat, reps=2000, seed=48):
    """patient-cluster bootstrap CI of (sum numb / sum den_b) - (sum numa / sum den_a)."""
    up, inv = np.unique(pat, return_inverse=True)
    A, B, Da, Db = (np.bincount(inv, x) for x in (numa, numb, den_a, den_b))
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))
        d.append((w @ B) / max(w @ Db, 1e-9) - (w @ A) / max(w @ Da, 1e-9))
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def write_reports(name, split, adm, keys, b0s=None):
    import e7_refine as R
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    gen = HOME / "work/gen"
    pd.DataFrame({"key": list(keys), "generated": [A5.realize(by.get(k, [])) for k in keys]}).to_parquet(gen / f"{name}_{split}.parquet", index=False)
    if b0s is not None:
        pd.DataFrame({"key": list(keys), "generated": [R.edit_report(k, by.get(k, []), b0s) for k in keys]}).to_parquet(
            gen / f"{name.replace('anchor', 'edit')}_{split}.parquet", index=False)


def load_sentences(genname):
    """sentence-level claims of a generator's reports (work/corpus_gen/<genname>), for evidence-gated editing."""
    rows = {}
    for p in sorted((HOME / f"work/corpus_gen/{genname}").glob("shard_*.parquet")):
        d = pd.read_parquet(p, columns=["key", "sid", "sentence", "findings_json"])
        for k, sid, s, js in zip(d.key, d.sid, d.sentence, d.findings_json):
            cl = [(E4.canon(str(f.get("finding", ""))), f.get("laterality") or "none", f.get("polarity")) for f in json.loads(js)]
            rows.setdefault(k, []).append((sid, s, cl))
    return {k: sorted(v) for k, v in rows.items()}
