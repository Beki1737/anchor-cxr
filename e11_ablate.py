#!/usr/bin/env python3
"""E11: what does each ANCHOR component buy, at a MATCHED certified risk? (~/venvs/anchor, small GPU for the probe)

Reviewer questions answered:
  Q1 is retrieval needed, or is ANCHOR just a classifier?   -> candidate-source ablation + PROBE-EXHAUSTIVE classifier
     (every vocabulary finding scored by the probe, side from the region resolver, no retrieval, no VLM)
  Q2 which evidence admits claims?                         -> scorer-feature ablation (votes only, probe only, VLM flag ...)
  Q3 what is the price of the guarantee?                   -> recall at alpha in {0.05, 0.10, 0.20} (risk-recall frontier)
Protocol identical to the frozen configuration: scorer fit on valid fold 0, CRC (capped false claims) on valid fold 1,
evaluation on valid folds 2-3 (selection data) and, POST HOC, on test (reported as ablation, never used to choose).
Also saves the frozen systems' admitted claims (key, finding, lat) for E14 (independent labeller). Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e5_anchor as A5  # noqa: E402
import e6_deep as D  # noqa: E402
import e8_final as F  # noqa: E402

OUT = HOME / "work/e11"
ALPHAS = [0.05, 0.10, 0.20]


def per_study(adm, ref, keys):
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    cov = np.array([sum(E4.matches(r, by.get(k, [])) for r in ref.get(k, {})) for k in keys], float)
    nref = np.array([len(ref.get(k, {})) for k in keys], float)
    ncl = np.array([len(by.get(k, [])) for k in keys], float)
    tp = np.array([sum(E4.matches(c, set(ref.get(k, {}))) for c in by.get(k, [])) for k in keys], float)
    return cov, nref, ncl, tp


def summ(ps):
    cov, nref, ncl, tp = ps
    fp = ncl - tp
    return {"recall": round(cov.sum() / max(nref.sum(), 1), 4), "precision": round(tp.sum() / max(ncl.sum(), 1), 4),
            "false_per_report": round(fp.mean(), 3), "risk": round(float((np.minimum(fp, 3) / 3).mean()), 4)}


def boot_recall_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); N = len(a[0]); out = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        out.append(a[0][i].sum() / max(a[1][i].sum(), 1) - b[0][i].sum() / max(b[1][i].sum(), 1))
    d = a[0].sum() / a[1].sum() - b[0].sum() / b[1].sum()
    return round(float(d), 4), [round(float(v), 4) for v in np.percentile(out, [2.5, 97.5])]


def drop_const(df, feats):
    tr = df[df.fold == 0]
    return [f for f in feats if tr[f].astype(float).std() > 1e-9]


def exhaustive(split, keys, probe, ref, tau, man):
    """Probe-only classifier candidates: every vocabulary finding for every study."""
    rows = []
    vis = probe["Ar"] > 0
    for k in keys:
        i = probe["pos"].get(k)
        if i is None:
            continue
        for f, j in probe["F"].items():
            pr = probe["Pr"][i, vis[i], j]
            lat = D.resolve(probe, k, f, "none", tau)
            rows.append((k, man.get(k, k), f, lat, float(probe["Pg"][i, j]), float(pr.max()) if pr.size else float(probe["Pg"][i, j])))
    d = pd.DataFrame(rows, columns=["key", "patient", "finding", "lat", "probe_global", "probe_region"])
    d["label"] = [E4.matches((f, l), set(ref.get(k, {}))) for k, f, l in zip(d.key, d.finding, d.lat)]
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(F.OUT / "freeze.json"))
    tau, delta = fz["tau"], fz["delta"]
    V = F.load_cands("valid"); T = F.load_cands("test")
    refV = E4.load_claims(HOME / "work/corpus_v2/valid"); refT = E4.load_claims(HOME / "work/corpus_v2/test")
    kV = sorted(V[V.fold >= 2].key.unique()); kT = sorted(T.key.unique())
    print(f"valid studies with candidates: fold0 {V[V.fold == 0].key.nunique()} | fold1 {V[V.fold == 1].key.nunique()} | "
          f"eval folds 2-3 {len(kV)} | test {len(kT)}")
    src = lambda d: {"V": d.from_vlm > 0, "R": d.region_rank < 99, "G": d.global_rank < 99}
    res = {"configs": {}}
    store = {}

    def run(name, dv, dt, feats, keep=False):
        feats = drop_const(dv, feats)
        m = F.fit(dv, feats)
        dv = dv.assign(score=F.apply_scorer(dv, m, delta)); dt = dt.assign(score=F.apply_scorer(dt, m, delta))
        ev = dv[dv.fold >= 2]
        r = {"feats": feats, "n_candidates_per_study_test": round(len(dt) / len(kT), 2),
             "claim_auroc_valid_eval": round(E4.auroc(ev.score, ev.label), 4), "claim_auroc_test": round(E4.auroc(dt.score, dt.label), 4)}
        grid = np.quantile(dv.score, np.linspace(1, 0, 500)); cal = dv[dv.fold == 1]
        for al in ALPHAS:
            t = A5.crc_threshold(cal, F.fp_loss, al, grid)
            psV, psT = per_study(ev[ev.score >= t], refV, kV), per_study(dt[dt.score >= t], refT, kT)
            r[str(al)] = {"valid_eval": summ(psV), "test": summ(psT)}
            if keep and al == 0.10:
                store[name] = (psV, psT)
        res["configs"][name] = r
        a = r["0.1"]
        print(f"  {name:<34} AUROC {r['claim_auroc_valid_eval']:.3f}/{r['claim_auroc_test']:.3f} | alpha .10 recall "
              f"{a['valid_eval']['recall']:.3f}/{a['test']['recall']:.3f} prec {a['valid_eval']['precision']:.3f}/{a['test']['precision']:.3f} "
              f"risk {a['valid_eval']['risk']:.3f}/{a['test']['risk']:.3f} | recall@.05 {r['0.05']['test']['recall']:.3f} @.20 {r['0.2']['test']['recall']:.3f}", flush=True)

    # reference row: raw MedGemma claims
    for nm, d, keys, ref in [("valid_eval", V[(V.fold >= 2) & (V.from_vlm > 0)], kV, refV), ("test", T[T.from_vlm > 0], kT, refT)]:
        res.setdefault("b0_raw", {})[nm] = summ(per_study(d, ref, keys))
    store["B0 raw (no admission)"] = (per_study(V[(V.fold >= 2) & (V.from_vlm > 0)], refV, kV), per_study(T[T.from_vlm > 0], refT, kT))
    print(f"\nMedGemma B0 raw claims (valid-eval / test): {res['b0_raw']['valid_eval']} / {res['b0_raw']['test']}")
    print("\n#### Q1 candidate sources (full scorer; numbers valid-eval / test) ####")
    for combo in ["V", "VG", "VR", "VRG", "G", "R", "RG"]:
        mv = np.logical_or.reduce([src(V)[c] for c in combo]); mt = np.logical_or.reduce([src(T)[c] for c in combo])
        run(f"cands {combo}", V[mv].copy(), T[mt].copy(), A5.FEATS, keep=combo in ("V", "VRG", "RG"))
    probeV = F.probe_arrays("valid", V.key.unique()); probeT = F.probe_arrays("test", T.key.unique())
    manV = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key").patient.astype(str).to_dict()
    manT = pd.read_parquet(HOME / "work/manifest/rex_test.parquet").set_index("key").patient.astype(str).to_dict()
    XV = exhaustive("valid", sorted(V.key.unique()), probeV, refV, tau, manV)
    XV["fold"] = XV.key.map(V.drop_duplicates("key").set_index("key").fold)
    XT = exhaustive("test", kT, probeT, refT, tau, manT); XT["fold"] = 9
    print(f"  probe-exhaustive candidates: {len(XT) / len(kT):.1f} per study (vocabulary {len(probeT['F'])}); oracle recall test "
          f"{summ(per_study(XT[XT.label], refT, kT))['recall']:.3f} vs ANCHOR pool {summ(per_study(T[T.label], refT, kT))['recall']:.3f}")
    run("probe-exhaustive classifier", XV, XT, ["probe_global", "probe_region"], keep=True)
    print("\n#### Q2 admission evidence (all candidates) ####")
    for nm, fs in [("votes+ranks only (retrieval consensus)", ["inv_region_rank", "inv_global_rank", "votes_global", "votes_region_max", "votes_region_mean"]),
                   ("probe only", ["probe_region", "probe_global"]),
                   ("VLM flag + votes (no image evidence)", ["from_vlm", "inv_region_rank", "inv_global_rank", "votes_global", "votes_region_max", "votes_region_mean"]),
                   ("full minus VLM flag", [f for f in A5.FEATS if f != "from_vlm"]),
                   ("full (ANCHOR)", A5.FEATS)]:
        run(nm, V.copy(), T.copy(), fs)
    print("\n#### paired recall differences at alpha 0.10 (bootstrap over studies) ####")
    base = "cands VRG"
    for other in ["probe-exhaustive classifier", "cands V", "cands RG", "B0 raw (no admission)"]:
        dv = boot_recall_diff(store[base][0], store[other][0]); dt = boot_recall_diff(store[base][1], store[other][1])
        res.setdefault("paired", {})[f"{base} - {other}"] = {"valid_eval": dv, "test": dt}
        print(f"  ANCHOR minus {other:<30} valid-eval {dv[0]:+.4f} {dv[1]} | test {dt[0]:+.4f} {dt[1]}")
    json.dump(res, open(OUT / "e11_results.json", "w"), indent=1)
    # admitted claims of the frozen systems (for E14)
    for split, d, keys in [("test", T, kT)]:
        d = d.assign(score=F.apply_scorer(d, fz["scorer_full"], delta))
        vf = d[(d.region_rank < 99) | (d.global_rank < 99)].copy(); vf["score"] = F.apply_scorer(vf, fz["scorer_vlmfree"], delta)
        ref = refT; probe = probeT
        G = F.groups_of(split, keys, fz["group_tops"])
        th = pd.Series(-np.inf, index=keys)
        for gname, g in G.items():
            th = np.maximum(th, g.map(lambda v: fz["group_thresholds"].get(f"{gname}={v}", fz["t_full"])).reindex(keys).fillna(fz["t_full"]))
        sysd = {"b0": d[d.from_vlm > 0], "anchor": F.resolve_sides(d[d.score >= fz["t_full"]], probe, tau, ref),
                "multi": F.resolve_sides(d[d.score >= d.key.map(th)], probe, tau, ref),
                "vlmfree": F.resolve_sides(vf[vf.score >= fz["t_vlmfree"]], probe, tau, ref)}
        pd.concat([s[["key", "finding", "lat"]].assign(system=n) for n, s in sysd.items()]).to_parquet(OUT / f"admitted_{split}.parquet", index=False)
        print(f"saved admitted claims ({split}) for E14: " + ", ".join(f"{n} {len(s)}" for n, s in sysd.items()))
    print("E11_DONE")


if __name__ == "__main__":
    main()
