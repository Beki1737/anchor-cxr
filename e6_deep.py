#!/usr/bin/env python3
"""E6: deep-dive experiments on the E5 pipeline (valid primary cohort, same 4 patient folds; ~/venvs/anchor, 1 GPU).
Folds: 0 = fit scorer / tune, 1 = conformal calibration, 2-3 = evaluation (never used for fitting or calibration).

  H6 side    anatomy is needed for WHERE: region probe discriminates right vs left for lateral findings; global probe
             cannot by construction. Side resolver re-assigns the side of admitted claims from region evidence (neighbour
             claims carry the neighbour's side). Strict (side-aware) claim metrics before/after.
  H7 pool    VLM-free ANCHOR: retrieval + probe only. How much does the generator still add?
  H8 prior   the patient's own prior report as an extra candidate channel (studies with a strict prior).
  H9 shift   conformal guarantee per subgroup (view PA/AP, institution, manufacturer): marginal vs Mondrian CRC.
  H10 triage conformal autonomous NORMAL reporting: largest set of studies reported as normal with a guaranteed
             bound on missed actionable abnormality (joint risk), plus coverage.
  per-finding recall / precision, B0 vs ANCHOR.
Writes extra report variants (realizer) for judging: e5anchorsr_fp30, e5anchorsr_any30, e5poolonly_fp30, e5prior_fp30.
Aggregates only.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e4b_verify as E4B  # noqa: E402
import e5_anchor as A5  # noqa: E402
from strata import study_strata, KEY  # noqa: E402

OUT = HOME / "work/e6"
LATF = {"pleural effusion", "pneumothorax", "consolidation", "airspace opacity", "atelectasis", "nodule", "mass",
        "fracture", "elevated hemidiaphragm", "interstitial opacity", "calcified granuloma"}
RIGHT = [r for r in E4.RN if r.startswith("R_")]
LEFT = [r for r in E4.RN if r.startswith("L_")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def side_strict(c, r):
    if r in ("right", "left", "bilateral"):
        return c == r
    return True


def strict_match(c, R):
    return any(c[0] == r[0] and side_strict(c[1], r[1]) for r in R)


def metrics(adm, ref, keys, strict=False):
    keys = list(keys)
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    m = strict_match if strict else E4.matches
    cov = sum(sum(m(r, by.get(k, [])) for r in ref.get(k, {})) for k in keys)
    nref = sum(len(ref.get(k, {})) for k in keys)
    tp = sum(sum(m(c, set(ref.get(k, {}))) for c in by.get(k, [])) for k in keys)
    n = sum(len(by.get(k, [])) for k in keys)
    return {"recall": round(cov / max(nref, 1), 4), "precision": round(tp / max(n, 1), 4),
            "false_per_report": round((n - tp) / max(len(keys), 1), 3), "claims_per_report": round(n / max(len(keys), 1), 3)}


def fit_score(df, feats):
    X = df[feats].astype(float).to_numpy(); y = df.label.astype(float).to_numpy()
    tr = (df.fold == 0).to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w, b = E4B.logreg_fit((X[tr] - mu) / sd, y[tr])
    return 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b))), dict(zip(feats, np.round(w, 3)))


def crc_admit(df, kind, alpha):
    grid = np.quantile(df.score, np.linspace(1, 0, 400))
    loss = {"any": lambda d, t: float((A5.fp_per_study(d, t, sorted(d.key.unique())) >= 1).mean()),
            "fp": lambda d, t: float((np.minimum(A5.fp_per_study(d, t, sorted(d.key.unique())), 3) / 3).mean())}[kind]
    cal = df[df.fold == 1]
    t = A5.crc_threshold(cal, loss, alpha, grid)
    ev = df[df.fold >= 2]
    return ev[ev.score >= t], t, loss(ev, t)


def write_gen(name, adm, keys):
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    pd.DataFrame({"key": keys, "generated": [A5.realize(by.get(k, [])) for k in keys]}).to_parquet(
        HOME / "work/gen" / f"{name}_valid.parquet", index=False)


def prepare():
    import torch
    import e4c_probe as P
    df = pd.read_parquet(A5.OUT / "candidates.parquet")
    pr = pd.read_parquet(A5.OUT / "verify_probe.parquet")
    df["probe_region"] = pr.probe_region.fillna(0.5).to_numpy(); df["probe_global"] = pr.probe_global.fillna(0.5).to_numpy()
    df["inv_region_rank"] = 1 / df.region_rank; df["inv_global_rank"] = 1 / df.global_rank
    df["fold"] = [int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 4 for p in df.patient]
    df["score"], w = fit_score(df, A5.FEATS)
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    man = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key")
    reg, glo, vocab = P.load_probe(torch)
    keys, Em, Ar, Cl = P.load_split("valid", sorted(set(man.index) & set(df.key)))
    Pr, Pg = P.predict(torch, reg, glo, Em, Cl)
    probe = {"pos": {k: i for i, k in enumerate(keys)}, "F": {f: j for j, f in enumerate(vocab)}, "Pr": Pr, "Pg": Pg, "Ar": Ar}
    return df, ref, man, probe


def side_scores(probe, key, f):
    i, j = probe["pos"].get(key), probe["F"].get(f)
    if i is None or j is None:
        return None
    ar = probe["Ar"][i]
    sr = max([probe["Pr"][i, E4.RN.index(r), j] for r in RIGHT if ar[E4.RN.index(r)] > 0] or [np.nan])
    sl = max([probe["Pr"][i, E4.RN.index(r), j] for r in LEFT if ar[E4.RN.index(r)] > 0] or [np.nan])
    return (sr, sl) if sr == sr and sl == sl else None


def resolve(probe, key, f, lat, tau):
    if f not in LATF:
        return lat
    s = side_scores(probe, key, f)
    if s is None:
        return lat
    sr, sl = s
    if min(sr, sl) >= tau:
        return "bilateral"
    return "right" if sr >= sl else "left"


def h6_side(df, ref, probe, res):
    print("\n#### H6 side: anatomy for laterality ####")
    folds = df.drop_duplicates("key").set_index("key").fold
    rows = []
    for k, cl in ref.items():
        if k not in folds.index:
            continue
        for (f, lat) in cl:
            if f in LATF and lat in ("right", "left", "bilateral"):
                s = side_scores(probe, k, f)
                if s is not None:
                    rows.append((k, f, lat, s[0], s[1], folds[k]))
    S = pd.DataFrame(rows, columns=["key", "f", "lat", "sr", "sl", "fold"])
    lr = S[S.lat != "bilateral"]
    au = E4.auroc(lr.sr - lr.sl, lr.lat == "right")
    per = {f: round(E4.auroc(g.sr - g.sl, g.lat == "right"), 3) for f, g in lr.groupby("f") if (g.lat == "right").sum() >= 30 and (g.lat == "left").sum() >= 30}
    print(f"right-vs-left discrimination of sided reference findings (n={len(lr)}): region probe AUROC {au:.3f} | global probe 0.500 by construction")
    print("  per finding:", per)
    best = (0, 0.5)
    for tau in np.arange(0.2, 0.95, 0.05):                      # tune bilateral threshold on fold 0 only
        t0 = S[S.fold == 0]
        pred = np.where(np.minimum(t0.sr, t0.sl) >= tau, "bilateral", np.where(t0.sr >= t0.sl, "right", "left"))
        acc = float((pred == t0.lat).mean())
        best = max(best, (acc, tau))
    tau = best[1]
    ev = S[S.fold >= 2]
    pred = np.where(np.minimum(ev.sr, ev.sl) >= tau, "bilateral", np.where(ev.sr >= ev.sl, "right", "left"))
    acc = wilson(int((pred == ev.lat).sum()), len(ev))
    print(f"side assignment accuracy on eval sided findings (3-way right/left/bilateral, tau={tau:.2f} tuned on fold 0): "
          f"{acc[0]:.3f} [{acc[1]:.3f}, {acc[2]:.3f}] | majority-class baseline {ev.lat.value_counts(normalize=True).max():.3f}")
    res["h6"] = {"lr_auroc": round(au, 4), "per_finding": per, "tau": round(float(tau), 2), "side_acc_eval": round(acc[0], 4)}
    eval_keys = sorted(df[df.fold >= 2].key.unique())
    for kind, alpha, name in [("fp", 0.10, "fp30"), ("any", 0.30, "any30")]:
        adm, t, risk = crc_admit(df, kind, alpha)
        sr_adm = adm.copy()
        sr_adm["lat"] = [resolve(probe, k, f, l, tau) for k, f, l in zip(adm.key, adm.finding, adm.lat)]
        sr_adm = sr_adm.drop_duplicates(["key", "finding", "lat"])
        before, after = metrics(adm, ref, eval_keys, strict=True), metrics(sr_adm, ref, eval_keys, strict=True)
        lb, la = metrics(adm, ref, eval_keys), metrics(sr_adm, ref, eval_keys)
        print(f"  ANCHOR {name}: STRICT side-aware recall {before['recall']:.3f} -> {after['recall']:.3f}, precision "
              f"{before['precision']:.3f} -> {after['precision']:.3f} | lenient recall {lb['recall']:.3f} -> {la['recall']:.3f}")
        res["h6"][name] = {"strict_before": before, "strict_after": after, "lenient_before": lb, "lenient_after": la}
        write_gen(f"e5anchorsr_{name}", sr_adm, eval_keys)
    b0 = df[(df.fold >= 2) & df.from_vlm]
    res["h6"]["b0_strict"] = metrics(b0, ref, eval_keys, strict=True)
    print(f"  MedGemma B0: STRICT recall {res['h6']['b0_strict']['recall']:.3f}, precision {res['h6']['b0_strict']['precision']:.3f}")


def h7_pool(df, ref, res):
    print("\n#### H7 VLM-free ANCHOR (retrieval + probe only) ####")
    eval_keys = sorted(df[df.fold >= 2].key.unique())
    d = df[(df.region_rank < 99) | (df.global_rank < 99)].copy()
    feats = [f for f in A5.FEATS if f != "from_vlm"]
    d["score"], w = fit_score(d, feats)
    for kind, alpha, name in [("fp", 0.10, "fp30"), ("any", 0.30, "any30")]:
        adm, t, risk = crc_admit(d, kind, alpha)
        full, _, _ = crc_admit(df, kind, alpha)
        m, mf = metrics(adm, ref, eval_keys), metrics(full, ref, eval_keys)
        print(f"  {name}: VLM-free recall {m['recall']:.3f} precision {m['precision']:.3f} false/rpt {m['false_per_report']:.3f} "
              f"| full ANCHOR recall {mf['recall']:.3f} precision {mf['precision']:.3f} false/rpt {mf['false_per_report']:.3f} | eval risk {risk:.3f} (alpha {alpha})")
        res.setdefault("h7", {})[name] = {"vlm_free": m, "full": mf, "eval_risk": round(risk, 4)}
        if name == "fp30":
            write_gen("e5poolonly_fp30", adm, eval_keys)


def h8_prior(df, ref, man, probe, res):
    print("\n#### H8 prior-report channel ####")
    eval_keys = sorted(df[df.fold >= 2].key.unique())
    pk = man.prior_key.reindex(sorted(df.key.unique()))
    has = pk.dropna()
    have = set(has.index)
    exist = set(zip(df.key, df.finding, df.lat))
    new_rows = []
    df = df.copy(); df["in_prior"] = 0.0
    prior_claims = {k: set(ref.get(p, {})) for k, p in has.items()}
    idx = {(k, f, l): i for i, (k, f, l) in enumerate(zip(df.key, df.finding, df.lat))}
    for k, pcs in prior_claims.items():
        for (f, l) in pcs:
            if (k, f, l) in idx:
                df.iat[idx[(k, f, l)], df.columns.get_loc("in_prior")] = 1.0
            else:
                i, j = probe["pos"].get(k), probe["F"].get(f)
                pg = float(probe["Pg"][i, j]) if i is not None and j is not None else 0.5
                regs = E4B.claim_regions((f, l), ref.get(has[k], {}).get((f, l), set()))
                ri = [E4.RN.index(x) for x in regs if i is not None and probe["Ar"][i, E4.RN.index(x)] > 0]
                prr = float(probe["Pr"][i, ri, j].max()) if ri and j is not None else pg
                new_rows.append({"key": k, "patient": df.loc[df.key == k, "patient"].iloc[0] if (df.key == k).any() else k,
                                 "finding": f, "lat": l, "regions": ",".join(regs), "from_vlm": False, "region_rank": 99,
                                 "global_rank": 99, "votes_global": 0.0, "votes_region_max": 0.0, "votes_region_mean": 0.0,
                                 "label": E4.matches((f, l), set(ref.get(k, {}))), "stratum": None, "probe_region": prr,
                                 "probe_global": pg, "inv_region_rank": 1 / 99, "inv_global_rank": 1 / 99,
                                 "fold": int(hashlib.md5(str(df.loc[df.key == k, "patient"].iloc[0] if (df.key == k).any() else k).encode()).hexdigest(), 16) % 4,
                                 "in_prior": 1.0})
    d = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    stab = [E4.matches(c, set(ref.get(k, {}))) for k, pcs in prior_claims.items() for c in pcs]
    print(f"studies with a strict prior: {len(have)} ({len(have & set(eval_keys))} in eval) | prior positive claims {len(stab)}, "
          f"still true now {np.mean(stab):.3f} | new candidates added {len(new_rows)}")
    ek = sorted(have & set(eval_keys))
    base_cand = df[df.key.isin(ek)]; pri_cand = d[d.key.isin(ek)]
    nref = sum(len(ref.get(k, {})) for k in ek)
    oc = lambda c: E4B.covered(c[c.label], ref) / max(nref, 1)
    print(f"  pool ceiling on eval studies with prior: {oc(base_cand):.3f} -> {oc(pri_cand):.3f}")
    d["score"], w = fit_score(d, A5.FEATS + ["in_prior"])
    adm_p, _, risk_p = crc_admit(d, "fp", 0.10)
    adm_b, _, _ = crc_admit(df, "fp", 0.10)
    mp, mb = metrics(adm_p, ref, ek), metrics(adm_b, ref, ek)
    print(f"  fp30 on eval studies with prior: recall {mb['recall']:.3f} -> {mp['recall']:.3f}, precision {mb['precision']:.3f} -> "
          f"{mp['precision']:.3f}, false/rpt {mb['false_per_report']:.3f} -> {mp['false_per_report']:.3f} | in_prior weight {w['in_prior']:+.2f} | eval risk {risk_p:.3f}")
    res["h8"] = {"n_with_prior_eval": len(ek), "stability": round(float(np.mean(stab)), 4), "ceiling_before": round(oc(base_cand), 4),
                 "ceiling_after": round(oc(pri_cand), 4), "base": mb, "with_prior": mp, "w_in_prior": float(w["in_prior"])}
    write_gen("e5prior_fp30", adm_p, eval_keys)


def h9_shift(df, man, res):
    print("\n#### H9 conformal guarantee per subgroup (marginal vs Mondrian CRC) ####")
    meta = man.reindex(sorted(df.key.unique()))
    groupers = {"view": meta.frontal_view.astype(str)}
    for col, top in [("institution", 6), ("manufacturer", 4)]:
        v = meta[col].astype(str)
        keep = v[v.index.isin(df[df.fold >= 2].key)].value_counts().index[:top]
        groupers[col] = v.where(v.isin(keep), "other")
    res["h9"] = {}
    for kind, alpha in [("any", 0.30), ("fp", 0.10)]:
        lossf = {"any": lambda d, t: (A5.fp_per_study(d, t, sorted(d.key.unique())) >= 1).astype(float),
                 "fp": lambda d, t: np.minimum(A5.fp_per_study(d, t, sorted(d.key.unique())), 3) / 3}[kind]
        grid = np.quantile(df.score, np.linspace(1, 0, 400))
        cal, ev = df[df.fold == 1], df[df.fold >= 2]
        t_marg = A5.crc_threshold(cal, lambda d, t: float(lossf(d, t).mean()), alpha, grid)
        for gname, g in groupers.items():
            print(f"  loss={kind} alpha={alpha} grouping={gname}")
            out = {}
            for val in sorted(g.unique()):
                ks = set(g[g == val].index)
                cg, eg = cal[cal.key.isin(ks)], ev[ev.key.isin(ks)]
                if eg.key.nunique() < 40 or cg.key.nunique() < 40:
                    continue
                rm = lossf(eg, t_marg); t_g = A5.crc_threshold(cg, lambda d, t: float(lossf(d, t).mean()), alpha, grid)
                rg = lossf(eg, t_g)
                se = rm.std() / np.sqrt(len(rm))
                flag = "VIOLATED" if rm.mean() - 1.96 * se > alpha else ("over" if rm.mean() > alpha else "ok")
                out[val] = {"n_eval": len(rm), "marginal_risk": round(float(rm.mean()), 4), "mondrian_risk": round(float(rg.mean()), 4), "flag": flag}
                print(f"    {val:<10} n={len(rm):>5}  marginal {rm.mean():.3f} ({flag})  mondrian {rg.mean():.3f}")
            res["h9"][f"{kind}|{gname}"] = out


def h10_triage(df, ref, res):
    print("\n#### H10 conformal autonomous normal reporting ####")
    st = study_strata(HOME / "work/corpus_v2/valid")
    s = df.groupby("key").score.max()
    f = df.drop_duplicates("key").set_index("key").fold
    T = pd.DataFrame({"smax": s, "fold": f}).join(st[["abn", "key_abn"]], how="inner")
    res["h10"] = {"auroc_key_abnormal": round(E4.auroc(T.smax, T.key_abn), 4), "auroc_any_abnormal": round(E4.auroc(T.smax, T.abn), 4)}
    print(f"study-level abnormality AUROC (max claim score): key findings {res['h10']['auroc_key_abnormal']:.3f} | any {res['h10']['auroc_any_abnormal']:.3f}")
    cal, ev = T[T.fold == 1], T[T.fold >= 2]
    grid = np.quantile(T.smax, np.linspace(0, 1, 500))
    for target in ["key_abn", "abn"]:
        for beta in [0.01, 0.02, 0.05]:
            n = len(cal); best = -np.inf
            for t in grid:                                     # ascending: more studies auto-normal as t grows
                r = float(((cal.smax < t) & cal[target]).mean())
                if n / (n + 1) * r + 1 / (n + 1) <= beta:
                    best = t
                else:
                    break
            auto = ev.smax < best
            joint = float((auto & ev[target]).mean()); cov = float(auto.mean())
            cond = float(ev[target][auto].mean()) if auto.any() else float("nan")
            normal_caught = float(auto[~ev.abn].mean())
            res["h10"][f"{target}@{beta}"] = {"coverage": round(cov, 4), "joint_risk": round(joint, 4), "miss_rate_among_auto": round(cond, 4),
                                              "share_of_true_normals_auto": round(normal_caught, 4)}
            print(f"  missed={target:<7} beta={beta:.2f}: auto-normal {cov:.3f} of studies | joint risk {joint:.4f} "
                  f"({'OK' if joint <= beta else 'VIOLATED'}) | abnormal among auto-normal {cond:.3f} | true normals captured {normal_caught:.3f}")


def per_finding(df, ref, res):
    print("\n#### per-finding recall / precision on eval (lenient side) ####")
    ek = sorted(df[df.fold >= 2].key.unique())
    adm, _, _ = crc_admit(df, "fp", 0.10)
    b0 = df[(df.fold >= 2) & df.from_vlm]
    cnt = pd.Series([f for k in ek for (f, l) in ref.get(k, {})]).value_counts()
    out = {}
    print(f"{'finding':<24}{'n_ref':>6}  {'B0 rec/prec':>14}  {'ANCHOR fp30 rec/prec':>22}")
    for f in cnt.index[:16]:
        r = {k: {c for c in ref.get(k, {}) if c[0] == f} for k in ek}
        mb = metrics(b0[b0.finding == f], r, ek); ma = metrics(adm[adm.finding == f], r, ek)
        out[f] = {"n": int(cnt[f]), "b0": mb, "anchor": ma}
        print(f"{f:<24}{cnt[f]:>6}  {mb['recall']:.2f}/{mb['precision']:.2f}{'':>5}  {ma['recall']:.2f}/{ma['precision']:.2f}")
    res["per_finding"] = out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="h6,h7,h8,h9,h10,pf")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    df, ref, man, probe = prepare()
    print(f"E6 on {df.key.nunique()} valid studies, {len(df)} candidates; eval studies {df[df.fold >= 2].key.nunique()}")
    res = {}
    todo = a.only.split(",")
    if "h6" in todo: h6_side(df, ref, probe, res)
    if "h7" in todo: h7_pool(df, ref, res)
    if "h8" in todo: h8_prior(df, ref, man, probe, res)
    if "h9" in todo: h9_shift(df, man, res)
    if "h10" in todo: h10_triage(df, ref, res)
    if "pf" in todo: per_finding(df, ref, res)
    json.dump(res, open(OUT / "e6_results.json", "w"), indent=1, default=str)
    print("E6_DONE")


if __name__ == "__main__":
    main()
