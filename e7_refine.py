#!/usr/bin/env python3
"""E7: refinements targeting the weaknesses found in E5/E6 (valid, same 4 patient folds; ~/venvs/anchor, 1 GPU).

  H12 edit     GREEN/RadGraph/RaTEScore punish the terse realizer (B0 claims through the realizer lose 0.14 GREEN).
               ANCHOR-edit = evidence-gated EDITING of MedGemma's own report: keep its sentences whose positive claims
               are all admitted (and whose negatives do not contradict an admitted claim), drop the rest, append admitted
               pool claims. Keeps the VLM's fluency and pertinent negatives, certified content.
  H13 sidexp   side as part of the candidate: every lateralisable claim expanded to right / left / bilateral variants with
               region-probe side evidence + neighbour side agreement as features (replaces the post-hoc resolver).
  H11 focal    ANCHOR suppresses rare focal findings (nodule, fracture recall -> 0). Finding-group score offset tuned on
               fold 0 to maximise clinically weighted recall, then CRC on fold 1 as usual.
  H9b multi    simultaneous validity: per study, the strictest of its view / institution / manufacturer Mondrian thresholds.
  H10b triage  study-level normality score learned from probe heads (fold 0) for autonomous normal reporting.
Writes report variants: e7edit_fp30, e7edit_any15, e7sidexp_fp30, e7focal_fp30, e7multi_any30. Aggregates only.
"""
import argparse
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
import e6_deep as D  # noqa: E402
from strata import study_strata, KEY  # noqa: E402

OUT = HOME / "work/e7"
SIG = {"pneumothorax": 1.0, "pleural effusion": 0.5, "consolidation": 0.5, "airspace opacity": 0.5, "pulmonary edema": 0.5,
       "nodule": 0.5, "mass": 0.5, "fracture": 0.5, "mediastinal widening": 0.5, "cardiomegaly": 0.5, "interstitial opacity": 0.5,
       "atelectasis": 0.25, "hyperinflation": 0.25, "elevated hemidiaphragm": 0.25, "scoliosis": 0.25, "calcified granuloma": 0.25,
       "hiatal hernia": 0.25, "degenerative changes": 0.0, "aortic calcification": 0.0, "postsurgical change": 0.0,
       "support device": 0.0, "osteopenia": 0.0}
FOCAL = {"nodule", "mass", "fracture", "pneumothorax", "calcified granuloma"}


def wrecall(adm, ref, keys):
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    num = den = 0.0
    for k in keys:
        for r in ref.get(k, {}):
            w = SIG.get(r[0], 0.25); den += w
            if E4.matches(r, by.get(k, [])):
                num += w
    return num / max(den, 1e-9)


def load_b0_sentences():
    rows = {}
    for p in sorted((HOME / "work/corpus_gen/b0_medgemma4b_valid").glob("shard_*.parquet")):
        d = pd.read_parquet(p, columns=["key", "sid", "sentence", "findings_json"])
        for k, sid, s, js in zip(d.key, d.sid, d.sentence, d.findings_json):
            cl = [(E4.canon(str(f.get("finding", ""))), f.get("laterality") or "none", f.get("polarity")) for f in json.loads(js)]
            rows.setdefault(k, []).append((sid, s, cl))
    return {k: sorted(v) for k, v in rows.items()}


def edit_report(key, admitted, b0s):
    A = set(admitted)
    supported = lambda f, l: any(a[0] == f and (a[1] == l or l == "none") for a in A)
    kept, kept_pos = [], []
    for sid, s, cl in b0s.get(key, []):
        pos = [(f, l) for f, l, p in cl if p in ("present", "uncertain")]
        neg = [(f, l) for f, l, p in cl if p == "absent"]
        if any(not supported(f, l) for f, l in pos):
            continue                                            # a positive claim the evidence did not admit -> drop
        if any(any(f == a[0] and E4.side_ok(l, a[1]) for a in A) for f, l in neg):
            continue                                            # a negative that contradicts an admitted finding -> drop
        kept.append(s.strip()); kept_pos += pos
    extra = [a for a in sorted(A) if not any(f == a[0] and (l == a[1] or "none" in (l, a[1])) for f, l in kept_pos)]
    add = [x.strip() for x in A5.realize(extra).split(".") if x.strip().startswith("There is")]
    text = " ".join(kept + [x + "." for x in add]).strip()
    return text if text else A5.realize(sorted(A))


def h12_edit(df, ref, probe, tau, b0s, res):
    print("\n#### H12 ANCHOR-edit (evidence-gated editing of the VLM report) ####")
    ek = sorted(df[df.fold >= 2].key.unique())
    for kind, alpha, name in [("fp", 0.10, "fp30"), ("any", 0.15, "any15")]:
        adm, t, risk = D.crc_admit(df, kind, alpha)
        adm = adm.copy()
        adm["lat"] = [D.resolve(probe, k, f, l, tau) for k, f, l in zip(adm.key, adm.finding, adm.lat)]
        adm = adm.drop_duplicates(["key", "finding", "lat"])
        by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
        texts = [edit_report(k, by.get(k, []), b0s) for k in ek]
        pd.DataFrame({"key": ek, "generated": texts}).to_parquet(HOME / "work/gen" / f"e7edit_{name}_valid.parquet", index=False)
        kept = np.mean([len(t.split()) for t in texts])
        print(f"  {name}: eval risk {risk:.3f} (alpha {alpha}) | mean words {kept:.1f} | example length B0-like, content certified")
        res.setdefault("h12", {})[name] = {"eval_risk": round(risk, 4), "mean_words": round(float(kept), 1)}


def h13_sidexp(df, ref, probe, res):
    print("\n#### H13 side-expanded candidates ####")
    ek = sorted(df[df.fold >= 2].key.unique())
    num = ["from_vlm", "inv_region_rank", "inv_global_rank", "votes_global", "votes_region_max", "votes_region_mean",
           "probe_region", "probe_global"]
    rows = []
    lat_mask = df.finding.isin(D.LATF)
    base = df[~lat_mask].copy()
    base["s_side"] = base.probe_region; base["side_margin"] = 0.0; base["nb_side"] = 1.0; base["label_strict"] = [
        D.strict_match((f, l), set(ref.get(k, {}))) for k, f, l in zip(base.key, base.finding, base.lat)]
    for (k, f), g in df[lat_mask].groupby(["key", "finding"]):
        s = D.side_scores(probe, k, f)
        agg = g[num].max()
        sides = set(g.lat)
        for side in ["right", "left", "bilateral"]:
            if s is None:
                if side not in sides:
                    continue
                ss, mg = float(g.probe_region.max()), 0.0
            else:
                sr, sl = s
                ss = {"right": sr, "left": sl, "bilateral": min(sr, sl)}[side]
                mg = {"right": sr - sl, "left": sl - sr, "bilateral": min(sr, sl) - abs(sr - sl)}[side]
            nb = 1.0 if side in sides else (0.5 if "none" in sides else 0.0)
            r = {**{c: agg[c] for c in num}, "key": k, "patient": g.patient.iloc[0], "finding": f, "lat": side,
                 "regions": g.regions.iloc[0], "fold": g.fold.iloc[0], "s_side": ss, "side_margin": mg, "nb_side": nb}
            r["label"] = E4.matches((f, side), set(ref.get(k, {})))
            r["label_strict"] = D.strict_match((f, side), set(ref.get(k, {})))
            rows.append(r)
    X = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    X["from_vlm"] = X.from_vlm.astype(float)
    X = X.assign(label=X.label_strict)                         # train and calibrate on side-aware truth
    X["score"], w = D.fit_score(X, num + ["s_side", "side_margin", "nb_side"])
    adm, t, risk = D.crc_admit(X, "fp", 0.10)
    old, _, _ = D.crc_admit(df, "fp", 0.10)
    ms, ml = D.metrics(adm, ref, ek, strict=True), D.metrics(adm, ref, ek)
    os_, ol = D.metrics(old, ref, ek, strict=True), D.metrics(old, ref, ek)
    print(f"  fp30 STRICT recall {os_['recall']:.3f} -> {ms['recall']:.3f}, precision {os_['precision']:.3f} -> {ms['precision']:.3f} | "
          f"lenient recall {ol['recall']:.3f} -> {ml['recall']:.3f} | eval risk (strict FP) {risk:.3f}")
    print(f"  side weights: s_side {w['s_side']:+.2f}, side_margin {w['side_margin']:+.2f}, nb_side {w['nb_side']:+.2f}")
    res["h13"] = {"strict_before": os_, "strict_after": ms, "lenient_before": ol, "lenient_after": ml, "eval_risk": round(risk, 4),
                  "weights": {k: float(v) for k, v in w.items()}}
    D.write_gen("e7sidexp_fp30", adm, ek)


def h11_focal(df, ref, res):
    print("\n#### H11 rare focal findings: group score offset ####")
    ek = sorted(df[df.fold >= 2].key.unique())
    k0 = sorted(df[df.fold == 0].key.unique())
    foc = df.finding.isin(FOCAL).to_numpy()
    best = (-1, 0.0)
    for delta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        d = df.assign(score=df.score + delta * foc)
        f0 = d[d.fold == 0].assign(fold=1)                       # tune on fold 0 only: calibrate and score in-fold
        grid = np.quantile(d.score, np.linspace(1, 0, 400))
        loss = lambda x, t: float((np.minimum(A5.fp_per_study(x, t, sorted(x.key.unique())), 3) / 3).mean())
        t = A5.crc_threshold(f0, loss, 0.10, grid)
        wr = wrecall(f0[f0.score >= t], ref, k0)
        best = max(best, (wr, delta))
    delta = best[1]
    d = df.assign(score=df.score + delta * foc)
    adm, t, risk = D.crc_admit(d, "fp", 0.10)
    old, _, _ = D.crc_admit(df, "fp", 0.10)
    fr = lambda a: D.metrics(a[a.finding.isin(FOCAL)], {k: {c for c in ref.get(k, {}) if c[0] in FOCAL} for k in ek}, ek)
    print(f"  delta={delta} (tuned on fold 0) | focal recall {fr(old)['recall']:.3f} -> {fr(adm)['recall']:.3f}, focal precision "
          f"{fr(old)['precision']:.3f} -> {fr(adm)['precision']:.3f} | weighted recall {wrecall(old, ref, ek):.3f} -> {wrecall(adm, ref, ek):.3f} "
          f"| eval risk {risk:.3f}")
    res["h11"] = {"delta": delta, "focal_before": fr(old), "focal_after": fr(adm), "wrecall_before": round(wrecall(old, ref, ek), 4),
                  "wrecall_after": round(wrecall(adm, ref, ek), 4), "eval_risk": round(risk, 4)}
    D.write_gen("e7focal_fp30", adm, ek)


def h9b_multi(df, man, res):
    print("\n#### H9b simultaneous subgroup validity (strictest Mondrian threshold per study) ####")
    meta = man.reindex(sorted(df.key.unique()))
    ek = sorted(df[df.fold >= 2].key.unique())
    groups = {"view": meta.frontal_view.astype(str)}
    for col, top in [("institution", 6), ("manufacturer", 4)]:
        v = meta[col].astype(str); keep = v[v.index.isin(ek)].value_counts().index[:top]
        groups[col] = v.where(v.isin(keep), "other")
    alpha = 0.30
    lossf = lambda d, t: (A5.fp_per_study(d, t, sorted(d.key.unique())) >= 1).astype(float)
    grid = np.quantile(df.score, np.linspace(1, 0, 400))
    cal, ev = df[df.fold == 1], df[df.fold >= 2]
    thr = pd.Series(-np.inf, index=meta.index)
    for gname, g in groups.items():
        for val in g.unique():
            ks = set(g[g == val].index)
            t = A5.crc_threshold(cal[cal.key.isin(ks)], lambda d, t_: float(lossf(d, t_).mean()), alpha, grid)
            idx = list(ks & set(thr.index)); thr[idx] = np.maximum(thr[idx], t)
    evt = ev.assign(th=ev.key.map(thr))
    adm = evt[evt.score >= evt.th]
    t_m = A5.crc_threshold(cal, lambda d, t_: float(lossf(d, t_).mean()), alpha, grid)
    marg = ev[ev.score >= t_m]
    fp_multi = pd.Series(A5.fp_per_study(adm.assign(score=1.0), 0.5, ek), index=ek)
    fp_marg = pd.Series(A5.fp_per_study(marg.assign(score=1.0), 0.5, ek), index=ek)
    worst_m = worst_g = 0; out = {}
    for gname, g in groups.items():
        for val in sorted(g.unique()):
            ks = [k for k in ek if g.get(k) == val]
            if len(ks) < 40:
                continue
            rm, rg = float((fp_marg[ks] >= 1).mean()), float((fp_multi[ks] >= 1).mean())
            worst_m, worst_g = max(worst_m, rm), max(worst_g, rg)
            out[f"{gname}={val}"] = {"n": len(ks), "marginal": round(rm, 4), "multi": round(rg, 4)}
    mm, mg = D.metrics(marg, D.REF, ek), D.metrics(adm, D.REF, ek)
    print(f"  worst subgroup risk: marginal {worst_m:.3f} -> multi-group {worst_g:.3f} (alpha {alpha}) | overall risk "
          f"{(fp_marg >= 1).mean():.3f} -> {(fp_multi >= 1).mean():.3f} | recall {mm['recall']:.3f} -> {mg['recall']:.3f}")
    for k, v in out.items():
        print(f"    {k:<20} n={v['n']:>5} marginal {v['marginal']:.3f} multi {v['multi']:.3f}")
    res["h9b"] = {"groups": out, "worst_marginal": worst_m, "worst_multi": worst_g, "recall_marginal": mm, "recall_multi": mg}
    D.write_gen("e7multi_any30", adm, ek)


def h10b_triage(df, probe, res):
    print("\n#### H10b autonomous normal reporting with a learned normality score ####")
    st = study_strata(HOME / "work/corpus_v2/valid")
    f = df.drop_duplicates("key").set_index("key").fold
    kv = [k for k in KEY if k in probe["F"]]
    feats = []
    smax = df.groupby("key").score.max()
    for k in f.index:
        i = probe["pos"].get(k)
        pg = probe["Pg"][i, [probe["F"][x] for x in kv]] if i is not None else np.full(len(kv), 0.5)
        feats.append(list(pg) + [float(smax.get(k, 0.0))])
    Fm = pd.DataFrame(feats, index=f.index).join(st[["abn", "key_abn"]], how="inner")
    fo = f.reindex(Fm.index).to_numpy()
    X = Fm.drop(columns=["abn", "key_abn"]).to_numpy(); y = Fm.key_abn.astype(float).to_numpy()
    tr = fo == 0
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w, b = E4B.logreg_fit((X[tr] - mu) / sd, y[tr])
    s = 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
    T = pd.DataFrame({"s": s, "fold": fo, "abn": Fm.abn.to_numpy(), "key_abn": Fm.key_abn.to_numpy()}, index=Fm.index)
    ev, cal = T[T.fold >= 2], T[T.fold == 1]
    print(f"  normality score AUROC on eval (key abnormality): {E4.auroc(ev.s, ev.key_abn):.3f} (E6 max-claim score: 0.833)")
    res["h10b"] = {"auroc": round(E4.auroc(ev.s, ev.key_abn), 4)}
    grid = np.quantile(T.s, np.linspace(0, 1, 800))
    for beta in [0.01, 0.02, 0.05]:
        n = len(cal); best = -np.inf
        for t in grid:
            r = float(((cal.s < t) & cal.key_abn).mean())
            if n / (n + 1) * r + 1 / (n + 1) <= beta:
                best = t
            else:
                break
        auto = ev.s < best
        joint = float((auto & ev.key_abn).mean())
        print(f"  beta={beta:.2f}: auto-normal {auto.mean():.3f} | joint risk {joint:.4f} ({'OK' if joint <= beta else 'over'}) | "
              f"key-abnormal among auto-normal {ev.key_abn[auto].mean() if auto.any() else float('nan'):.3f} | true normals captured {auto[~ev.abn].mean():.3f}")
        res["h10b"][str(beta)] = {"coverage": round(float(auto.mean()), 4), "joint": round(joint, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="h12,h13,h11,h9b,h10b")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    df, ref, man, probe = D.prepare()
    D.REF = ref
    tau = json.load(open(HOME / "work/e6/e6_results.json"))["h6"]["tau"]
    res = {}
    todo = a.only.split(",")
    if "h12" in todo: h12_edit(df, ref, probe, tau, load_b0_sentences(), res)
    if "h13" in todo: h13_sidexp(df, ref, probe, res)
    if "h11" in todo: h11_focal(df, ref, res)
    if "h9b" in todo: h9b_multi(df, man, res)
    if "h10b" in todo: h10b_triage(df, probe, res)
    json.dump(res, open(OUT / "e7_results.json", "w"), indent=1, default=str)
    print("E7_DONE")


if __name__ == "__main__":
    main()
