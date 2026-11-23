#!/usr/bin/env python3
"""E8: FREEZE the final ANCHOR configuration on valid, then apply it ONCE to the held-out test split (~/venvs/anchor).

Final configuration (each component chosen on valid evidence, D59):
  candidates  MedGemma B0 claims U top-8 region-retrieved U top-8 global-retrieved (k=10 neighbours)
  scorer      logistic on votes + ranks + trained region probe (fit: valid fold 0)
  focal       +delta score offset for rare focal findings (delta tuned on fold 0 in E7)
  admission   CRC, loss = min(false claims, 3)/3, alpha = 0.10 (calibrated: valid fold 1)
              + multi-group variant: per study, strictest of its view / institution / manufacturer thresholds
  side        region-probe side resolver (tau tuned on fold 0 in E6)
  output      'anchor' = deterministic realizer;  'edit' = evidence-gated editing of the MedGemma report
  vlmfree     same, without MedGemma claims (retrieval + probe only)
  triage      autonomous-normal score (probe heads + max claim score), CRC and high-probability (Clopper-Pearson,
              delta = 0.1) thresholds for critical-miss risk beta in {0.02, 0.05}

  freeze               fit + calibrate on valid, save ~/anchor/work/e8/freeze.json (+ code checksums), write valid-eval reports
  apply --split test   load the frozen config, write test reports, print claim-level results and realized risks
Aggregates only.
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import e4a_pool as E4  # noqa: E402
import e4b_verify as E4B  # noqa: E402
import e5_anchor as A5  # noqa: E402
import e6_deep as D  # noqa: E402
import e7_refine as R  # noqa: E402
from strata import KEY  # noqa: E402

OUT = HOME / "work/e8"
CRIT = {"pneumothorax", "pleural effusion", "consolidation", "airspace opacity", "pulmonary edema", "nodule", "mass", "fracture"}
ALPHA = 0.10


def cand_dir(split):
    return HOME / "work" / ("e5" if split == "valid" else f"e8_{split}")


def load_cands(split):
    d = cand_dir(split)
    df = pd.read_parquet(d / "candidates.parquet")
    pr = pd.read_parquet(d / "verify_probe.parquet")
    df["probe_region"] = pr.probe_region.fillna(0.5).to_numpy(); df["probe_global"] = pr.probe_global.fillna(0.5).to_numpy()
    df["inv_region_rank"] = 1 / df.region_rank; df["inv_global_rank"] = 1 / df.global_rank
    df["from_vlm"] = df.from_vlm.astype(float)
    df["fold"] = [int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 4 for p in df.patient] if split == "valid" else 9
    return df


def probe_arrays(split, keys):
    import torch
    import e4c_probe as P
    reg, glo, vocab = P.load_probe(torch)
    ks, Em, Ar, Cl = P.load_split(split, sorted(keys))
    Pr, Pg = P.predict(torch, reg, glo, Em, Cl)
    return {"pos": {k: i for i, k in enumerate(ks)}, "F": {f: j for j, f in enumerate(vocab)}, "Pr": Pr, "Pg": Pg, "Ar": Ar}


def fit(df, feats):
    X = df[feats].astype(float).to_numpy(); y = df.label.astype(float).to_numpy(); tr = (df.fold == 0).to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w, b = E4B.logreg_fit((X[tr] - mu) / sd, y[tr])
    return {"feats": feats, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(b)}


def apply_scorer(df, m, delta):
    X = df[m["feats"]].astype(float).to_numpy()
    z = ((X - np.array(m["mu"])) / np.array(m["sd"])) @ np.array(m["w"]) + m["b"]
    return 1 / (1 + np.exp(-z)) + delta * df.finding.isin(R.FOCAL).to_numpy()


def fp_loss(d, t):
    return float((np.minimum(A5.fp_per_study(d, t, sorted(d.key.unique())), 3) / 3).mean())


def groups_of(split, keys, tops):
    man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key").reindex(sorted(keys))
    out = {"view": man.frontal_view.astype(str)}
    for col in ["institution", "manufacturer"]:
        v = man[col].astype(str); out[col] = v.where(v.isin(tops[col]), "other")
    return out


def cp_upper(k, n, delta):
    """Clopper-Pearson upper bound for a Bernoulli rate (exact binomial tail, bisection)."""
    if k >= n:
        return 1.0
    def tail(p):                                       # P(X <= k | n, p)
        s = 0.0
        for i in range(k + 1):
            s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(max(p, 1e-300)) + (n - i) * math.log(max(1 - p, 1e-300)))
        return s
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if tail(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def triage_features(df, probe, keys):
    kv = [k for k in KEY if k in probe["F"]]
    smax = df.groupby("key").score.max()
    rows = []
    for k in keys:
        i = probe["pos"].get(k)
        pg = probe["Pg"][i, [probe["F"][x] for x in kv]] if i is not None else np.full(len(kv), 0.5)
        rows.append(list(pg) + [float(smax.get(k, 0.0))])
    return np.array(rows)


def realize_and_edit(adm, keys, b0s):
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    real = [A5.realize(by.get(k, [])) for k in keys]
    edit = [R.edit_report(k, by.get(k, []), b0s) for k in keys] if b0s is not None else None
    return real, edit


def load_b0_sentences(split):
    rows = {}
    for p in sorted((HOME / f"work/corpus_gen/b0_medgemma4b_{split}").glob("shard_*.parquet")):
        d = pd.read_parquet(p, columns=["key", "sid", "sentence", "findings_json"])
        for k, sid, s, js in zip(d.key, d.sid, d.sentence, d.findings_json):
            cl = [(E4.canon(str(f.get("finding", ""))), f.get("laterality") or "none", f.get("polarity")) for f in json.loads(js)]
            rows.setdefault(k, []).append((sid, s, cl))
    return {k: sorted(v) for k, v in rows.items()}


def resolve_sides(adm, probe, tau, ref):
    adm = adm.copy()
    adm["lat"] = [D.resolve(probe, k, f, l, tau) for k, f, l in zip(adm.key, adm.finding, adm.lat)]
    adm = adm.drop_duplicates(["key", "finding", "lat"])
    adm["label"] = [E4.matches((f, l), set(ref.get(k, {}))) for k, f, l in zip(adm.key, adm.finding, adm.lat)]  # truth of the FINAL claim
    return adm


def report(name, adm, ref, keys, extra=""):
    ml, ms = D.metrics(adm, ref, keys), D.metrics(adm, ref, keys, strict=True)
    fp = A5.fp_per_study(adm.assign(score=1.0, label=adm.label), 0.5, keys)
    print(f"  {name:<12} recall {ml['recall']:.3f} (strict {ms['recall']:.3f}) precision {ml['precision']:.3f} "
          f"(strict {ms['precision']:.3f}) false/rpt {ml['false_per_report']:.3f} capped-FP risk {float((np.minimum(fp, 3) / 3).mean()):.4f} {extra}")
    return {"lenient": ml, "strict": ms, "capped_fp_risk": round(float((np.minimum(fp, 3) / 3).mean()), 4)}


def cmd_freeze(a):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_cands("valid")
    e6 = json.load(open(HOME / "work/e6/e6_results.json")); e7 = json.load(open(HOME / "work/e7/e7_results.json"))
    tau, delta = e6["h6"]["tau"], float(e7["h11"]["delta"])
    m_full = fit(df, A5.FEATS)
    vf = df[(df.region_rank < 99) | (df.global_rank < 99)].copy()
    m_vf = fit(vf, [f for f in A5.FEATS if f != "from_vlm"])
    df["score"] = apply_scorer(df, m_full, delta); vf["score"] = apply_scorer(vf, m_vf, delta)
    grid = np.quantile(df.score, np.linspace(1, 0, 600))
    cal = df[df.fold == 1]
    t_full = A5.crc_threshold(cal, fp_loss, ALPHA, grid)
    t_vf = A5.crc_threshold(vf[vf.fold == 1], fp_loss, ALPHA, np.quantile(vf.score, np.linspace(1, 0, 600)))
    ek = sorted(df[df.fold >= 2].key.unique())
    tops = {}
    man = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key")
    for col, n in [("institution", 6), ("manufacturer", 4)]:
        tops[col] = man.loc[man.index.isin(ek), col].astype(str).value_counts().index[:n].tolist()
    G = groups_of("valid", df.key.unique(), tops)
    gthr = {}
    for gname, g in G.items():
        for val in g.unique():
            ks = set(g[g == val].index)
            c = cal[cal.key.isin(ks)]
            gthr[f"{gname}={val}"] = float(A5.crc_threshold(c, fp_loss, ALPHA, grid)) if c.key.nunique() >= 40 else float(t_full)
    # triage
    probe = probe_arrays("valid", df.key.unique())
    keys_all = sorted(df.key.unique())
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    crit = np.array([any(c[0] in CRIT for c in ref.get(k, {})) for k in keys_all])
    folds = df.drop_duplicates("key").set_index("key").fold.reindex(keys_all).to_numpy()
    Xt = triage_features(df, probe, keys_all)
    tr = folds == 0
    mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
    wt, bt = E4B.logreg_fit((Xt[tr] - mu) / sd, crit[tr].astype(float))
    st = 1 / (1 + np.exp(-(((Xt - mu) / sd) @ wt + bt)))
    tri = {"mu": mu.tolist(), "sd": sd.tolist(), "w": wt.tolist(), "b": float(bt), "thr": {}}
    cs, cc = st[folds == 1], crit[folds == 1]; n = len(cs)
    for beta in [0.02, 0.05]:
        best_crc = best_cp = -np.inf
        for t in np.quantile(st, np.linspace(0, 1, 800)):
            k = int(((cs < t) & cc).sum())
            if n / (n + 1) * k / n + 1 / (n + 1) <= beta:
                best_crc = t
            if cp_upper(k, n, 0.10) <= beta:
                best_cp = t
            if k / n > beta:
                break
        tri["thr"][str(beta)] = {"crc": float(best_crc), "cp_delta0.1": float(best_cp)}
    code = {f: hashlib.md5(open(CODE / f, "rb").read()).hexdigest() for f in
            ["e4a_pool.py", "e4b_verify.py", "e4c_probe.py", "e5_anchor.py", "e6_deep.py", "e7_refine.py", "e8_final.py"]}
    code["probe.pt"] = hashlib.md5(open(HOME / "work/e4c/probe.pt", "rb").read()).hexdigest()
    fz = {"alpha": ALPHA, "tau": tau, "delta": delta, "scorer_full": m_full, "scorer_vlmfree": m_vf, "t_full": float(t_full),
          "t_vlmfree": float(t_vf), "group_thresholds": gthr, "group_tops": tops, "triage": tri, "code_md5": code}
    json.dump(fz, open(OUT / "freeze.json", "w"), indent=1)
    print(f"FROZEN: t_full {t_full:.4f}, t_vlmfree {t_vf:.4f}, tau {tau}, delta {delta}, {len(gthr)} group thresholds, "
          f"freeze md5 {hashlib.md5(open(OUT / 'freeze.json', 'rb').read()).hexdigest()}")
    run_outputs("valid", df, vf, fz, probe, ref, ek, keys_all, None)


def run_outputs(split, df, vf, fz, probe, ref, keys, keys_all, _):
    b0s = load_b0_sentences(split)
    ev = df[df.key.isin(keys)]; evf = vf[vf.key.isin(keys)]
    adm = resolve_sides(ev[ev.score >= fz["t_full"]], probe, fz["tau"], ref)
    G = groups_of(split, keys, fz["group_tops"])
    th = pd.Series(-np.inf, index=keys)
    for gname, g in G.items():
        th = np.maximum(th, g.map(lambda v: fz["group_thresholds"].get(f"{gname}={v}", fz["t_full"])).reindex(keys).fillna(fz["t_full"]))
    ev_m = ev.assign(th=ev.key.map(th))
    adm_m = resolve_sides(ev_m[ev_m.score >= ev_m.th], probe, fz["tau"], ref)
    adm_v = resolve_sides(evf[evf.score >= fz["t_vlmfree"]], probe, fz["tau"], ref)
    b0 = ev[ev.from_vlm > 0]
    print(f"\n=== {split.upper()} ({len(keys)} studies): claim level, frozen config ===")
    res = {"b0": report("MedGemma B0", b0, ref, keys), "anchor": report("ANCHOR", adm, ref, keys),
           "multi": report("ANCHOR-multi", adm_m, ref, keys), "vlmfree": report("VLM-free", adm_v, ref, keys)}
    print(f"  guarantee alpha {ALPHA}: ANCHOR realized {res['anchor']['capped_fp_risk']:.4f} "
          f"({'OK' if res['anchor']['capped_fp_risk'] <= ALPHA else 'OVER'}), multi {res['multi']['capped_fp_risk']:.4f}")
    print("  subgroup capped-FP risk (marginal ANCHOR | multi-group):")
    fpa = pd.Series(A5.fp_per_study(adm.assign(score=1.0), 0.5, keys), index=keys)
    fpm = pd.Series(A5.fp_per_study(adm_m.assign(score=1.0), 0.5, keys), index=keys)
    worst = [0, 0]
    for gname, g in G.items():
        for val in sorted(g.unique()):
            ks = g[g == val].index
            if len(ks) < 40:
                continue
            ra, rm = float((np.minimum(fpa[ks], 3) / 3).mean()), float((np.minimum(fpm[ks], 3) / 3).mean())
            worst = [max(worst[0], ra), max(worst[1], rm)]
            print(f"    {gname}={val:<8} n={len(ks):>5}  {ra:.3f} | {rm:.3f}")
    res["worst_subgroup"] = worst
    print(f"  worst subgroup: marginal {worst[0]:.3f} | multi-group {worst[1]:.3f} (alpha {ALPHA})")
    real, edit = realize_and_edit(adm, keys, b0s)
    gen = HOME / "work/gen"
    pd.DataFrame({"key": keys, "generated": real}).to_parquet(gen / f"e8anchor_{split}.parquet", index=False)
    pd.DataFrame({"key": keys, "generated": edit}).to_parquet(gen / f"e8edit_{split}.parquet", index=False)
    pd.DataFrame({"key": keys, "generated": realize_and_edit(adm_m, keys, None)[0]}).to_parquet(gen / f"e8multi_{split}.parquet", index=False)
    pd.DataFrame({"key": keys, "generated": realize_and_edit(adm_v, keys, None)[0]}).to_parquet(gen / f"e8vlmfree_{split}.parquet", index=False)
    # triage
    tri = fz["triage"]
    Xt = triage_features(df, probe, keys)
    s = 1 / (1 + np.exp(-(((Xt - np.array(tri["mu"])) / np.array(tri["sd"])) @ np.array(tri["w"]) + tri["b"])))
    crit = np.array([any(c[0] in CRIT for c in ref.get(k, {})) for k in keys])
    abn = np.array([len(ref.get(k, {})) > 0 for k in keys])
    print("  autonomous normal reporting (critical findings = pneumothorax, effusion, consolidation, opacity, edema, nodule, mass, fracture):")
    print(f"    critical-abnormality AUROC {E4.auroc(s, crit):.3f}")
    res["triage"] = {"auroc": round(E4.auroc(s, crit), 4)}
    for beta, th_ in tri["thr"].items():
        for kind, t in th_.items():
            auto = s < t
            joint = float((auto & crit).mean())
            res["triage"][f"{beta}|{kind}"] = {"coverage": round(float(auto.mean()), 4), "joint_risk": round(joint, 4),
                                                "critical_among_auto": round(float(crit[auto].mean()) if auto.any() else float("nan"), 4)}
            print(f"    beta={beta} {kind:<12}: auto-normal {auto.mean():.3f} | joint critical-miss {joint:.4f} "
                  f"({'OK' if joint <= float(beta) else 'OVER'}) | critical among auto-normal {crit[auto].mean() if auto.any() else float('nan'):.3f} "
                  f"| true normals auto {auto[~abn].mean() if (~abn).any() else float('nan'):.3f}")
    json.dump(res, open(OUT / f"results_{split}.json", "w"), indent=1, default=str)
    pd.DataFrame({"key": keys}).to_parquet(OUT / f"eval_keys_{split}.parquet", index=False)
    print(f"E8_{split.upper()}_DONE")


def cmd_apply(a):
    fz = json.load(open(OUT / "freeze.json"))
    for f, h in fz["code_md5"].items():
        p = (HOME / "work/e4c/probe.pt") if f == "probe.pt" else CODE / f
        if hashlib.md5(open(p, "rb").read()).hexdigest() != h:
            print(f"WARNING: {f} changed since freeze")
    df = load_cands(a.split)
    vf = df[(df.region_rank < 99) | (df.global_rank < 99)].copy()
    df["score"] = apply_scorer(df, fz["scorer_full"], fz["delta"]); vf["score"] = apply_scorer(vf, fz["scorer_vlmfree"], fz["delta"])
    keys = sorted(df.key.unique())
    probe = probe_arrays(a.split, keys)
    ref = E4.load_claims(HOME / f"work/corpus_v2/{a.split}")
    run_outputs(a.split, df, vf, fz, probe, ref, keys, keys, None)


def cmd_build(a):
    E4B.OUT = cand_dir(a.split); E4B.OUT.mkdir(parents=True, exist_ok=True)
    E4B.cmd_build(argparse.Namespace(n=10 ** 9, k=10, m=8, vlm=f"b0_medgemma4b_{a.split}", split=a.split))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "build", "apply"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    {"freeze": cmd_freeze, "build": cmd_build, "apply": cmd_apply}[a.stage](a)


if __name__ == "__main__":
    main()
