#!/usr/bin/env python3
"""E16: pivot after E11. At matched certified risk the probe-only classifier over the WHOLE vocabulary out-recalled
ANCHOR (+0.036 test recall, CI excludes 0) because the retrieved pool caps oracle recall (0.740 vs 0.892), while ANCHOR's
pool admits more precisely (0.758 vs 0.700). E16 tests whether that holds at REPORT level and whether a fusion of both
dominates. (~/venvs/anchor, small GPU for the probe)

  EXH    candidates = every vocabulary finding (side from the region resolver); scorer = probe heads
  FUS    candidates = EXH  U  out-of-vocabulary ANCHOR candidates; each (study, finding) carries the probe heads PLUS the
         retrieval votes / ranks / VLM-mention of the matching ANCHOR candidates (0 when nobody proposed it)
  Same protocol as the frozen ANCHOR: fit valid fold 0, CRC capped false claims alpha 0.10 on fold 1 (marginal and
  multi-group), focal offset delta, outputs = realizer and evidence-gated edit of the MedGemma report.

  select              valid only: claim-level table on folds 2-3, writes e16{exh,exhedit,fus,fusedit,fusmulti}_valid
                      for the judges (selection data), saves the fitted configuration to work/e16/freeze2.json
  apply --split S     frozen freeze2 on S (test = declared second look, MIMIC = clean external); writes e16*_S
Aggregates only."""
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
import e5_anchor as A5  # noqa: E402
import e8_final as F  # noqa: E402
import e11_ablate as X  # noqa: E402

OUT = HOME / "work/e16"
AGG = ["from_vlm", "inv_region_rank", "inv_global_rank", "votes_global", "votes_region_max", "votes_region_mean"]
FEX = ["probe_global", "probe_region"]
FFU = FEX + AGG + ["in_vocab"]


def build(split, C, probe, ref, tau):
    keys = sorted(C.key.unique())
    man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key").patient.astype(str).to_dict()
    ex = X.exhaustive(split, keys, probe, ref, tau, man)
    ex["in_vocab"] = 1.0
    ag = C.groupby(["key", "finding"])[AGG].max().reset_index()
    fu = ex.merge(ag, on=["key", "finding"], how="left")
    fu[AGG] = fu[AGG].fillna(0.0)
    oov = C[~C.finding.isin(set(probe["F"]))]
    if len(oov):
        o = oov.groupby(["key", "finding"]).agg({**{c: "max" for c in AGG}, "lat": "first", "patient": "first",
                                                 "probe_global": "max", "probe_region": "max"}).reset_index()
        o["in_vocab"] = 0.0
        o["label"] = [E4.matches((f, l), set(ref.get(k, {}))) for k, f, l in zip(o.key, o.finding, o.lat)]
        fu = pd.concat([fu, o[fu.columns]], ignore_index=True)
    fold = C.drop_duplicates("key").set_index("key").fold
    for d in (ex, fu):
        d["fold"] = d.key.map(fold).fillna(9).astype(int)
    return ex, fu


def outputs(split, name, adm, keys, b0s):
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    gen = HOME / "work/gen"
    pd.DataFrame({"key": keys, "generated": [A5.realize(by.get(k, [])) for k in keys]}).to_parquet(gen / f"e16{name}_{split}.parquet", index=False)
    if b0s is not None:
        import e7_refine as R
        pd.DataFrame({"key": keys, "generated": [R.edit_report(k, by.get(k, []), b0s) for k in keys]}).to_parquet(
            gen / f"e16{name}edit_{split}.parquet", index=False)


def claim_table(tag, adm, ref, keys):
    import e6_deep as D
    ml, ms = D.metrics(adm, ref, keys), D.metrics(adm, ref, keys, strict=True)
    fp = A5.fp_per_study(adm.assign(score=1.0), 0.5, keys)
    risk = float((np.minimum(fp, 3) / 3).mean())
    print(f"  {tag:<22} recall {ml['recall']:.3f} (strict {ms['recall']:.3f}) precision {ml['precision']:.3f} (strict {ms['precision']:.3f}) "
          f"false/rpt {ml['false_per_report']:.3f} claims/rpt {ml['claims_per_report']:.2f} capped risk {risk:.4f}")
    return {"lenient": ml, "strict": ms, "risk": round(risk, 4)}


def group_thr(split, d, keys, fz, t_marg, tops=None, cal=None, grid=None):
    G = F.groups_of(split, keys, tops)
    if cal is not None:
        thr = {}
        for gname, g in G.items():
            for val in g.unique():
                c = cal[cal.key.isin(set(g[g == val].index))]
                thr[f"{gname}={val}"] = float(A5.crc_threshold(c, F.fp_loss, F.ALPHA, grid)) if c.key.nunique() >= 40 else float(t_marg)
        return thr
    th = pd.Series(-np.inf, index=keys)
    for gname, g in G.items():
        th = np.maximum(th, g.map(lambda v: fz["group_thresholds"].get(f"{gname}={v}", t_marg)).reindex(keys).fillna(t_marg))
    return th


def admitted(split, fz, ex, fu, keys):
    ex = ex.assign(score=F.apply_scorer(ex, fz["scorer_exh"], fz["delta"])); fu = fu.assign(score=F.apply_scorer(fu, fz["scorer_fus"], fz["delta"]))
    e, f = ex[ex.key.isin(keys)], fu[fu.key.isin(keys)]
    th = group_thr(split, f, keys, {"group_thresholds": fz["group_thresholds_fus"]}, fz["t_fus"], tops=fz["group_tops"])
    return {"exh": e[e.score >= fz["t_exh"]], "fus": f[f.score >= fz["t_fus"]], "fusmulti": f[f.score >= f.key.map(th)]}


def cmd_select(a):
    OUT.mkdir(parents=True, exist_ok=True)
    fz0 = json.load(open(F.OUT / "freeze.json"))
    C = F.load_cands("valid"); ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    probe = F.probe_arrays("valid", C.key.unique())
    ex, fu = build("valid", C, probe, ref, fz0["tau"])
    fz = {"alpha": F.ALPHA, "tau": fz0["tau"], "delta": fz0["delta"], "group_tops": fz0["group_tops"]}
    for nm, d, fs in [("exh", ex, FEX), ("fus", fu, FFU)]:
        fs = X.drop_const(d, fs)
        m = F.fit(d, fs); s = F.apply_scorer(d, m, fz["delta"])
        grid = np.quantile(s, np.linspace(1, 0, 600)); cal = d.assign(score=s)[d.fold == 1]
        fz[f"scorer_{nm}"] = m; fz[f"t_{nm}"] = float(A5.crc_threshold(cal, F.fp_loss, F.ALPHA, grid))
        if nm == "fus":
            fz["group_thresholds_fus"] = group_thr("valid", d, sorted(d.key.unique()), None, fz["t_fus"], tops=fz["group_tops"], cal=cal, grid=grid)
            print("fusion scorer weights (standardised): " + ", ".join(f"{f_}={w:+.2f}" for f_, w in zip(fs, m["w"])))
    code = {f_: hashlib.md5(open(pathlib.Path(__file__).resolve().parent / f_, "rb").read()).hexdigest() for f_ in ["e16_fusion.py", "e11_ablate.py", "e8_final.py"]}
    fz["code_md5"] = code
    json.dump(fz, open(OUT / "freeze2.json", "w"), indent=1)
    print(f"FREEZE2 saved: t_exh {fz['t_exh']:.4f}, t_fus {fz['t_fus']:.4f}, md5 {hashlib.md5(open(OUT / 'freeze2.json', 'rb').read()).hexdigest()}")
    keys = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key)
    run(a, "valid", fz, ex, fu, keys, ref, probe)


def run(a, split, fz, ex, fu, keys, ref, probe):
    A = admitted(split, fz, ex, fu, keys)
    print(f"\n=== {split.upper()} ({len(keys)} studies), claim level at alpha {F.ALPHA} ===")
    res = {}
    b0 = F.load_cands(split); b0 = b0[(b0.from_vlm > 0) & b0.key.isin(keys)]
    res["b0"] = claim_table("MedGemma B0", b0, ref, keys)
    b0s = F.load_b0_sentences(split)
    for nm, adm in A.items():
        adm = F.resolve_sides(adm, probe, fz["tau"], ref)
        res[nm] = claim_table({"exh": "EXH classifier", "fus": "FUS fusion", "fusmulti": "FUS multi-group"}[nm], adm, ref, keys)
        outputs(split, nm, adm, keys, b0s if nm != "fusmulti" else None)
    G = F.groups_of(split, keys, fz["group_tops"])
    worst = {}
    for nm in ["fus", "fusmulti"]:
        adm = F.resolve_sides(A[nm], probe, fz["tau"], ref)
        fp = pd.Series(A5.fp_per_study(adm.assign(score=1.0), 0.5, keys), index=keys)
        w = max(float((np.minimum(fp[g[g == v].index], 3) / 3).mean()) for g in G.values() for v in g.unique() if (g == v).sum() >= 40)
        worst[nm] = round(w, 4)
    print(f"  worst subgroup capped risk: FUS {worst['fus']:.3f} | FUS multi-group {worst['fusmulti']:.3f} (alpha {F.ALPHA})")
    res["worst_subgroup"] = worst
    json.dump(res, open(OUT / f"results_{split}.json", "w"), indent=1, default=str)
    pd.DataFrame({"key": keys}).to_parquet(OUT / f"eval_keys_{split}.parquet", index=False)
    print(f"E16_{split.upper()}_DONE")


def cmd_apply(a):
    fz = json.load(open(OUT / "freeze2.json"))
    for f_, h in fz["code_md5"].items():
        if hashlib.md5(open(pathlib.Path(__file__).resolve().parent / f_, "rb").read()).hexdigest() != h:
            print(f"WARNING: {f_} changed since freeze2")
    C = F.load_cands(a.split); ref = E4.load_claims(HOME / f"work/corpus_v2/{a.split}")
    probe = F.probe_arrays(a.split, C.key.unique())
    ex, fu = build(a.split, C, probe, ref, fz["tau"])
    run(a, a.split, fz, ex, fu, sorted(C.key.unique()), ref, probe)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "apply"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    {"select": cmd_select, "apply": cmd_apply}[a.stage](a)


if __name__ == "__main__":
    main()
