#!/usr/bin/env python3
"""E26 (round 48): GENERATOR INDEPENDENCE (Prof. Jang, point 3). ANCHOR on a second medical VLM, Lingshu-7B
(Qwen2.5-VL backbone, medical training; a different family from MedGemma). ~/venvs/anchor, GPU for the build.

  python e26_plugin.py build --split valid|mimic|test   candidates = Lingshu draft claims + the SAME per-region and
                                                         global neighbours (writes work/e26_<split>/candidates.parquet)
  (then: python e4c_probe.py score --dir e26_<split> --split <split>)
  python e26_plugin.py run --split valid                calibrate on fold 1, evaluate folds 2-3, write reports
  python e26_plugin.py run --split mimic                apply the valid-calibrated thresholds (external)
  python e26_plugin.py run --split test                 GATED: one labelled test look of the plug-in

PRE-STATED (round 48, before any Lingshu run):
 primary = PLUG-IN: the frozen MedGemma-fitted scorer is reused unchanged ('from_vlm' now means 'Lingshu said it');
           only the admission threshold is re-estimated on valid fold 1 (stated claims, CRC alpha 0.10, multi-group).
 sensitivity = REFIT: the same recipe re-run (scorer refitted on fold 0).
 H26a  ANCHOR-L improves CRIMSON over the Lingshu draft on valid folds 2-3 (paired CI > 0) and on MIMIC-CXR.
 H26b  ANCHOR-L at least halves false findings per report vs the Lingshu draft and meets alpha on folds 2-3.
 H26c  ANCHOR-L-edit is non-inferior to the Lingshu draft on GREEN (lower CI >= -0.01).
 Disproof: H26a CI includes 0 or is negative on valid. Aggregates only."""
import argparse
import json

import numpy as np
import pandas as pd

import e2x_common as X

E4, A5, F = X.E4, X.A5, X.F
GEN = "b0_lingshu7b"
OUTJ = X.HOME / "work/e26"


def cmd_build(a):
    import e4b_verify as E4B
    E4B.OUT = X.HOME / f"work/e26_{a.split}"; E4B.OUT.mkdir(parents=True, exist_ok=True)
    E4B.cmd_build(argparse.Namespace(n=10 ** 9, k=10, m=8, vlm=f"{GEN}_{a.split}", split=a.split))
    print(f"E26_BUILD_DONE {a.split}")


def cmd_run(a):
    OUTJ.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(F.OUT / "freeze.json")); tops = fz["group_tops"]
    C = X.load_cands_dir(f"e26_{a.split}", a.split)
    ref = E4.load_claims(X.HOME / f"work/corpus_v2/{a.split}")
    keys_all = sorted(C.key.unique())
    probe = F.probe_arrays(a.split, keys_all)
    keys = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key) if a.split == "valid" else keys_all
    keys = [k for k in keys if k in set(keys_all)]
    if a.split == "valid":
        m_ref = F.fit(C, A5.FEATS)
        cal = {}
        for mode, m in [("plugin", fz["scorer_full"]), ("refit", m_ref)]:
            c = C.assign(score=F.apply_scorer(C, m, fz["delta"]))
            fin = X.final_claims(c, probe, fz["tau"], ref)
            t, g = X.calibrate(fin[fin.fold == 1], np.quantile(c.score, np.linspace(1, 0, 600)), "valid", keys_all, tops)
            cal[mode] = {"scorer": m, "t": t, "groups": g}
            print(f"{mode}: threshold {t:.4f} (MedGemma ANCHOR frozen {fz['t_full']:.4f})")
        json.dump(cal, open(OUTJ / "freeze_lingshu.json", "w"), indent=1, default=float)
        print(f"freeze_lingshu.json md5 {X.md5(OUTJ / 'freeze_lingshu.json')} (valid only)")
    cal = json.load(open(OUTJ / "freeze_lingshu.json"))
    print(f"\n=== E26 {a.split.upper()} ({len(keys)} studies): ANCHOR on Lingshu-7B drafts, stated claims ===")
    res = {"draft": X.claim_summary("Lingshu-7B draft", X.raw_claims(C[(C.from_vlm > 0) & C.key.isin(set(keys))], ref), ref, keys, a.split, tops)}
    sents = X.load_sentences(f"{GEN}_{a.split}")
    for mode in ["plugin", "refit"]:
        c = C.assign(score=F.apply_scorer(C, cal[mode]["scorer"], fz["delta"]))
        fin = X.final_claims(c, probe, fz["tau"], ref)
        adm = X.admit(fin, keys, cal[mode]["t"])
        admm = X.admit(fin, keys, X.multi_thresholds(a.split, keys, cal[mode]["groups"], cal[mode]["t"], tops))
        res[mode] = X.claim_summary(f"ANCHOR-L ({mode})", adm, ref, keys, a.split, tops)
        res[mode + "_multi"] = X.claim_summary(f"ANCHOR-L-multi ({mode})", admm, ref, keys, a.split, tops)
        L = set(zip(c[c.from_vlm > 0].key, c[c.from_vlm > 0].finding))
        print(f"  ({mode}) share of admitted claims that Lingshu proposed: {np.mean([(k, f) in L for k, f in zip(adm.key, adm.finding)]):.3f}")
        if mode == "plugin":
            X.write_reports("e26anchorl", a.split, adm, keys, sents)
    json.dump(res, open(OUTJ / f"results_{a.split}.json", "w"), indent=1, default=float)
    print("E26_RUN_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["build", "run"])
    ap.add_argument("--split", default="valid", choices=["valid", "mimic", "test"])
    a = ap.parse_args(argv)
    {"build": cmd_build, "run": cmd_run}[a.stage](a)


if __name__ == "__main__":
    main()
