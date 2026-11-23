#!/usr/bin/env python3
"""E12: the closest published alternative to ANCHOR's admission step = conformal factuality with a SELF-CONSISTENCY
claim score (Mohri & Hashimoto 2024; RadFlag, ML4H 2024; ConfLVLM, EMNLP 2025): sample the VLM N times, score each
claim of the greedy report by how often the samples repeat it, keep claims above a conformally calibrated threshold.
Same loss (capped false claims), same alpha, same folds (fit 0 / calibrate 1 / evaluate 2-3) as ANCHOR. (~/venvs/anchor)

  SC-CRC                 MedGemma claims, score = sample agreement frequency
  ANCHOR on VLM claims   MedGemma claims, score = ANCHOR evidence scorer (image region probe + neighbour votes)
  ANCHOR (frozen)        full candidate pool, frozen scorer and threshold
  ANCHOR + SC feature    full pool, scorer refit with the frequency as an extra feature (is self-consistency complementary?)
Needs: work/corpus_gen/b0sc{s}_medgemma4b_valid (gentext of the N samples). Aggregates only."""
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


def main():
    fz = json.load(open(F.OUT / "freeze.json")); delta = fz["delta"]
    V = F.load_cands("valid")
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    dirs = sorted((HOME / "work/corpus_gen").glob("b0sc*_medgemma4b_valid"))
    S = [E4.load_claims(d) for d in dirs]
    print(f"self-consistency samples found: {len(S)}")
    V["sc_freq"] = [np.mean([E4.matches((f, l), set(s.get(k, {}))) for s in S]) for k, f, l in zip(V.key, V.finding, V.lat)]
    V["score_anchor"] = F.apply_scorer(V, fz["scorer_full"], delta)
    kE = sorted(V[V.fold >= 2].key.unique())
    vl = V[V.from_vlm > 0]
    ev = vl[vl.fold >= 2]
    res = {"n_samples": len(S), "auroc_on_vlm_claims_eval": {"sc_freq": round(E4.auroc(ev.sc_freq, ev.label), 4),
           "anchor_score": round(E4.auroc(ev.score_anchor, ev.label), 4), "probe_region": round(E4.auroc(ev.probe_region, ev.label), 4)}}
    print("claim-level AUROC on MedGemma's own claims (valid eval folds):", res["auroc_on_vlm_claims_eval"])
    print(f"self-consistency of retrieved (non-VLM) candidates: mean freq {V[V.from_vlm == 0].sc_freq.mean():.3f} | "
          f"VLM claims {vl.sc_freq.mean():.3f}")
    store = {}

    def crc_eval(name, d, col):
        cal, e = d[d.fold == 1], d[d.fold >= 2]
        grid = np.unique(np.quantile(d[col], np.linspace(1, 0, 500)))[::-1]
        r = {}
        for al in X.ALPHAS:
            t = A5.crc_threshold(cal.assign(score=cal[col]), F.fp_loss, al, grid)
            ps = X.per_study(e[e[col] >= t], ref, kE)
            r[str(al)] = X.summ(ps)
            if al == 0.10:
                store[name] = ps
        res[name] = r
        print(f"  {name:<26} alpha .05 {r['0.05']} | .10 {r['0.1']} | .20 {r['0.2']}", flush=True)

    print("\n#### admission at matched certified risk (valid eval folds) ####")
    crc_eval("SC-CRC", vl, "sc_freq")
    crc_eval("ANCHOR on VLM claims", vl, "score_anchor")
    crc_eval("ANCHOR (full pool)", V, "score_anchor")
    fs = X.drop_const(V, A5.FEATS + ["sc_freq"])
    m = F.fit(V, fs); V["score_sc"] = F.apply_scorer(V, m, delta)
    print("  ANCHOR + SC weights (standardised): " + ", ".join(f"{f}={w:+.2f}" for f, w in zip(fs, m["w"])))
    crc_eval("ANCHOR + SC feature", V, "score_sc")
    store["B0 raw"] = X.per_study(V[(V.fold >= 2) & (V.from_vlm > 0)], ref, kE)
    print("\npaired recall differences at alpha 0.10:")
    for a, b in [("ANCHOR on VLM claims", "SC-CRC"), ("ANCHOR (full pool)", "SC-CRC"), ("ANCHOR + SC feature", "ANCHOR (full pool)"), ("SC-CRC", "B0 raw")]:
        d = X.boot_recall_diff(store[a], store[b]); res[f"{a} - {b}"] = d
        print(f"  {a} minus {b}: {d[0]:+.4f} {d[1]}")
    (HOME / "work/e12").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HOME / "work/e12/e12_results.json", "w"), indent=1)
    print("E12_DONE")


if __name__ == "__main__":
    main()
