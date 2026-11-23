#!/usr/bin/env python3
"""E14: is the ANCHOR guarantee an artefact of OUR extractor? (~/venvs/anchor_eval, small GPU for CheXbert)
The conformal labels come from the E1 LLM extractor (held-out finding+polarity 0.835). Here the test references are
re-labelled by an INDEPENDENT labeller (CheXbert, 13 conditions) and every system's admitted claims are scored in that
label space:
  - label-level precision / recall / false labels per report and capped false-label risk under CheXbert
  - the same quantities under our extractor mapped to the same label space (apples to apples)
  - per-label agreement (kappa) of the two labellers
If the system ranking and the risk level agree under both labellers, the guarantee is not labeller-specific.
Needs work/e11/admitted_test.parquet (E11). Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
from chexbert_strata import CONDITIONS, kappa  # noqa: E402
MAP = {"pleural effusion": "Pleural Effusion", "pneumothorax": "Pneumothorax", "consolidation": "Consolidation",
       "atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly", "pulmonary edema": "Edema",
       "airspace opacity": "Lung Opacity", "interstitial opacity": "Lung Opacity", "nodule": "Lung Lesion",
       "mass": "Lung Lesion", "fracture": "Fracture", "support device": "Support Devices",
       "mediastinal widening": "Enlarged Cardiomediastinum"}                     # same map as e9_mimic_labels.py

OUT = HOME / "work/e14"


def chexbert_labels(split, keys):
    p = OUT / f"chexbert_{split}.parquet"
    if p.exists():
        return pd.read_parquet(p).set_index("key")
    import eval_smoke
    eval_smoke.alias_radeval()
    from radeval.metrics.f1chexbert.f1chexbert import F1CheXbert
    fm = sys.modules[F1CheXbert.__module__]
    names = next((v for v in vars(fm).values() if isinstance(v, (list, tuple)) and len(v) == 14 and all(isinstance(x, str) for x in v)), None)
    names = list(names) if names else CONDITIONS
    refs = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key").loc[keys, "findings"].fillna("").tolist()
    lab = np.array(F1CheXbert().get_labels(refs, mode="rrg"))
    df = pd.DataFrame(lab, columns=names); df.insert(0, "key", keys)
    df.to_parquet(p, index=False)
    return df.set_index("key")


def score(pred, truth, keys, labels):
    tp = fp = fn = 0; fps = np.zeros(len(keys))
    for i, k in enumerate(keys):
        P, Tr = pred.get(k, set()), truth.get(k, set())
        a, b, c = len(P & Tr), len(P - Tr), len(Tr - P)
        tp += a; fp += b; fn += c; fps[i] = b
    Pp, Rr = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(Pp, 4), "recall": round(Rr, 4), "f1": round(2 * Pp * Rr / max(Pp + Rr, 1e-9), 4),
            "false_labels_per_report": round(fp / len(keys), 3), "capped_false_label_risk": round(float((np.minimum(fps, 3) / 3).mean()), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adm = pd.read_parquet(HOME / "work/e11/admitted_test.parquet")
    keys = sorted(pd.read_parquet(HOME / "work/e8/eval_keys_test.parquet").key)
    cx = chexbert_labels("test", keys)
    labels = sorted(set(MAP.values()) & set(cx.columns))
    truth_cx = {k: {L for L in labels if cx.at[k, L] == 1} for k in keys}
    ref = E4.load_claims(HOME / "work/corpus_v2/test")
    truth_e1 = {k: {MAP[f] for f, _ in ref.get(k, {}) if f in MAP} for k in keys}
    print(f"test studies {len(keys)} | label space: {labels}")
    agree = {L: round(kappa([L in truth_cx[k] for k in keys], [L in truth_e1[k] for k in keys]), 3) for L in labels}
    print("labeller agreement (kappa, CheXbert vs our extractor, per label):", agree)
    res = {"kappa": agree, "systems": {}}
    print(f"\n{'system':<10}{'labeller':<10}{'precision':>10}{'recall':>8}{'F1':>7}{'false/rpt':>11}{'capped risk':>13}")
    for s in ["b0", "anchor", "multi", "vlmfree"]:
        a = adm[adm.system == s]
        pred = {k: {MAP[f] for f in g.finding if f in MAP} for k, g in a.groupby("key")}
        for nm, tr in [("CheXbert", truth_cx), ("extractor", truth_e1)]:
            r = score(pred, tr, keys, labels)
            res["systems"][f"{s}|{nm}"] = r
            print(f"{s:<10}{nm:<10}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}{r['false_labels_per_report']:>11.3f}{r['capped_false_label_risk']:>13.4f}")
    json.dump(res, open(OUT / "e14_results.json", "w"), indent=1)
    print("E14_DONE")


if __name__ == "__main__":
    main()
