#!/usr/bin/env python3
"""E9b: MIMIC-CXR external check against RADIOLOGIST labels (mimic-cxr-2.1.0-test-set-labeled.csv), independent of our
LLM extractor. (~/venvs/anchor, GPU for the probe.)
  - probe zero-shot transfer: per-label AUROC of the probe's study-level head (trained on ReXGradient only)
  - label-level precision / recall / F1 of the claims each system asserts (MedGemma B0, frozen ANCHOR, multi, VLM-free)
Positive = radiologist label 1; label 0 = negative; uncertain (-1) and blank excluded per label. Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e8_final as F  # noqa: E402

MAP = {"pleural effusion": "Pleural Effusion", "pneumothorax": "Pneumothorax", "consolidation": "Consolidation",
       "atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly", "pulmonary edema": "Edema",
       "airspace opacity": "Lung Opacity", "interstitial opacity": "Lung Opacity", "nodule": "Lung Lesion",
       "mass": "Lung Lesion", "fracture": "Fracture", "support device": "Support Devices",
       "mediastinal widening": "Enlarged Cardiomediastinum"}


def main():
    lab = pd.read_csv(HOME / "data/meta/mimic-cxr-2.1.0-test-set-labeled.csv")
    lab["key"] = "m" + lab.study_id.astype(int).astype(str)
    labels = sorted(set(MAP.values()) & set(lab.columns))
    vals = pd.Series(lab[labels].to_numpy().ravel()).value_counts(dropna=False).to_dict()
    print(f"radiologist-labelled MIMIC test studies: {len(lab)} | label value counts: {vals}")
    fz = json.load(open(F.OUT / "freeze.json"))
    df = F.load_cands("mimic")
    vf = df[(df.region_rank < 99) | (df.global_rank < 99)].copy()
    df["score"] = F.apply_scorer(df, fz["scorer_full"], fz["delta"]); vf["score"] = F.apply_scorer(vf, fz["scorer_vlmfree"], fz["delta"])
    keys = sorted(set(df.key) & set(lab.key))
    lab = lab.set_index("key").loc[keys]
    probe = F.probe_arrays("mimic", keys)
    # probe transfer
    print("\nprobe zero-shot transfer (study-level head, trained on ReXGradient train only): AUROC vs radiologist labels")
    res = {"n": len(keys), "probe_auroc": {}}
    for L in labels:
        fs = [f for f, l in MAP.items() if l == L and f in probe["F"]]
        y = lab[L]; m = y.isin([0, 1])
        if not fs or (y[m] == 1).sum() < 10:
            continue
        s = np.array([max(probe["Pg"][probe["pos"][k], probe["F"][f]] for f in fs) if k in probe["pos"] else 0.5 for k in y[m].index])
        res["probe_auroc"][L] = round(E4.auroc(s, y[m] == 1), 4)
        print(f"  {L:<28} n={int(m.sum()):>5} pos={int((y[m] == 1).sum()):>4}  AUROC {res['probe_auroc'][L]:.3f}")
    print(f"  macro AUROC {np.mean(list(res['probe_auroc'].values())):.3f}")
    # label-level claims
    G = F.groups_of("mimic", keys, fz["group_tops"])
    th = pd.Series(-np.inf, index=keys)
    for gname, g in G.items():
        th = np.maximum(th, g.map(lambda v: fz["group_thresholds"].get(f"{gname}={v}", fz["t_full"])).reindex(keys).fillna(fz["t_full"]))
    ev = df[df.key.isin(keys)]
    systems = {"MedGemma B0": ev[ev.from_vlm > 0], "ANCHOR": ev[ev.score >= fz["t_full"]],
               "ANCHOR-multi": ev[ev.score >= ev.key.map(th)], "VLM-free": vf[vf.key.isin(keys) & (vf.score >= fz["t_vlmfree"])]}
    print("\nlabel-level vs radiologists (micro over labels; per-label F1 for key labels)")
    res["systems"] = {}
    for name, adm in systems.items():
        pred = {k: {MAP[f] for f in g.finding if f in MAP} for k, g in adm.groupby("key")}
        tp = fp = fn = 0; per = {}; fps = pd.Series(0.0, index=keys)
        for L in labels:
            y = lab[L]; m = y.isin([0, 1])
            p = np.array([L in pred.get(k, set()) for k in y[m].index]); t = (y[m] == 1).to_numpy()
            a, b, c = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
            tp += a; fp += b; fn += c
            fps[y[m].index[p & ~t]] += 1
            per[L] = round(2 * a / max(2 * a + b + c, 1), 3)
        risk = float((np.minimum(fps, 3) / 3).mean())
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res["systems"][name] = {"precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / max(P + R, 1e-9), 4),
                                "false_labels_per_report": round(fp / len(keys), 3), "capped_false_label_risk": round(risk, 4), "per_label_f1": per}
        print(f"  {name:<13} precision {P:.3f} recall {R:.3f} F1 {2 * P * R / max(P + R, 1e-9):.3f} false labels/report {fp / len(keys):.3f} capped risk vs radiologists {risk:.4f} | "
              + " ".join(f"{k.split()[0]}={v}" for k, v in per.items() if k in ("Pleural Effusion", "Pneumothorax", "Cardiomegaly", "Edema", "Consolidation", "Lung Opacity")))
    json.dump(res, open(F.OUT / "mimic_radiologist_labels.json", "w"), indent=1)
    print("E9_LABELS_DONE")


if __name__ == "__main__":
    main()
