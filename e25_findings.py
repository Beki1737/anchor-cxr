#!/usr/bin/env python3
"""E25 (round 48): CLINICALLY IMPORTANT FINDINGS (Prof. Jang, point 4). Descriptive analysis of FROZEN systems, no
selection. ~/venvs/anchor, small GPU for the probe.
  python e25_findings.py --split valid      folds 2-3 (default)      | --split mimic   external + radiologist labels
  python e25_findings.py --split test       GATED (post hoc analysis of the frozen one-shot outputs)
Per finding (pneumothorax, nodule, mass, pleural effusion, consolidation, oedema, fracture, airspace opacity) and
system (MedGemma-4B draft, ANCHOR, ANCHOR-multi, FUS-multi): reference count, TP, FP, FN, precision, recall, false and
missed findings per 1,000 reports; patient-cluster 95% CIs for ANCHOR and FUS-multi minus the draft. Lenient side
matching (same as the paper). On MIMIC also study-level precision / recall against the radiologists' labels.
Aggregates only."""
import argparse
import json

import numpy as np
import pandas as pd

import e2x_common as X

E4, A5, F = X.E4, X.A5, X.F
RAD = {"pneumothorax": ["Pneumothorax"], "pleural effusion": ["Pleural Effusion"], "consolidation": ["Consolidation"],
       "pulmonary edema": ["Edema"], "fracture": ["Fracture"], "nodule": ["Lung Lesion"], "mass": ["Lung Lesion"],
       "airspace opacity": ["Airspace Opacity"]}


def systems(split, keys, probe, ref):
    import e16_fusion as FU
    fz = json.load(open(F.OUT / "freeze.json")); fz2 = json.load(open(FU.OUT / "freeze2.json"))
    C = F.load_cands(split); C["score"] = F.apply_scorer(C, fz["scorer_full"], fz["delta"])
    fin = X.final_claims(C, probe, fz["tau"], ref)
    _, fu = FU.build(split, C, probe, ref, fz2["tau"]); fu["score"] = F.apply_scorer(fu, fz2["scorer_fus"], fz2["delta"])
    finf = X.final_claims(fu, probe, fz2["tau"], ref)
    S = {"MedGemma-4B draft": X.raw_claims(C[(C.from_vlm > 0) & C.key.isin(set(keys))], ref),
         "ANCHOR": X.admit(fin, keys, fz["t_full"]),
         "ANCHOR-multi": X.admit(fin, keys, X.multi_thresholds(split, keys, fz["group_thresholds"], fz["t_full"], fz["group_tops"])),
         "FUS-multi": X.admit(finf, keys, X.multi_thresholds(split, keys, fz2["group_thresholds_fus"], fz2["t_fus"], fz2["group_tops"]))}
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="valid", choices=["valid", "mimic", "test"])
    a = ap.parse_args(argv)
    ref = E4.load_claims(X.HOME / f"work/corpus_v2/{a.split}")
    C0 = F.load_cands(a.split)
    keys = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key) if a.split == "valid" else sorted(C0.key.unique())
    probe = F.probe_arrays(a.split, sorted(C0.key.unique()))
    S = systems(a.split, keys, probe, ref)
    pat = X.patients(a.split, keys) if a.split != "mimic" else pd.read_parquet(X.HOME / "work/manifest/rex_mimic.parquet").set_index("key").reindex(keys).patient.astype(str).to_numpy()
    N = len(keys); out = {}
    print(f"=== E25 {a.split.upper()} ({N} studies): clinically important findings, extractor references, lenient side ===")
    print(f"{'finding':<18}{'refs':>6} | " + " | ".join(f"{s[:14]:>14} P/R  FP/1k FN/1k" for s in S))
    for f in X.KEYF:
        row, per = {}, {}
        for s, adm in S.items():
            tp, fp, cov, n = X.per_study_finding(adm, ref, keys, f); per[s] = (tp, fp, cov, n)
            P = tp.sum() / max(tp.sum() + fp.sum(), 1); R = cov.sum() / max(n.sum(), 1)
            row[s] = {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(n.sum() - cov.sum()), "precision": round(float(P), 4),
                      "recall": round(float(R), 4), "fp_per_1k": round(1000 * fp.sum() / N, 1), "fn_per_1k": round(1000 * (n.sum() - cov.sum()) / N, 1)}
        nref = int(per["ANCHOR"][3].sum())
        print(f"{f:<18}{nref:>6} | " + " | ".join(f"{row[s]['precision']:.2f}/{row[s]['recall']:.2f} {row[s]['fp_per_1k']:>5.0f} {row[s]['fn_per_1k']:>5.0f}" for s in S))
        b = per["MedGemma-4B draft"]
        for s in ["ANCHOR", "FUS-multi"]:
            t = per[s]
            rlo, rhi = X.cluster_ratio_diff(b[2], t[2], b[3], t[3], pat)
            plo, phi = X.cluster_ratio_diff(b[0], t[0], b[0] + b[1], t[0] + t[1], pat)
            row[f"{s}_minus_draft"] = {"recall_ci": [rlo, rhi], "precision_ci": [plo, phi],
                                       "fp_per_1k": row[s]["fp_per_1k"] - row["MedGemma-4B draft"]["fp_per_1k"]}
            print(f"{'':<24}{s} minus draft: recall {row[s]['recall'] - row['MedGemma-4B draft']['recall']:+.3f} [{rlo:+.3f}, {rhi:+.3f}], "
                  f"precision {row[s]['precision'] - row['MedGemma-4B draft']['precision']:+.3f} [{plo:+.3f}, {phi:+.3f}], "
                  f"false/1k {row[s]['fp_per_1k'] - row['MedGemma-4B draft']['fp_per_1k']:+.0f}")
        out[f] = row
    if a.split == "mimic":
        lab = pd.read_csv(X.HOME / "data/meta/mimic-cxr-2.1.0-test-set-labeled.csv"); lab["key"] = "m" + lab.study_id.astype(int).astype(str)
        lab = lab.set_index("key")
        kk = [k for k in keys if k in lab.index]
        print(f"\nMIMIC study level vs RADIOLOGIST labels ({len(kk)} labelled studies; positive = label 1, negative = 0)")
        out["radiologist"] = {}
        for L in sorted({l for v in RAD.values() for l in v}):
            fs = [f for f, v in RAD.items() if L in v]
            if L not in lab.columns:
                print(f"  {L:<18} column missing in label file, skipped"); continue
            y = lab.loc[kk, L]; m = y.isin([0, 1]).to_numpy(); yy = (y.to_numpy() == 1)[m]
            r = {}
            for s, adm in S.items():
                pos = set(adm[adm.finding.isin(fs)].key)
                p = np.array([k in pos for k in kk])[m]
                tp = int((p & yy).sum()); r[s] = {"precision": round(tp / max(p.sum(), 1), 3), "recall": round(tp / max(yy.sum(), 1), 3),
                                                  "fp": int((p & ~yy).sum()), "fn": int((~p & yy).sum())}
            out["radiologist"][L] = {"n": int(m.sum()), "pos": int(yy.sum()), **r}
            print(f"  {L:<18} n={int(m.sum()):>4} pos={int(yy.sum()):>4} | " + " | ".join(f"{s[:12]} P {r[s]['precision']:.2f} R {r[s]['recall']:.2f} FP {r[s]['fp']}" for s in S))
    json.dump(out, open(X.HOME / f"work/e8/e25_{a.split}.json", "w"), indent=1, default=float)
    print("E25_DONE")


if __name__ == "__main__":
    main()
