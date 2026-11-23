#!/usr/bin/env python3
"""One-page E3 summary from saved judge outputs (aggregates only, safe to share).
Systems: normal template, MedGemma B0, ret1 (copy top-1 neighbour), E3 RAG (MedGemma + 3 neighbours).
- CRIMSON / GREEN means, paired diffs vs template and vs B0
- CRIMSON error anatomy per study: false findings and missing findings, by stratum (E1 v2 labels)
- CheXbert-strata check (r23) if present"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
J = HOME / "work/judge"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import study_strata  # noqa: E402

SYS = ["normal_template", "b0_medgemma4b", "ret1_cls", "e3rag_medgemma4b_k3"]


def load(judge):
    parts = [pd.read_parquet(J / t / f"{judge}_scores.parquet") for t in ["valid", "valid_e3"] if (J / t / f"{judge}_scores.parquet").exists()]
    return pd.concat(parts).drop_duplicates(["cond", "key"]) if parts else None


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus_v2"
    st = study_strata(HOME / "work" / corpus / "valid")
    print(f"strata from {corpus}: normal {int((st.stratum == 'normal').sum())}, abnormal {int((st.stratum == 'abnormal').sum())}")
    for judge in ["crimson", "green"]:
        sc = load(judge)
        if sc is None:
            continue
        print(f"\n=== {judge.upper()} (valid primary cohort) ===")
        W = sc.pivot_table(index="key", columns="cond", values="score").join(st, how="inner")
        for s in [x for x in SYS if x in W]:
            row = [f"{s:<22}"]
            for name in ["all", "normal", "abnormal"]:
                x = W if name == "all" else W[W.stratum == name]
                row.append(f"{name}={x[s].mean():.3f}")
            print("  ".join(row))
        for f in sorted((J / "valid_e3").glob(f"{judge}_paired_*.json")):
            r = json.load(open(f))
            print(f"PAIRED {r['a']} vs {r['b']}: diff={r['diff']:+.4f} CI={r['ci95']} win/tie/loss={r['win']}/{r['tie']}/{r['loss']} n={r['n']}")
        if judge == "crimson" and "n_false_findings" in sc.columns:
            print("\nCRIMSON error anatomy (mean per study): false findings | missing findings")
            E = sc.merge(st[["stratum"]], left_on="key", right_index=True, how="inner")
            for s in [x for x in SYS if x in set(E.cond)]:
                x = E[E.cond == s]
                cells = [f"all: FP={x.n_false_findings.mean():.2f} FN={x.n_missing_findings.mean():.2f}"]
                for name in ["normal", "abnormal"]:
                    y = x[x.stratum == name]
                    cells.append(f"{name}: FP={y.n_false_findings.mean():.2f} FN={y.n_missing_findings.mean():.2f}")
                print(f"  {s:<22} " + " | ".join(cells))
    for f in ["strata_chexbert.json"]:
        p = J / "valid" / f
        if p.exists():
            print(f"\n=== {f} (MedGemma B0 vs template, CheXbert strata) ===")
            for k, v in json.load(open(p)).items():
                print(f"{k:<26} n={v['n']} sys={v['mean_sys']} template={v['mean_template']} diff={v['diff']:+.4f} CI={v['ci95']}")
    print("E3_SUMMARY_DONE")


if __name__ == "__main__":
    main()
