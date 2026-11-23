#!/usr/bin/env python3
"""E5 summary on the evaluation half (patients never used to fit the scorer or calibrate CRC). Aggregates only.
Per judge: mean all / normal / abnormal, paired diff vs MedGemma B0 and vs template (patient-free study bootstrap),
CRIMSON false / missing findings per report."""
import json
import pathlib
import sys

import pandas as pd

HOME = pathlib.Path.home() / "anchor"
J = HOME / "work/judge"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import paired, study_strata  # noqa: E402

SYS = ["normal_template", "b0_medgemma4b", "e3rag_medgemma4b_k3", "e5b0real", "e5anchor_any15", "e5anchor_any30",
       "e5anchor_fp15", "e5anchor_fp30", "e5anchorsr_fp30", "e5anchorsr_any30", "e5poolonly_fp30", "e5prior_fp30", "e7edit_fp30", "e7edit_any15", "e7sidexp_fp30",
       "e7focal_fp30", "e7multi_any30", "e5oracle"]


def main():
    keys = set(pd.read_parquet(HOME / "work/e5/eval_keys.parquet").key)
    st = study_strata(HOME / "work/corpus_v2/valid")
    st = st[st.index.isin(keys)]
    print(f"E5 evaluation set: {len(keys)} studies (normal {int((st.stratum == 'normal').sum())}, abnormal {int((st.stratum == 'abnormal').sum())})")
    out = {}
    for judge in ["crimson", "green"]:
        parts = [pd.read_parquet(J / t / f"{judge}_scores.parquet") for t in ["valid", "valid_e3", "valid_e5", "valid_e6", "valid_e7"]
                 if (J / t / f"{judge}_scores.parquet").exists()]
        sc = pd.concat(parts).drop_duplicates(["cond", "key"])
        sc = sc[sc.key.isin(keys)]
        W = sc.pivot_table(index="key", columns="cond", values="score").join(st, how="inner")
        print(f"\n=== {judge.upper()} ===")
        print(f"{'system':<22}{'all':>7}{'normal':>8}{'abnormal':>10}   paired vs B0 (all)              paired vs template (all)        paired vs ANCHOR fp30 (all)")
        for s in [x for x in SYS if x in W]:
            r = {n: W[s].mean() if n == "all" else W[W.stratum == n][s].mean() for n in ["all", "normal", "abnormal"]}
            line = f"{s:<22}{r['all']:>7.3f}{r['normal']:>8.3f}{r['abnormal']:>10.3f}"
            for base in ["b0_medgemma4b", "normal_template", "e5anchor_fp30"]:
                if s != base and base in W:
                    x = W[[s, base]].dropna().rename(columns={s: "sys"})
                    p = paired(x, "sys", base)
                    out[f"{judge}|{s}|vs_{base}"] = p
                    line += f"   {p['diff']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]"
                else:
                    line += "   " + " " * 26
            print(line)
        if judge == "crimson" and "n_false_findings" in sc.columns:
            if "n_location_errors" in sc.columns:
                print("CRIMSON location / severity / certainty attribute errors per report:")
                for s_ in [x for x in SYS if x in set(sc.cond)]:
                    x = sc[sc.cond == s_]
                    print(f"  {s_:<22} location {x.n_location_errors.mean():.3f}  severity {x.n_severity_errors.mean():.3f}  "
                          f"certainty {x.n_certainty_errors.mean():.3f}  temporal {x.n_temporal_errors.mean():.3f}")
            print("CRIMSON false / missing findings per report (all | normal | abnormal):")
            E = sc.merge(st[["stratum"]], left_on="key", right_index=True)
            for s in [x for x in SYS if x in set(E.cond)]:
                x = E[E.cond == s]
                cells = [f"{x.n_false_findings.mean():.2f}/{x.n_missing_findings.mean():.2f}"]
                for n in ["normal", "abnormal"]:
                    y = x[x.stratum == n]; cells.append(f"{y.n_false_findings.mean():.2f}/{y.n_missing_findings.mean():.2f}")
                print(f"  {s:<22} " + " | ".join(cells))
    json.dump(out, open(HOME / "work/e5/e5_summary.json", "w"), indent=1)
    print("E5_SUMMARY_DONE")


if __name__ == "__main__":
    main()
