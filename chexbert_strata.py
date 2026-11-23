#!/usr/bin/env python3
"""Independent normal/abnormal strata for the valid cohort from the CheXbert labeller (run in ~/venvs/anchor_eval),
so the D40 normal-vs-abnormal result does not rest only on the E1 LLM extraction.
abnormal = any positive CheXbert condition other than 'No Finding' and 'Support Devices'.
Also scores e1_recheck_v2.csv if it has been graded. Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
CONDITIONS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
              "Support Devices", "No Finding"]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def recheck():
    p, k = HOME / "work/e1_recheck_v2.csv", HOME / "work/e1_recheck_v2_key.csv"
    if not p.exists() or not k.exists():
        return
    s = pd.read_csv(p).merge(pd.read_csv(k), on="row")
    for f in ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding"]:
        s[f] = pd.to_numeric(s[f], errors="coerce")
    if s.finding_and_polarity_ok.notna().sum() < 20:
        print("recheck sheet not graded yet"); return
    for name, sel in [("v1-error rows", ~s.is_control), ("v1-correct controls", s.is_control)]:
        x = s[sel]
        print(f"RECHECK v2 on {name} (n={len(x)}): " + ", ".join(f"{f}={x[f].mean():.3f} (n={int(x[f].notna().sum())})"
              for f in ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding"]))


def main():
    recheck()
    import eval_smoke
    eval_smoke.alias_radeval()
    from radeval.metrics.f1chexbert.f1chexbert import F1CheXbert
    fm = sys.modules[F1CheXbert.__module__]          # 'import a.b.c as m' breaks under the RadEval alias
    names = next((v for v in vars(fm).values() if isinstance(v, (list, tuple)) and len(v) == 14 and all(isinstance(x, str) for x in v)), None)
    print("label names from module:", "found" if names else "not found, using standard CheXbert order")
    names = list(names) if names else CONDITIONS
    keys = sorted(pd.read_parquet(HOME / "work/manifest/flags_valid.parquet").query("cohort_primary").key)
    refs = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key").loc[keys, "findings"].fillna("").tolist()
    lab = np.array(F1CheXbert().get_labels(refs, mode="rrg"))
    print("label matrix", lab.shape, "values", sorted(set(lab.ravel().tolist())))
    keep = [i for i, n in enumerate(names) if n not in ("No Finding", "Support Devices")]
    abn = (lab[:, keep] == 1).any(axis=1)
    cx = pd.DataFrame({"chex_abn": abn}, index=keys)
    print("CheXbert prevalence:", {n: round(float((lab[:, i] == 1).mean()), 3) for i, n in enumerate(names)})
    from strata import study_strata, paired
    e1 = study_strata(HOME / "work/corpus/valid")
    j = cx.join(e1, how="inner")
    print(f"CHEX_vs_E1 n={len(j)} abnormal rate chexbert={j.chex_abn.mean():.3f} e1={j.abn.mean():.3f} "
          f"agreement={(j.chex_abn == j.abn).mean():.3f} kappa={kappa(j.chex_abn, j.abn):.3f}")
    res = {}
    for judge in ["crimson", "green"]:
        f = HOME / f"work/judge/valid/{judge}_scores.parquet"
        if not f.exists():
            continue
        w = pd.read_parquet(f).pivot_table(index="key", columns="cond", values="score")[["b0_medgemma4b", "normal_template"]].dropna().join(j, how="inner")
        for name, sel in [("chex_normal", ~w.chex_abn), ("chex_abnormal", w.chex_abn),
                          ("both_normal", ~w.chex_abn & ~w.abn), ("both_abnormal", w.chex_abn & w.abn)]:
            r = paired(w[sel].rename(columns={"b0_medgemma4b": "sys"}), "sys", "normal_template")
            res[f"{judge}|{name}"] = r
            print(f"{judge:<8} {name:<14} {json.dumps(r)}")
    json.dump(res, open(HOME / "work/judge/valid/strata_chexbert.json", "w"), indent=1)
    print("CHEX_STRATA_DONE")


if __name__ == "__main__":
    main()
