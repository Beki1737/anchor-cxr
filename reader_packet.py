#!/usr/bin/env python3
"""D74 reader-study packet (CPU only; ~/venvs/anchor). Everything stays on devbox in ~/anchor/work/reader/ and may only be
given to a reader who has accepted the ReXGradient-160K terms (contains report text and images). Never upload it.

  reader_sheet.csv   110 items: 100 test studies (40 key-abnormal, 30 other-abnormal, 30 normal by the E1 strata, seed 7)
                     + 10 hidden repeats for intra-reader reliability. Each item: image file, reference findings, report A,
                     report B (MedGemma B0 vs ANCHOR-edit, side randomized per item), empty grading columns.
  reader_key.csv     item -> study key, which system is A/B, stratum, repeat_of. KEEP AWAY FROM THE READER.
  images/item_NNN.png  copy of the frontal image
  e1_rater2.csv      held-out E1 extractor sheet with the first rater's grades removed (second rater -> kappa)
Prints counts only."""
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import study_strata  # noqa: E402

OUT = HOME / "work/reader"
GRADE = ["false_significant", "missed_significant", "wrong_side", "false_minor", "missed_minor"]


def main():
    if (OUT / "reader_key.csv").exists():
        print("packet exists; delete ~/anchor/work/reader to rebuild"); return
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    keys = set(pd.read_parquet(HOME / "work/e8/eval_keys_test.parquet").key)
    st = study_strata(HOME / "work/corpus_v2/test")
    st = st[st.index.isin(keys)]
    gen = {s: pd.read_parquet(HOME / f"work/gen/{f}_test.parquet").set_index("key").generated for s, f in
           [("B0", "b0_medgemma4b"), ("ANCHOR-edit", "e8edit")]}
    man = pd.read_parquet(HOME / "work/manifest/rex_test.parquet").set_index("key")
    rng = np.random.default_rng(7)
    pick = []
    for name, sel, n in [("key_abnormal", st.key_abn, 40), ("other_abnormal", st.abn & ~st.key_abn, 30), ("normal", ~st.abn, 30)]:
        ks = sorted(st.index[sel & st.index.isin(gen["B0"].index) & st.index.isin(gen["ANCHOR-edit"].index)])
        pick += [(k, name) for k in rng.choice(ks, n, replace=False)]
    rep = [pick[i] for i in rng.choice(len(pick), 10, replace=False)]
    items = [(k, s, None) for k, s in pick] + [(k, s, "repeat") for k, s in rep]
    order = rng.permutation(len(items))
    rows, key = [], []
    first = {}
    for it, j in enumerate(order, 1):
        k, s, r = items[j]
        a_is_anchor = bool(rng.integers(0, 2))
        A, B = ("ANCHOR-edit", "B0") if a_is_anchor else ("B0", "ANCHOR-edit")
        img = f"images/item_{it:03d}.png"
        src = man.at[k, "frontal_path"]
        if src and pathlib.Path(src).exists():
            shutil.copy(src, OUT / img)
        row = {"item": it, "image": img, "reference_findings": man.at[k, "findings"], "report_A": gen[A][k], "report_B": gen[B][k]}
        for side in "AB":
            row.update({f"{side}_{g}": "" for g in GRADE})
        row.update({"preferred": "", "confidence_1to5": "", "comment": ""})
        rows.append(row)
        key.append({"item": it, "key": k, "A": A, "B": B, "stratum": s, "repeat_of": first.get(k, "")})   # 2nd showing -> 1st item
        first.setdefault(k, it)
    pd.DataFrame(rows).to_csv(OUT / "reader_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(OUT / "reader_key.csv", index=False)
    print(f"reader packet: {len(rows)} items ({len(pick)} studies + {len(rep)} repeats) -> {OUT}")
    print(pd.DataFrame(key).stratum.value_counts().to_dict())
    h = HOME / "work/e1_label_200_v2_heldout.csv"
    if h.exists():
        e = pd.read_csv(h)
        for c in ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding", "note", "err_code"]:
            if c in e:
                e[c] = ""
        e.to_csv(OUT / "e1_rater2.csv", index=False)
        print(f"E1 second-rater sheet: {len(e)} rows -> e1_rater2.csv (first rater's grades removed)")
    print("READER_PACKET_DONE")


if __name__ == "__main__":
    main()
