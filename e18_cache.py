#!/usr/bin/env python3
"""E18 cache (round 44): score the frozen ANCHOR and FUS candidate pools on VALID and MIMIC once, so the council
experiments E18 (two-sided certification), E19 (anytime-valid monitoring) and E20 (label-free shift reweighting) run
on CPU. Never touches the ReXGradient test split. (~/venvs/anchor, small GPU for the probe; GPUs 0/1 only)
Writes work/e18/{anchor,fus}_{valid,mimic}.parquet (key, patient, finding, lat, label, score, fold),
work/e18/studies_{split}.parquet (key, patient, fold, view, t_anchor_multi, t_fus_multi) and
work/e18/feat_{split}.npz (keys, RAD-DINO CLS, global-probe probabilities). Frozen scorers and thresholds only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e8_final as F  # noqa: E402
import e16_fusion as FU  # noqa: E402

OUT = HOME / "work/e18"
COLS = ["key", "patient", "finding", "lat", "label", "score", "fold"]


def probe_and_cls(split, keys):
    """Same as e8_final.probe_arrays, but also returns the CLS embeddings it already loads."""
    import torch
    import e4c_probe as P
    reg, glo, vocab = P.load_probe(torch)
    ks, Em, Ar, Cl = P.load_split(split, sorted(keys))
    Pr, Pg = P.predict(torch, reg, glo, Em, Cl)
    probe = {"pos": {k: i for i, k in enumerate(ks)}, "F": {f: j for j, f in enumerate(vocab)}, "Pr": Pr, "Pg": Pg, "Ar": Ar}
    return probe, np.array(ks), np.asarray(Cl, np.float32), Pg, list(vocab)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(F.OUT / "freeze.json")); fz2 = json.load(open(FU.OUT / "freeze2.json"))
    for split in ["valid", "mimic"]:
        C = F.load_cands(split); ref = E4.load_claims(HOME / f"work/corpus_v2/{split}")
        keys = sorted(C.key.unique())
        probe, ks, cls, pg, vocab = probe_and_cls(split, keys)
        ex, fu = FU.build(split, C, probe, ref, fz2["tau"])
        fu["score"] = F.apply_scorer(fu, fz2["scorer_fus"], fz2["delta"])
        C["score"] = F.apply_scorer(C, fz["scorer_full"], fz["delta"])
        C[COLS].to_parquet(OUT / f"anchor_{split}.parquet", index=False)
        fu[COLS].to_parquet(OUT / f"fus_{split}.parquet", index=False)
        tf = FU.group_thr(split, fu, keys, {"group_thresholds": fz2["group_thresholds_fus"]}, fz2["t_fus"], tops=fz2["group_tops"])
        ta = FU.group_thr(split, C, keys, {"group_thresholds": fz["group_thresholds"]}, fz["t_full"], tops=fz["group_tops"])
        man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key").reindex(keys)
        fold = C.drop_duplicates("key").set_index("key").fold.reindex(keys)
        st = pd.DataFrame({"key": keys, "patient": man.patient.astype(str).to_numpy(), "fold": fold.to_numpy(),
                           "view": man.frontal_view.astype(str).to_numpy(), "t_anchor_multi": ta.reindex(keys).to_numpy(),
                           "t_fus_multi": tf.reindex(keys).to_numpy()})
        st.to_parquet(OUT / f"studies_{split}.parquet", index=False)
        np.savez_compressed(OUT / f"feat_{split}.npz", keys=ks, cls=cls.astype(np.float16), pg=np.asarray(pg, np.float16), vocab=np.array(vocab))
        print(f"{split}: studies {len(keys)} | ANCHOR candidates {len(C)} | FUS candidates {len(fu)} | features {cls.shape} probe {np.shape(pg)} | "
              f"multi thresholds ANCHOR {np.nanmin(ta):.3f}-{np.nanmax(ta):.3f}, FUS {np.nanmin(tf):.3f}-{np.nanmax(tf):.3f}", flush=True)
    json.dump({"t_anchor": fz["t_full"], "t_fus": fz2["t_fus"]}, open(OUT / "thresholds.json", "w"))
    print("E18_CACHE_DONE")


if __name__ == "__main__":
    main()
