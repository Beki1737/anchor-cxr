#!/usr/bin/env python3
"""E4a: claim-pool ceiling + verification-signal test (no generation; run in ~/venvs/anchor, GPU for kNN).

Why: E3 showed whole-image RAG trades errors ~1:1 (FN -0.07, FP +0.06) and that misses (FN ~2 per abnormal study)
dominate. So ANCHOR must (1) PROPOSE a candidate pool with higher recall than the VLM, then (2) ADMIT selectively.
This script measures both on structured claims (E1 v2 extractor on references, corpus and VLM output):

  pools     VLM (MedGemma B0 claims) | GLOBAL k-NN (RAD-DINO CLS) | REGION k-NN (per ANCHOR region embedding,
            neighbour claims kept only if anatomically compatible with that region)
  ceiling   claim recall of VLM U pool if a perfect verifier admitted only true claims (oracle)
  ranking   recall@M / precision@M when pool claims are ranked by neighbour support (equal claim budget)
  signal    AUROC of neighbour support for telling true from false VLM claims (the M4 neighbour-vote signal),
            global vs region votes

Claim = (canonical finding, laterality); positive = present or uncertain. Match: same finding, compatible side.
Aggregates only.
"""
import argparse
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
RN = ["R_apical", "R_upper", "R_mid", "R_base", "L_apical", "L_upper", "L_mid", "L_base", "R_upper_lobe", "R_middle_lobe",
      "R_lower_lobe", "L_upper_lobe", "L_lower_lobe", "R_lung", "L_lung", "heart", "upper_mediastinum", "cardiomediastinum",
      "trachea", "aortic_arch", "R_hemidiaphragm", "L_hemidiaphragm", "spine", "clavicles"]
CANON = [("pneumothorax", "pneumothorax"), ("effusion", "pleural effusion"), ("consolidation", "consolidation"),
         ("atelecta", "atelectasis"), ("edema", "pulmonary edema"), ("cardiomegaly", "cardiomegaly"),
         ("cardiac enlargement", "cardiomegaly"), ("enlarged cardiac", "cardiomegaly"), ("enlarged heart", "cardiomegaly"),
         ("nodul", "nodule"), ("mass", "mass"), ("fracture", "fracture"), ("scoliosis", "scoliosis"),
         ("degenerative", "degenerative changes"), ("granuloma", "calcified granuloma"), ("hernia", "hiatal hernia"),
         ("emphysema", "hyperinflation"), ("hyperinflat", "hyperinflation"), ("interstitial", "interstitial opacity"),
         ("pneumonia", "consolidation"), ("opacit", "airspace opacity"), ("aortic calcif", "aortic calcification"),
         ("tortuous", "aortic calcification"), ("sternotomy", "postsurgical change"), ("surgical clip", "postsurgical change"),
         ("postsurgical", "postsurgical change"), ("support device", "support device"), ("tube", "support device"),
         ("catheter", "support device"), ("pacemaker", "support device"), ("line", "support device"),
         ("mediastinal widening", "mediastinal widening"), ("elevat", "elevated hemidiaphragm")]


def canon(name):
    n = name.lower().strip()
    if n.startswith("aortic calcif"):
        return "aortic calcification"
    for kw, c in CANON:
        if kw in n:
            return c
    return n


def compat_region(q, c):
    if q == c:
        return True
    side = q[:2] if q[:2] in ("R_", "L_") else None
    if side and c == side + "lung" and q != side + "hemidiaphragm":
        return True
    if side and q == side + "lung" and c.startswith(side) and c != side + "hemidiaphragm":
        return True
    zl = {"R_upper_lobe": {"R_apical", "R_upper"}, "R_middle_lobe": {"R_mid", "R_base"}, "R_lower_lobe": {"R_mid", "R_base"},
          "L_upper_lobe": {"L_apical", "L_upper", "L_mid"}, "L_lower_lobe": {"L_mid", "L_base"},
          "R_hemidiaphragm": {"R_base"}, "L_hemidiaphragm": {"L_base"}}
    if c in zl.get(q, ()) or q in zl.get(c, ()):
        return True
    med = {"heart", "upper_mediastinum", "cardiomediastinum", "trachea", "aortic_arch"}
    return q in med and c in med and "cardiomediastinum" in (q, c)


COMPAT = None


def side_ok(a, b):
    return a == b or "none" in (a, b) or "bilateral" in (a, b)


def load_claims(dirpath):
    """key -> list of (finding, lat, regions) positive claims, de-duplicated."""
    out = {}
    for p in sorted(pathlib.Path(dirpath).glob("shard_*.parquet")):
        d = pd.read_parquet(p, columns=["key", "findings_json"])
        for k, js in zip(d.key, d.findings_json):
            lst = out.setdefault(k, {})
            for f in json.loads(js):
                if f.get("polarity") in ("present", "uncertain"):
                    c = (canon(str(f.get("finding", ""))), f.get("laterality") or "none")
                    lst.setdefault(c, set()).update(f.get("regions") or [])
    return out


def matches(c, ref):
    return any(c[0] == r[0] and side_ok(c[1], r[1]) for r in ref)


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def load_region_emb(split, keys):
    ks, em, ar = [], [], []
    for sp in sorted((HOME / "work/regions" / split).glob("shard_*.npz")):
        z = np.load(sp)
        ks.append(z["keys"]); em.append(z["emb"]); ar.append(z["area"])
    K, E, A = np.concatenate(ks), np.concatenate(em), np.concatenate(ar).astype(np.float32)
    idx = pd.Series(np.arange(len(K)), index=K)
    idx = idx[~idx.index.duplicated()]
    sel = idx.reindex(keys).dropna().astype(int)
    return sel.index.to_numpy(), E[sel.to_numpy()], A[sel.to_numpy()]


def region_knn(qk, Qe, Qa, ck, Ce, Ca, k):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for ri, r in enumerate(RN):
        cm = Ca[:, ri] > 0
        if cm.sum() < k:
            continue
        C = torch.from_numpy(Ce[cm, ri, :].astype(np.float32)).to(dev)
        C = torch.nn.functional.normalize(C, dim=1).half() if dev == "cuda" else torch.nn.functional.normalize(C, dim=1)
        cidx = np.where(cm)[0]
        qm = np.where(Qa[:, ri] > 0)[0]
        for s in range(0, len(qm), 2048):
            qi = qm[s:s + 2048]
            Q = torch.nn.functional.normalize(torch.from_numpy(Qe[qi, ri, :].astype(np.float32)).to(dev), dim=1).to(C.dtype)
            top = (Q @ C.T).float().topk(k, dim=1).indices.cpu().numpy()
            for a, row in zip(qi, top):
                out.setdefault(a, []).append((r, ck[cidx[row]]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--vlm", default="b0_medgemma4b_valid")
    ap.add_argument("--budgets", default="1,2,3,5,8")
    a = ap.parse_args(argv)
    t0 = time.time()
    flags = pd.read_parquet(HOME / "work/manifest/flags_valid.parquet")
    qkeys = sorted(flags[flags.cohort_primary].key)
    ref = load_claims(HOME / "work/corpus_v2/valid")
    corp = load_claims(HOME / "work/corpus_v2/train")
    vlm = load_claims(HOME / "work/corpus_gen" / a.vlm)
    print(f"claims loaded in {time.time() - t0:.0f}s | ref studies {len(ref)} corpus {len(corp)} vlm {len(vlm)}", flush=True)

    glob = pd.read_parquet(HOME / "work/retrieval/global_cls_valid.parquet").set_index("key").nbr
    trf = pd.read_parquet(HOME / "work/manifest/flags_train.parquet")
    corpus_keys = sorted(set(trf[trf.cohort_primary].key) & set(corp))
    qk, Qe, Qa = load_region_emb("valid", qkeys)
    ck, Ce, Ca = load_region_emb("train", corpus_keys)
    print(f"region embeddings: queries {len(qk)} corpus {len(ck)} ({time.time() - t0:.0f}s)", flush=True)
    global COMPAT
    COMPAT = {(q, c) for q in RN for c in RN if compat_region(q, c)}
    rnn = region_knn(qk, Qe, Qa, ck, Ce, Ca, a.k)
    qpos = {k: i for i, k in enumerate(qk)}
    print(f"region kNN done ({time.time() - t0:.0f}s)", flush=True)

    B = [int(x) for x in a.budgets.split(",")]
    acc = {p: {"tp": 0, "n_pool": 0, "studies": 0, "pool_true": 0} for p in ["vlm", "global", "region", "vlm+global", "vlm+region"]}
    rank = {p: {m: [0, 0] for m in B} for p in ["global", "region"]}
    n_ref, sig = 0, {"global": ([], []), "region": ([], [])}
    rows = []
    for q in qkeys:
        R = set(ref.get(q, {}))
        if q not in glob.index or q not in qpos:
            continue
        n_ref += len(R)
        # global pool + support
        gs = {}
        nb = [x for x in glob[q][: a.k]]
        for n in nb:
            for c in corp.get(n, {}):
                gs[c] = gs.get(c, 0) + 1 / len(nb)
        # region pool + support (max over regions of fraction of that region's k neighbours asserting it there)
        rs = {}
        for r, nbrs in rnn.get(qpos[q], []):
            cnt = {}
            for n in nbrs:
                for c, regs in corp.get(n, {}).items():
                    if any((r, x) in COMPAT for x in regs):
                        cnt[c] = cnt.get(c, 0) + 1
            for c, v in cnt.items():
                rs[c] = max(rs.get(c, 0), v / len(nbrs))
        V = set(vlm.get(q, {}))
        pools = {"vlm": V, "global": set(gs), "region": set(rs), "vlm+global": V | set(gs), "vlm+region": V | set(rs)}
        covered = {p: sum(matches(r_, P) for r_ in R) for p, P in pools.items()}
        for p, P in pools.items():
            acc[p]["tp"] += covered[p]; acc[p]["n_pool"] += len(P); acc[p]["studies"] += 1
            acc[p]["pool_true"] += sum(matches(c, R) for c in P)
        for p, S in [("global", gs), ("region", rs)]:
            order = sorted(S, key=lambda c: -S[c])
            for m in B:
                top = order[:m]
                rank[p][m][0] += sum(matches(r_, top) for r_ in R)
                rank[p][m][1] += sum(matches(c, R) for c in top)
        for c in V:
            y = matches(c, R)
            sig["global"][0].append(gs.get(c, 0.0)); sig["global"][1].append(y)
            sig["region"][0].append(rs.get(c, 0.0)); sig["region"][1].append(y)
        rows.append({"key": q, "n_ref": len(R), "vlm_recall_hits": covered["vlm"], "vlmregion_hits": covered["vlm+region"]})

    res = {"k": a.k, "n_studies": acc["vlm"]["studies"], "n_ref_claims": n_ref}
    print(f"\nE4a  k={a.k}  studies={res['n_studies']}  reference positive claims={n_ref}")
    print(f"{'pool':<12}{'oracle recall':>15}{'mean pool size':>16}{'pool precision':>16}")
    for p, v in acc.items():
        prec = v["pool_true"] / max(v["n_pool"], 1)
        res[p] = {"oracle_recall": round(v["tp"] / max(n_ref, 1), 4), "mean_pool": round(v["n_pool"] / max(v["studies"], 1), 2),
                  "precision": round(prec, 4)}
        print(f"{p:<12}{res[p]['oracle_recall']:>15.3f}{res[p]['mean_pool']:>16.2f}{prec:>16.3f}")
    print("\nranked by neighbour support, equal claim budget M per study: recall@M / precision@M")
    for p in ["global", "region"]:
        cells = []
        for m in B:
            rec = rank[p][m][0] / max(n_ref, 1); pre = rank[p][m][1] / max(m * res["n_studies"], 1)
            cells.append(f"M={m}: {rec:.3f}/{pre:.3f}")
            res.setdefault(f"{p}_ranked", {})[m] = [round(rec, 4), round(pre, 4)]
        print(f"  {p:<8} " + "  ".join(cells))
    for p in ["global", "region"]:
        res[f"auroc_{p}_votes_on_vlm_claims"] = round(auroc(*sig[p]), 4)
    print(f"\nverification signal (true vs false VLM claims, n={len(sig['global'][1])}, VLM claim precision "
          f"{np.mean(sig['global'][1]):.3f}): AUROC global votes {res['auroc_global_votes_on_vlm_claims']:.3f} | "
          f"region votes {res['auroc_region_votes_on_vlm_claims']:.3f}")
    out = HOME / "work/e4a"; out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"e4a_k{a.k}.json", "w"), indent=1)
    pd.DataFrame(rows).to_parquet(out / f"e4a_k{a.k}_per_study.parquet", index=False)
    print(f"E4A_DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
