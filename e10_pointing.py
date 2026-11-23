#!/usr/bin/env python3
"""E10: external zero-shot GROUNDING on PadChest-GR (radiologist boxes), no box supervision anywhere (~/venvs/anchor).
The region probe was trained on ReXGradient report text only. For every abnormal PadChest-GR finding whose label maps
to the probe vocabulary and that has boxes:
  pointing hit   centroid of the region the probe ranks highest for that finding falls inside a radiologist box
                 (2.5% margin); baselines: finding-specific anatomical PRIOR (most frequent region of that finding in the
                 ReXGradient training reports, no image) and RANDOM region (exact expectation over visible regions)
  box coverage   share of the union of boxes covered by the predicted region mask (>= 0.10 counts as overlap)
  side           for single-side findings (all boxes left or right of the midline): right-vs-left from region probe
                 (s_R - s_L) vs prior majority side; AUROC and accuracy
Coordinates: boxes are normalised to the full image; the region cache works on the centre square -> transformed.
Image left = patient right. Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402

MAP = {"cardiomegaly": "cardiomegaly", "pleural effusion": "pleural effusion", "costophrenic angle blunting": "pleural effusion",
       "interstitial pattern": "interstitial opacity", "infiltrates": "airspace opacity", "alveolar pattern": "airspace opacity",
       "increased density": "airspace opacity", "consolidation": "consolidation", "laminar atelectasis": "atelectasis",
       "atelectasis": "atelectasis", "lobar atelectasis": "atelectasis", "segmental atelectasis": "atelectasis",
       "nodule": "nodule", "multiple nodules": "nodule", "mass": "mass", "calcified granuloma": "calcified granuloma",
       "callus rib fracture": "fracture", "fracture": "fracture", "rib fracture": "fracture", "vertebral fracture": "fracture",
       "scoliosis": "scoliosis", "vertebral degenerative changes": "degenerative changes", "aortic atheromatosis": "aortic calcification",
       "aortic elongation": "aortic calcification", "hiatal hernia": "hiatal hernia", "hemidiaphragm elevation": "elevated hemidiaphragm",
       "diaphragmatic eventration": "elevated hemidiaphragm", "pacemaker": "support device", "NSG tube": "support device",
       "central venous catheter via jugular vein": "support device", "central venous catheter": "support device",
       "endotracheal tube": "support device", "sternotomy": "postsurgical change", "suture material": "postsurgical change",
       "air trapping": "hyperinflation", "emphysema": "hyperinflation", "pneumothorax": "pneumothorax",
       "pulmonary edema": "pulmonary edema", "heart insufficiency": "pulmonary edema", "osteopenia": "osteopenia"}
RIGHT = [i for i, r in enumerate(E4.RN) if r.startswith("R_")]
LEFT = [i for i, r in enumerate(E4.RN) if r.startswith("L_")]


def load_regions(split):
    out = {k: [] for k in ["keys", "emb", "cls", "area", "m64", "h", "w"]}
    for sp in sorted((HOME / "work/regions" / split).glob("shard_*.npz")):
        z = np.load(sp)
        for k in out:
            out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def to_square(b, h, w):
    s = min(h, w); y0, x0 = (h - s) / 2, (w - s) / 2
    return np.clip([(b[0] * w - x0) / s, (b[1] * h - y0) / s, (b[2] * w - x0) / s, (b[3] * h - y0) / s], 0, 1)


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return (xs.mean() + 0.5) / mask.shape[1], (ys.mean() + 0.5) / mask.shape[0]


def inside(pt, boxes, m=0.025):
    return any(b[0] - m <= pt[0] <= b[2] + m and b[1] - m <= pt[1] <= b[3] + m for b in boxes)


def coverage(mask, boxes):
    g = np.zeros_like(mask, dtype=bool); n = mask.shape[0]
    for b in boxes:
        g[int(b[1] * n):max(int(np.ceil(b[3] * n)), int(b[1] * n) + 1), int(b[0] * n):max(int(np.ceil(b[2] * n)), int(b[0] * n) + 1)] = True
    return (mask & g).sum() / max(g.sum(), 1)


FIXED = {"cardiomegaly", "scoliosis", "degenerative changes", "hiatal hernia", "aortic calcification", "hyperinflation"}


def mcnemar(a, b):
    from math import comb
    x, y = int((a & ~b).sum()), int((~a & b).sum()); n = x + y
    return x, y, (min(1.0, 2 * sum(comb(n, k) for k in range(0, min(x, y) + 1)) / 2 ** n) if n else 1.0)


def extended(T, out):
    """E10b (added after E10; fixed rules, nothing tuned on PadChest): location-variable subset, probe x prior hybrid,
    side analysis restricted to findings ANCHOR lateralises, frozen resolver decisions (tau from E6)."""
    import e6_deep as D
    fz = json.load(open(HOME / "work/e8/freeze.json"))
    print("\n#### E10b ####")
    for nm, g in [("ALL", T), ("location-variable (excl. fixed-anatomy findings)", T[~T.finding.isin(FIXED)]), ("fixed-anatomy", T[T.finding.isin(FIXED)])]:
        b, c, p = mcnemar(g.hit_probe, g.hit_prior); bh, ch, ph = mcnemar(g.hit_hybrid, g.hit_prior)
        print(f"  {nm:<50} n={len(g):>5} hit probe {g.hit_probe.mean():.3f} | prior {g.hit_prior.mean():.3f} | hybrid {g.hit_hybrid.mean():.3f} "
              f"| random {g.hit_random.mean():.3f} | overlap probe {(g.cov_probe >= .1).mean():.3f} prior {(g.cov_prior >= .1).mean():.3f} "
              f"hybrid {(g.cov_hybrid >= .1).mean():.3f} | McNemar probe-prior {b}/{c} p={p:.2g}, hybrid-prior {bh}/{ch} p={ph:.2g}")
        out[f"E10b|{nm}"] = {"n": len(g), "hit_probe": round(g.hit_probe.mean(), 4), "hit_prior": round(g.hit_prior.mean(), 4),
                              "hit_hybrid": round(g.hit_hybrid.mean(), 4), "hit_random": round(g.hit_random.mean(), 4),
                              "mcnemar_probe_prior": [b, c, p], "mcnemar_hybrid_prior": [bh, ch, ph]}
    S = T[T.single_side & T.sr.notna() & T.sl.notna() & T.finding.isin(D.LATF)].copy()
    acc = ((S.sr >= S.sl) == S.true_right)
    bal = 0.5 * (acc[S.true_right].mean() + acc[~S.true_right].mean())
    dec = np.where(np.minimum(S.sr, S.sl) >= fz["tau"], "bilateral", np.where(S.sr >= S.sl, "right", "left"))
    one = dec != "bilateral"
    cor = (dec == np.where(S.true_right, "right", "left"))
    print(f"  side, lateralised findings only (n={len(S)}): AUROC {E4.auroc(S.sr - S.sl, S.true_right):.3f} | argmax accuracy {acc.mean():.3f} "
          f"(balanced {bal:.3f}; right {acc[S.true_right].mean():.3f}, left {acc[~S.true_right].mean():.3f}) | prior majority "
          f"{(S.prior_right == S.true_right).mean():.3f}")
    print(f"  frozen resolver (tau {fz['tau']}): says one side for {one.mean():.3f}, correct side among those {cor[one].mean():.3f}, "
          f"'bilateral' (hedged) {1 - one.mean():.3f}, wrong side overall {(one & ~cor).mean():.3f}")
    out["E10b|side_latf"] = {"n": len(S), "auroc": round(E4.auroc(S.sr - S.sl, S.true_right), 4), "acc": round(float(acc.mean()), 4),
                             "balanced_acc": round(float(bal), 4), "resolver_one_side": round(float(one.mean()), 4),
                             "resolver_correct_given_one_side": round(float(cor[one].mean()), 4), "resolver_wrong_side": round(float((one & ~cor).mean()), 4)}


def main():
    import torch
    import e4c_probe as P
    reg, glo, vocab = P.load_probe(torch)
    F = {f: j for j, f in enumerate(vocab)}
    R = load_regions("padchest")
    pos = {k: i for i, k in enumerate(R["keys"])}
    Pr, Pg = P.predict(torch, reg, glo, R["emb"], R["cls"])
    man = pd.read_parquet(HOME / "work/manifest/rex_padchest.parquet")
    d = json.load(open(HOME / "data/padchest_gr/grounded_reports_20240819.json"))
    # anatomical prior from ReXGradient TRAIN report claims
    corp = E4.load_claims(HOME / "work/corpus_v2/train")
    prior, side_prior = {}, {}
    for cl in corp.values():
        for (f, lat), regs in cl.items():
            for r in regs:
                if r in E4.RN:
                    prior.setdefault(f, {}).setdefault(r, 0); prior[f][r] += 1
            if lat in ("right", "left"):
                side_prior.setdefault(f, {"right": 0, "left": 0})[lat] += 1
    rows, unmapped = [], {}
    for k, s in zip(man.key, d):
        i = pos.get(k)
        if i is None:
            continue
        h, w = int(R["h"][i]), int(R["w"][i])
        masks = np.unpackbits(R["m64"][i], axis=-1)[:, :4096].reshape(24, 64, 64).astype(bool)
        vis = [r for r in range(24) if R["area"][i, r] > 0 and masks[r].any()]
        for f in s["findings"]:
            if not f.get("abnormal") or not f.get("boxes"):
                continue
            labs = [MAP[l] for l in (f.get("labels") or []) if l in MAP and MAP[l] in F]
            for l in (f.get("labels") or []):
                if l not in MAP:
                    unmapped[l] = unmapped.get(l, 0) + 1
            if not labs or not vis:
                continue
            fn = labs[0]; j = F[fn]
            boxes = [to_square(b, h, w) for b in f["boxes"]]
            rp = max(vis, key=lambda r: Pr[i, r, j])
            pf = prior.get(fn, {}); tot = sum(pf.values())
            rh = max(vis, key=lambda r: Pr[i, r, j] * (pf.get(E4.RN[r], 0) + 1) / (tot + 24))   # pre-specified: probe x Laplace prior
            pr_order = sorted(prior.get(fn, {}).items(), key=lambda x: -x[1])
            rprior = next((E4.RN.index(r) for r, _ in pr_order if E4.RN.index(r) in vis), vis[0])
            hit_p = inside(centroid(masks[rp]), boxes); hit_q = inside(centroid(masks[rprior]), boxes)
            hit_rand = float(np.mean([inside(centroid(masks[r]), boxes) for r in vis]))
            cx = np.mean([(b[0] + b[2]) / 2 for b in boxes])
            single = all((b[0] + b[2]) / 2 < 0.45 for b in boxes) or all((b[0] + b[2]) / 2 > 0.55 for b in boxes)
            sr = max([Pr[i, r, j] for r in RIGHT if r in vis] or [np.nan]); sl = max([Pr[i, r, j] for r in LEFT if r in vis] or [np.nan])
            sp = side_prior.get(fn, {"right": 1, "left": 1})
            rows.append({"finding": fn, "hit_probe": hit_p, "hit_prior": hit_q, "hit_random": hit_rand,
                         "hit_hybrid": inside(centroid(masks[rh]), boxes), "cov_hybrid": coverage(masks[rh], boxes),
                         "sr": sr, "sl": sl,
                         "cov_probe": coverage(masks[rp], boxes), "cov_prior": coverage(masks[rprior], boxes),
                         "single_side": single, "true_right": cx < 0.5, "side_score": sr - sl,
                         "prior_right": sp["right"] >= sp["left"]})
    T = pd.DataFrame(rows)
    print(f"E10 PadChest-GR: {len(T)} boxed abnormal findings mapped to the probe vocabulary "
          f"({T.finding.nunique()} finding types); most frequent unmapped labels: {sorted(unmapped.items(), key=lambda x: -x[1])[:8]}")
    print(f"\n{'finding':<24}{'n':>6}{'hit probe':>11}{'hit prior':>11}{'hit random':>12}{'overlap probe':>15}{'overlap prior':>15}")
    out = {}
    for fn, g in list(T.groupby("finding")) + [("ALL", T)]:
        if len(g) < 20 and fn != "ALL":
            continue
        out[fn] = {"n": len(g), "hit_probe": round(g.hit_probe.mean(), 4), "hit_prior": round(g.hit_prior.mean(), 4),
                   "hit_random": round(g.hit_random.mean(), 4), "overlap_probe": round((g.cov_probe >= 0.1).mean(), 4),
                   "overlap_prior": round((g.cov_prior >= 0.1).mean(), 4)}
        o = out[fn]
        print(f"{fn:<24}{o['n']:>6}{o['hit_probe']:>11.3f}{o['hit_prior']:>11.3f}{o['hit_random']:>12.3f}{o['overlap_probe']:>15.3f}{o['overlap_prior']:>15.3f}")
    (HOME / "work/e10").mkdir(parents=True, exist_ok=True)
    T.to_parquet(HOME / "work/e10/pointing_rows.parquet", index=False)        # finding names + scores only, no IDs
    extended(T, out)
    S = T[T.single_side & T.side_score.notna()]
    acc_p = float(((S.side_score > 0) == S.true_right).mean()); acc_q = float((S.prior_right == S.true_right).mean())
    au = E4.auroc(S.side_score, S.true_right)
    print(f"\nside (single-side boxed findings, n={len(S)}): region-probe accuracy {acc_p:.3f} (AUROC {au:.3f}) | "
          f"prior majority-side accuracy {acc_q:.3f} | right share {S.true_right.mean():.3f}")
    per = {f: round(E4.auroc(g.side_score, g.true_right), 3) for f, g in S.groupby("finding") if g.true_right.sum() >= 10 and (~g.true_right).sum() >= 10}
    print("  side AUROC per finding:", per)
    # McNemar-style paired comparison of hits, probe vs prior
    b = int((T.hit_probe & ~T.hit_prior).sum()); c = int((~T.hit_probe & T.hit_prior).sum())
    from math import comb
    n = b + c; pval = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
    print(f"paired hits probe vs prior: probe-only {b}, prior-only {c}, exact McNemar p={pval:.2g}")
    out["side"] = {"n": len(S), "acc_probe": round(acc_p, 4), "auroc_probe": round(au, 4), "acc_prior": round(acc_q, 4), "per_finding": per}
    out["mcnemar"] = {"probe_only": b, "prior_only": c, "p": pval}
    (HOME / "work/e10").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(HOME / "work/e10/pointing.json", "w"), indent=1)
    print("E10_DONE")


if __name__ == "__main__":
    main()
