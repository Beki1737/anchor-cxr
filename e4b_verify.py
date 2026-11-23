#!/usr/bin/env python3
"""E4b: can ANCHOR ADMIT selectively? Image-grounded claim verification on a broad candidate pool (~/venvs/anchor).

E4a verdict: region retrieval is the better PROPOSER (oracle recall 0.83 vs 0.74; recall@8 +0.04 to +0.06) but
neighbour votes are a weak VERIFIER (AUROC 0.64 region, 0.71 global). E4b adds image evidence per claim:

  build    stratified subset of valid studies; candidates = VLM claims U top-M region-pool U top-M global-pool claims,
           features: source flags, ranks, global votes, region votes (max and mean over regions), label vs reference
  verify   P(yes) from VLM yes/no logprobs, two views x two model families:
           full image + question  |  region crop (CXAS bbox of the claim's region) + question
           verifiers: MedGemma-4B (same family as generator) and Qwen2.5-VL-7B (cross-family)
  analyze  AUROC per signal (VLM claims, pool claims, all); cross-fitted logistic combination (folds by patient);
           admission curve: recall of reference claims at the VLM's own precision and at 0.70 / 0.80, vs MedGemma B0

Decision (pre-registered): ANCHOR's core claim holds if some admitted set beats B0 on recall at >= B0 precision on the
held-out fold. Aggregates only; candidate files stay local.
"""
import argparse
import base64
import io
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402

OUT = HOME / "work/e4b"
PHRASE = {"R_apical": "right lung apex", "R_upper": "right upper lung zone", "R_mid": "right mid lung zone",
          "R_base": "right lung base", "L_apical": "left lung apex", "L_upper": "left upper lung zone",
          "L_mid": "left mid lung zone", "L_base": "left lung base", "R_upper_lobe": "right upper lobe",
          "R_middle_lobe": "right middle lobe", "R_lower_lobe": "right lower lobe", "L_upper_lobe": "left upper lobe",
          "L_lower_lobe": "left lower lobe", "R_lung": "right lung", "L_lung": "left lung", "heart": "heart",
          "upper_mediastinum": "upper mediastinum", "cardiomediastinum": "cardiomediastinal silhouette",
          "trachea": "trachea", "aortic_arch": "aortic arch", "R_hemidiaphragm": "right hemidiaphragm",
          "L_hemidiaphragm": "left hemidiaphragm", "spine": "thoracic spine", "clavicles": "clavicles"}
DEFAULT_REGION = {"cardiomegaly": ["heart"], "mediastinal widening": ["cardiomediastinum"], "scoliosis": ["spine"],
                  "degenerative changes": ["spine"], "aortic calcification": ["aortic_arch"], "hiatal hernia": ["heart"]}
MODELS = {"medgemma": "./models/Google/medgemma-4b-it",
          "qwenvl": str(HOME / "models/Qwen/Qwen2.5-VL-7B-Instruct")}


def claim_regions(c, regs):
    r = sorted(x for x in regs if x in PHRASE)
    if r:
        return r
    if c[0] in DEFAULT_REGION:
        return DEFAULT_REGION[c[0]]
    return {"right": ["R_lung"], "left": ["L_lung"], "bilateral": ["R_lung", "L_lung"]}.get(c[1], [])


def cmd_build(a):
    t0 = time.time()
    sp_ = getattr(a, "split", "valid")
    flags = pd.read_parquet(HOME / f"work/manifest/flags_{sp_}.parquet")
    man = pd.read_parquet(HOME / f"work/manifest/rex_{sp_}.parquet").set_index("key")
    from strata import study_strata
    st = study_strata(HOME / f"work/corpus_v2/{sp_}")
    keys = sorted(set(flags[flags.cohort_primary].key) & set(st.index))
    rng = np.random.default_rng(0)
    sub = sorted(rng.choice(keys, size=min(a.n, len(keys)), replace=False))     # proportional (random) subset
    ref = E4.load_claims(HOME / f"work/corpus_v2/{sp_}")
    corp = E4.load_claims(HOME / "work/corpus_v2/train")
    vlm = E4.load_claims(HOME / "work/corpus_gen" / a.vlm)
    glob = pd.read_parquet(HOME / f"work/retrieval/global_cls_{sp_}.parquet").set_index("key").nbr
    trf = pd.read_parquet(HOME / "work/manifest/flags_train.parquet")
    ckeys = sorted(set(trf[trf.cohort_primary].key) & set(corp))
    qk, Qe, Qa = E4.load_region_emb(sp_, sub)
    ck, Ce, Ca = E4.load_region_emb("train", ckeys)
    E4.COMPAT = {(q, c) for q in E4.RN for c in E4.RN if E4.compat_region(q, c)}
    rnn = E4.region_knn(qk, Qe, Qa, ck, Ce, Ca, a.k)
    qpos = {k: i for i, k in enumerate(qk)}
    rows, nref = [], 0
    for q in sub:
        if q not in qpos or q not in glob.index:
            continue
        R = set(ref.get(q, {})); nref += len(R)
        gs, greg = {}, {}
        nb = list(glob[q][: a.k])
        for n in nb:
            for c, regs in corp.get(n, {}).items():
                gs[c] = gs.get(c, 0) + 1 / len(nb); greg.setdefault(c, set()).update(regs)
        rmax, rsum, rreg, nreg = {}, {}, {}, 0
        for r, nbrs in rnn.get(qpos[q], []):
            nreg += 1; cnt = {}
            for n in nbrs:
                for c, regs in corp.get(n, {}).items():
                    if any((r, x) in E4.COMPAT for x in regs):
                        cnt[c] = cnt.get(c, 0) + 1; rreg.setdefault(c, set()).update(regs)
            for c, v in cnt.items():
                rmax[c] = max(rmax.get(c, 0), v / len(nbrs)); rsum[c] = rsum.get(c, 0) + v / len(nbrs)
        V = vlm.get(q, {})
        rorder = sorted(rmax, key=lambda c: -rmax[c]); gorder = sorted(gs, key=lambda c: -gs[c])
        cand = set(V) | set(rorder[: a.m]) | set(gorder[: a.m])
        for c in cand:
            regs = V.get(c) or rreg.get(c) or greg.get(c) or set()
            rows.append({"key": q, "patient": man.at[q, "patient"], "finding": c[0], "lat": c[1],
                         "regions": ",".join(claim_regions(c, regs)), "from_vlm": c in V,
                         "region_rank": rorder.index(c) + 1 if c in rmax else 99, "global_rank": gorder.index(c) + 1 if c in gs else 99,
                         "votes_global": gs.get(c, 0.0), "votes_region_max": rmax.get(c, 0.0),
                         "votes_region_mean": rsum.get(c, 0.0) / max(nreg, 1), "label": E4.matches(c, R),
                         "stratum": st.at[q, "stratum"]})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "candidates.parquet", index=False)
    vl = df[df.from_vlm]
    json.dump({"n_studies": len(sub), "n_ref_claims": nref}, open(OUT / "build_meta.json", "w"))
    print(f"E4b build: {len(sub)} studies, {len(df)} candidates ({len(df) / len(sub):.1f}/study), ref claims {nref}, "
          f"candidate precision {df.label.mean():.3f}, VLM claims {len(vl)} precision {vl.label.mean():.3f}, "
          f"oracle recall of candidate set {df.groupby('key').label.sum().sum() / max(nref, 1):.3f} ({time.time() - t0:.0f}s)")
    print("E4B_BUILD_DONE", flush=True)


def load_square(path):
    from PIL import Image
    with Image.open(path) as im:
        x = np.asarray(im)
    if x.ndim == 3:
        x = x[..., 0]
    x = x.astype(np.float32)
    lo, hi = np.percentile(x[::4, ::4], [0.5, 99.5])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = x.shape; s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
    return x[y0:y0 + s, x0:x0 + s]                    # same centre square as the region cache (bbox frame)


def to_url(a, max_side=768):
    from PIL import Image
    img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def crop(sq, boxes, pad=0.12, min_frac=0.25):
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, min_frac) / 2 * (1 + pad)
    s = sq.shape[0]
    X0, Y0 = int(max(0, (cx - half) * s)), int(max(0, (cy - half) * s))
    X1, Y1 = int(min(s, (cx + half) * s)), int(min(s, (cy + half) * s))
    return sq[Y0:Y1, X0:X1] if (X1 - X0) > 16 and (Y1 - Y0) > 16 else sq


def question(f, regs, view):
    where = " and ".join(PHRASE[r] for r in regs) if regs else "chest"
    name = "a support device (tube, line, catheter or pacemaker)" if f == "support device" else f
    pre = f"This image is a crop of the {where} from a chest X-ray. " if view == "crop" else "This is a chest X-ray. "
    return pre + f"Is there {name} in the {where}? Answer with one word: Yes or No."


def yes_prob(lp):
    py = pn = 0.0
    for v in lp.values():
        t = (v.decoded_token or "").strip().lower()
        if t == "yes":
            py += float(np.exp(v.logprob))
        elif t == "no":
            pn += float(np.exp(v.logprob))
    return py / (py + pn) if py + pn > 0 else 0.5


def cmd_verify(a):
    from vllm import LLM, SamplingParams
    import gen_b0
    df = pd.read_parquet(OUT / "candidates.parquet")
    man = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key")
    boxes = {}
    for sp in sorted((HOME / "work/regions/valid").glob("shard_*.npz")):
        z = np.load(sp)
        for k, bb, ar in zip(z["keys"], z["bbox"], z["area"]):
            boxes[str(k)] = (bb.astype(np.float32), ar.astype(np.float32))
    g = gen_b0.pick_gpu()
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(g); print(f"using GPU {g}", flush=True)
    llm = LLM(model=MODELS[a.model], dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=4096,
              limit_mm_per_prompt={"image": 1}, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    scores = {"full": np.full(len(df), np.nan), "crop": np.full(len(df), np.nan)}
    t0 = time.time()
    for key, grp in df.groupby("key", sort=False):
        sq = load_square(man.at[key, "frontal_path"])
        full_url = to_url(sq)
        bb, ar = boxes.get(key, (None, None))
        msgs, slots = [], []
        for i, r in grp.iterrows():
            regs = [x for x in r.regions.split(",") if x]
            for view in ["full", "crop"]:
                if view == "crop":
                    sel = [bb[E4.RN.index(x)] for x in regs if bb is not None and ar[E4.RN.index(x)] > 0]
                    url = to_url(crop(sq, sel)) if sel else full_url
                else:
                    url = full_url
                msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                           {"type": "text", "text": question(r.finding, regs, view)}]}])
                slots.append((i, view))
        outs = llm.chat(msgs, sp, use_tqdm=False)
        for (i, view), o in zip(slots, outs):
            scores[view][df.index.get_loc(i)] = yes_prob(o.outputs[0].logprobs[0])
    df[f"{a.model}_full"] = scores["full"]; df[f"{a.model}_crop"] = scores["crop"]
    df[["key", f"{a.model}_full", f"{a.model}_crop"]].to_parquet(OUT / f"verify_{a.model}.parquet", index=False)
    print(f"verify {a.model}: {len(df)} claims x 2 views in {time.time() - t0:.0f}s ({2 * len(df) / (time.time() - t0):.1f} calls/s)")
    print("E4B_VERIFY_DONE", flush=True)


def logreg_fit(X, y, l2=1.0, iters=500, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y)); b -= lr * g.mean()
    return w, b


def covered(adm, ref):
    """Reference claims covered by an admitted candidate set (per study, no double counting)."""
    tot = 0
    for key, g in adm.groupby("key"):
        A = list(zip(g.finding, g.lat))
        tot += sum(E4.matches(r, A) for r in ref.get(key, {}))
    return tot


def per_study(adm, ref):
    out = {}
    for key, g in adm.groupby("key"):
        A = list(zip(g.finding, g.lat))
        out[key] = (sum(E4.matches(r, A) for r in ref.get(key, {})), len(g), int(g.label.sum()))
    return out


def boot_vs_b0(A, vlm, df, ref, n_boot=1000):
    """Recall and precision differences (admitted set minus MedGemma B0), resampling patients."""
    keys = df.drop_duplicates("key")[["key", "patient"]]
    sa, sb = per_study(A, ref), per_study(vlm, ref)
    nref = np.array([len(ref.get(k, {})) for k in keys.key])
    ca = np.array([sa.get(k, (0, 0, 0)) for k in keys.key]); cb = np.array([sb.get(k, (0, 0, 0)) for k in keys.key])
    pats = keys.patient.astype(str).to_numpy(); up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0); dr, dp = [], []
    for _ in range(n_boot):
        sel = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        R = nref[sel].sum()
        dr.append(ca[sel, 0].sum() / R - cb[sel, 0].sum() / R)
        dp.append(ca[sel, 2].sum() / max(ca[sel, 1].sum(), 1) - cb[sel, 2].sum() / max(cb[sel, 1].sum(), 1))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"recall_diff": round(float(np.mean(dr)), 4), "recall_ci": q(dr), "precision_diff": round(float(np.mean(dp)), 4), "precision_ci": q(dp)}


def threshold_for(cal, target, min_n=20):
    c = cal.sort_values("combined", ascending=False)
    prec = c.label.cumsum().to_numpy() / np.arange(1, len(c) + 1)
    ok = np.where((prec >= target) & (np.arange(1, len(c) + 1) >= min_n))[0]
    return float(c.combined.iloc[ok.max()]) if len(ok) else np.inf


def cmd_analyze(a):
    import hashlib
    df = pd.read_parquet(OUT / "candidates.parquet")
    meta = json.load(open(OUT / "build_meta.json"))
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    for p in sorted(OUT.glob("verify_*.parquet")):            # zero-shot VLM verifiers and the trained region probe
        v = pd.read_parquet(p)
        for c in v.columns:
            if c != "key":
                df[c] = v[c].to_numpy()
    df["inv_region_rank"] = 1 / df.region_rank; df["inv_global_rank"] = 1 / df.global_rank
    votes = ["votes_global", "votes_region_max", "votes_region_mean"]
    zs = [c for c in ["medgemma_full", "medgemma_crop", "qwenvl_full", "qwenvl_crop"] if c in df]
    probe = [c for c in ["probe_region", "probe_global"] if c in df]
    ver = zs + probe
    df[votes] = df[votes].fillna(0.0); df[ver] = df[ver].fillna(0.5)
    print(f"E4b analyze: {meta['n_studies']} studies, {len(df)} candidates, ref claims {meta['n_ref_claims']}")
    print(f"{'signal':<20}{'AUROC all':>11}{'VLM claims':>12}{'pool-only':>11}")
    res = {"auroc": {}}
    for s_ in votes + ver:
        x = df[s_]
        r = [E4.auroc(x, df.label), E4.auroc(x[df.from_vlm], df.label[df.from_vlm]), E4.auroc(x[~df.from_vlm], df.label[~df.from_vlm])]
        res["auroc"][s_] = [round(v, 4) for v in r]
        print(f"{s_:<20}{r[0]:>11.3f}{r[1]:>12.3f}{r[2]:>11.3f}")
    fold = np.array([int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 3 for p in df.patient])   # 3 folds by patient
    y = df.label.astype(float).to_numpy()
    base = ["from_vlm", "inv_region_rank", "inv_global_rank"] + votes
    sets = {"votes only": base}
    if zs:
        sets["votes + zero-shot VLM verifiers"] = base + zs
    if probe:
        sets["votes + region probe"] = base + probe
    if zs and probe:
        sets["votes + VLM verifiers + probe"] = base + zs + probe
    vlm = df[df.from_vlm]
    b0_rec = covered(vlm, ref) / meta["n_ref_claims"]; b0_prec = vlm.label.mean()
    res["b0"] = {"recall": round(b0_rec, 4), "precision": round(b0_prec, 4), "claims_per_study": round(len(vlm) / meta["n_studies"], 2)}
    print(f"\nMedGemma B0 claims on this subset: recall {b0_rec:.3f} precision {b0_prec:.3f} ({len(vlm) / meta['n_studies']:.2f} claims/study)")
    targets = [round(float(b0_prec), 3), 0.70, 0.80]
    for name, feats in sets.items():
        X = df[feats].astype(float).to_numpy()
        oof = np.zeros(len(df)); adm = {t: [] for t in targets}
        for f in range(3):                      # rotate: train on fold f, calibrate threshold on f+1, test on f+2
            tr, ca, te = fold == f, fold == (f + 1) % 3, fold == (f + 2) % 3
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            w, b = logreg_fit((X[tr] - mu) / sd, y[tr])
            sc = 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
            oof[te] = sc[te]
            cal = df[ca].assign(combined=sc[ca]); tst = df[te].assign(combined=sc[te])
            for t in targets:
                adm[t].append(tst[tst.combined >= threshold_for(cal, t)])
        au = E4.auroc(oof, y)
        res[name] = {"auroc_heldout": round(au, 4)}
        print(f"\n[{name}] held-out AUROC {au:.3f}")
        for ti, t in enumerate(targets):
            A = pd.concat(adm[t])
            if ti == 0:
                res[name]["ci_vs_b0"] = boot_vs_b0(A, vlm, df, ref)
                print(f"  vs B0 at matched precision (patient bootstrap 95% CI): {res[name]['ci_vs_b0']}")
            rec = covered(A, ref) / meta["n_ref_claims"]; prec = A.label.mean() if len(A) else float("nan")
            res[name][f"admit@{t}"] = {"recall": round(rec, 4), "precision": round(float(prec), 4),
                                      "claims_per_study": round(len(A) / meta["n_studies"], 2)}
            print(f"  target precision {t:.3f}: held-out precision {prec:.3f}, recall {rec:.3f} "
                  f"({rec / max(b0_rec, 1e-9):.2f}x B0), {len(A) / meta['n_studies']:.2f} claims/study")
    json.dump(res, open(OUT / "e4b_results.json", "w"), indent=1)
    print("E4B_ANALYZE_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "verify", "analyze"])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--vlm", default="b0_medgemma4b_valid")
    ap.add_argument("--model", choices=list(MODELS), default="medgemma")
    ap.add_argument("--gpu-util", type=float, default=0.35)
    a = ap.parse_args(argv)
    {"build": cmd_build, "verify": cmd_verify, "analyze": cmd_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
