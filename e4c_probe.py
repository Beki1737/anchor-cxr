#!/usr/bin/env python3
"""E4c: trained, anatomy-conditioned evidence probe (run in ~/venvs/anchor, one GPU, minutes).

E4b: zero-shot yes/no VLM verifiers are near chance on MedGemma's own claims (AUROC 0.54 to 0.55) and region crops
did not help them. Instead, learn image evidence directly on the cached RAD-DINO REGION embeddings:
    P(finding f present in region r | region embedding e_r, region id r)        (region head, multi-label)
    P(finding f present in study | CLS)                                          (global head, claims without a region)
Weak labels come from the TRAIN corpus claims only (E1 v2, hierarchical region compatibility); valid/test untouched.
  train   fit both heads on train studies (primary cohort), early stop on a 5% patient-held-out train slice
  eval    region- and study-level AUROC per finding on VALID (labels from valid reference claims)
  score   probe score for every E4b candidate -> ~/anchor/work/e4b/verify_probe.parquet (picked up by e4b analyze)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e4b_verify as E4B  # noqa: E402

OUT = HOME / "work/e4c"


def load_split(split, keys):
    ks, em, ar, cl = [], [], [], []
    for sp in sorted((HOME / "work/regions" / split).glob("shard_*.npz")):
        z = np.load(sp)
        ks.append(z["keys"]); em.append(z["emb"]); ar.append(z["area"]); cl.append(z["cls"])
    K = np.concatenate(ks)
    idx = pd.Series(np.arange(len(K)), index=K)
    idx = idx[~idx.index.duplicated()].reindex(keys).dropna().astype(int)
    s = idx.to_numpy()
    return idx.index.to_numpy(), np.concatenate(em)[s], np.concatenate(ar)[s].astype(np.float32), np.concatenate(cl)[s]


def labels(keys, area, claims, vocab):
    """Y_reg [N,24,F] (uint8), Y_glob [N,F]: positive if a positive claim of f is anatomically compatible with region r."""
    F = {f: i for i, f in enumerate(vocab)}
    Yr = np.zeros((len(keys), len(E4.RN), len(vocab)), np.uint8); Yg = np.zeros((len(keys), len(vocab)), np.uint8)
    for n, k in enumerate(keys):
        for c, regs in claims.get(k, {}).items():
            fi = F.get(c[0])
            if fi is None:
                continue
            Yg[n, fi] = 1
            rr = E4B.claim_regions(c, regs)
            for ri, r in enumerate(E4.RN):
                if area[n, ri] > 0 and any((r, x) in E4.COMPAT for x in rr):
                    Yr[n, ri, fi] = 1
    return Yr, Yg


def auc_cols(P, Y, min_pos=20):
    out = {}
    for j in range(Y.shape[1]):
        if Y[:, j].sum() >= min_pos and (1 - Y[:, j]).sum() >= min_pos:
            out[j] = E4.auroc(P[:, j], Y[:, j])
    return out


def build_model(torch, F):
    nn = torch.nn
    reg = nn.Sequential(nn.LayerNorm(768 + 24), nn.Linear(768 + 24, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, F))
    glo = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, F))
    return reg, glo


def reg_input(torch, E, ri):
    oh = torch.nn.functional.one_hot(ri, 24).to(E.dtype)
    return torch.cat([E, oh], 1)


def cmd_train(a):
    import torch
    t0 = time.time()
    E4.COMPAT = {(q, c) for q in E4.RN for c in E4.RN if E4.compat_region(q, c)}
    corp = E4.load_claims(HOME / "work/corpus_v2/train")
    cnt = pd.Series([c[0] for v in corp.values() for c in v]).value_counts()
    vocab = [f for f in cnt.index if cnt[f] >= a.min_count][: a.vocab]
    trf = pd.read_parquet(HOME / "work/manifest/flags_train.parquet")
    keys = sorted(set(trf[trf.cohort_primary].key) & set(corp))
    keys, Em, Ar, Cl = load_split("train", keys)
    Yr, Yg = labels(keys, Ar, corp, vocab)
    pat = pd.read_parquet(HOME / "work/manifest/rex_train.parquet").set_index("key").loc[keys, "patient"].astype(str).to_numpy()
    import hashlib
    hold = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 20 == 0 for p in pat])
    print(f"vocab {len(vocab)} findings | train studies {len(keys)} (early-stop slice {hold.sum()}) | "
          f"region positives {int(Yr.sum())} | {time.time() - t0:.0f}s", flush=True)
    dev = "cuda"
    Eg = torch.from_numpy(Em).to(dev); Cg = torch.from_numpy(Cl).to(dev)
    Yrg = torch.from_numpy(Yr).to(dev); Ygg = torch.from_numpy(Yg).to(dev)
    Arg = torch.from_numpy(Ar > 0).to(dev)
    reg, glo = build_model(torch, len(vocab)); reg.to(dev); glo.to(dev)
    opt = torch.optim.AdamW(list(reg.parameters()) + list(glo.parameters()), lr=a.lr, weight_decay=1e-4)
    tr_idx = np.where(~hold)[0]; ho_idx = torch.from_numpy(np.where(hold)[0]).to(dev)
    pairs = torch.nonzero(Arg[torch.from_numpy(tr_idx).to(dev)])                 # (row in tr_idx, region)
    pairs[:, 0] = torch.from_numpy(tr_idx).to(dev)[pairs[:, 0]]
    bce = torch.nn.BCEWithLogitsLoss()
    best, best_state = -1, None
    for ep in range(a.epochs):
        reg.train(); glo.train()
        perm = pairs[torch.randperm(len(pairs), device=dev)]
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            x = reg_input(torch, Eg[b[:, 0], b[:, 1]].float(), b[:, 1])
            loss = bce(reg(x), Yrg[b[:, 0], b[:, 1]].float())
            gb = b[:, 0][: a.batch // 8]
            loss = loss + bce(glo(Cg[gb].float()), Ygg[gb].float())
            opt.zero_grad(); loss.backward(); opt.step()
        reg.eval(); glo.eval()
        with torch.no_grad():
            P = torch.sigmoid(glo(Cg[ho_idx].float())).cpu().numpy()
        au = auc_cols(P, Yg[hold], min_pos=5)
        m = float(np.nanmean(list(au.values()))) if au else float("nan")
        print(f"epoch {ep + 1}: early-stop slice study-level macro AUROC {m:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if best_state is None or (m == m and m > best):
            best = m; best_state = (dict((k, v.clone()) for k, v in reg.state_dict().items()),
                                    dict((k, v.clone()) for k, v in glo.state_dict().items()))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"reg": best_state[0], "glo": best_state[1], "vocab": vocab}, OUT / "probe.pt")
    print(f"saved probe (best slice AUROC {best:.3f})\nE4C_TRAIN_DONE", flush=True)


def load_probe(torch):
    ck = torch.load(OUT / "probe.pt", map_location="cuda")
    reg, glo = build_model(torch, len(ck["vocab"]))
    reg.load_state_dict(ck["reg"]); glo.load_state_dict(ck["glo"])
    return reg.cuda().eval(), glo.cuda().eval(), ck["vocab"]


def predict(torch, reg, glo, Em, Cl):
    with torch.no_grad():
        E = torch.from_numpy(Em).cuda().float(); N = E.shape[0]
        Pr = np.zeros((N, 24, reg[-1].out_features), np.float32)
        for ri in range(24):
            x = reg_input(torch, E[:, ri], torch.full((N,), ri, device="cuda"))
            Pr[:, ri] = torch.sigmoid(reg(x)).cpu().numpy()
        Pg = torch.sigmoid(glo(torch.from_numpy(Cl).cuda().float())).cpu().numpy()
    return Pr, Pg


def cmd_eval(a):
    import torch
    E4.COMPAT = {(q, c) for q in E4.RN for c in E4.RN if E4.compat_region(q, c)}
    reg, glo, vocab = load_probe(torch)
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    vf = pd.read_parquet(HOME / "work/manifest/flags_valid.parquet")
    keys, Em, Ar, Cl = load_split("valid", sorted(set(vf[vf.cohort_primary].key)))
    Yr, Yg = labels(keys, Ar, ref, vocab)
    Pr, Pg = predict(torch, reg, glo, Em, Cl)
    m = Ar > 0
    reg_auc = auc_cols(Pr[m], Yr[m]); glo_auc = auc_cols(Pg, Yg)
    pooled = auc_cols(np.where(m[..., None], Pr, 0).max(1), Yg)
    print(f"VALID ({len(keys)} studies): macro AUROC region-level {np.nanmean(list(reg_auc.values())):.3f} "
          f"| study-level global head {np.nanmean(list(glo_auc.values())):.3f} | study-level max-over-regions {np.nanmean(list(pooled.values())):.3f}")
    rows = [(vocab[j], int(Yg[:, j].sum()), round(glo_auc.get(j, np.nan), 3), round(reg_auc.get(j, np.nan), 3)) for j in range(len(vocab))]
    print("per finding (name, valid positives, study AUROC, region AUROC):")
    for r in sorted(rows, key=lambda x: -x[1])[:20]:
        print("  ", r)
    json.dump({"region_macro": float(np.nanmean(list(reg_auc.values()))), "global_macro": float(np.nanmean(list(glo_auc.values()))),
               "per_finding": rows}, open(OUT / "eval_valid.json", "w"), indent=1)
    print("E4C_EVAL_DONE")


def cmd_score(a):
    import torch
    reg, glo, vocab = load_probe(torch)
    F = {f: i for i, f in enumerate(vocab)}
    d = HOME / "work" / a.dir
    cand = pd.read_parquet(d / "candidates.parquet")
    keys, Em, Ar, Cl = load_split(a.split, sorted(set(cand.key)))
    Pr, Pg = predict(torch, reg, glo, Em, Cl)
    pos = {k: i for i, k in enumerate(keys)}
    pr, pg = [], []
    for k, f, regs in zip(cand.key, cand.finding, cand.regions):
        i, j = pos.get(k), F.get(f)
        if i is None or j is None:
            pr.append(np.nan); pg.append(np.nan); continue
        ri = [E4.RN.index(x) for x in regs.split(",") if x and Ar[i, E4.RN.index(x)] > 0]
        pr.append(float(Pr[i, ri, j].max()) if ri else float(Pg[i, j])); pg.append(float(Pg[i, j]))
    out = pd.DataFrame({"key": cand.key, "probe_region": pr, "probe_global": pg})
    out.to_parquet(d / "verify_probe.parquet", index=False)
    print(f"probe scores for {len(out)} candidates (finding outside probe vocabulary: {int(out.probe_region.isna().sum())})")
    print("E4C_SCORE_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval", "score"])
    ap.add_argument("--vocab", type=int, default=30)
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dir", default="e4b", help="score: candidate folder under ~/anchor/work")
    ap.add_argument("--split", default="valid")
    a = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval, "score": cmd_score}[a.stage](a)


if __name__ == "__main__":
    main()
