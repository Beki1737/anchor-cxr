#!/usr/bin/env python3
"""E3: global image-to-report retrieval baselines (run in ~/venvs/anchor).

  retrieve  RAD-DINO CLS cosine kNN: valid primary-cohort queries -> train primary-cohort corpus (patient-disjoint
            by the official split). Saves top-50 neighbours. Prints retrieval sanity aggregates.
  ret1      retrieval-only baseline: copy the top-1 neighbour's FINDINGS (CXR-RePaiR style)       -> gen/ret1_cls_valid.parquet
  gen       retrieval-augmented MedGemma-4B: image + indication + k distinct neighbour findings    -> gen/e3rag_medgemma4b_k3_valid.parquet

Neighbour texts are deduplicated (23 to 28% of findings are exact boilerplate duplicates), so k slots hold k distinct texts.
All outputs local (they contain report text). Only aggregates print.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def load_cls(split, keys=None):
    ks, vs, ok = [], [], []
    for sp in sorted((HOME / "work/regions" / split).glob("shard_*.npz")):
        z = np.load(sp)
        ks.append(z["keys"]); vs.append(z["cls"].astype(np.float32)); ok.append(z["ok"].astype(bool))
    K, V, O = np.concatenate(ks), np.concatenate(vs), np.concatenate(ok)
    df = pd.DataFrame({"key": K, "i": np.arange(len(K)), "ok": O}).drop_duplicates("key").set_index("key")
    if keys is not None:
        df = df.loc[[k for k in keys if k in df.index]]
    V = V[df.i.to_numpy()]
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    return df.index.to_numpy(), V, df.ok.to_numpy()


def primary(split):
    f = pd.read_parquet(HOME / f"work/manifest/flags_{split}.parquet")
    return sorted(f[f.cohort_primary].key)


def cmd_retrieve(a):
    import torch
    man_tr = pd.read_parquet(HOME / "work/manifest/rex_train.parquet").set_index("key")
    tr_keys = [k for k in primary("train") if isinstance(man_tr.at[k, "findings"], str) and man_tr.at[k, "findings"].strip()]
    qk, Q, qok = load_cls(a.split, primary(a.split))
    ck, Cm, cok = load_cls("train", tr_keys)
    ck, Cm = ck[cok], Cm[cok]                      # corpus: readable images with non-empty findings
    print(f"queries {len(qk)} (unreadable {int((~qok).sum())}) | corpus {len(ck)}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = torch.from_numpy(Cm).to(dev, torch.float16 if dev == "cuda" else torch.float32)
    nb, sims = [], []
    t0 = time.time()
    for s in range(0, len(qk), 1024):
        q = torch.from_numpy(Q[s:s + 1024]).to(dev, C.dtype)
        v, i = (q @ C.T).float().topk(a.topn, dim=1)
        nb.append(i.cpu().numpy()); sims.append(v.cpu().numpy())
    nb, sims = np.concatenate(nb), np.concatenate(sims)
    out = HOME / "work/retrieval"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"key": qk, "nbr": [list(ck[r]) for r in nb], "sim": [list(np.round(x, 4)) for x in sims]})
    df.to_parquet(out / f"global_cls_{a.split}.parquet", index=False)
    print(f"retrieval done in {time.time() - t0:.0f}s | top-1 cos p50={np.median(sims[:, 0]):.3f} p05={np.percentile(sims[:, 0], 5):.3f}")
    sanity(df, a.split)
    print("E3_RETRIEVE_DONE", flush=True)


def sanity(df, split):
    """Is retrieval clinically meaningful? Compare top-1 neighbour vs a random corpus study on E1 labels."""
    try:
        from strata import study_strata, KEY
    except Exception as e:
        print("sanity skipped:", e); return
    st_q = study_strata(HOME / f"work/corpus/{split}")
    st_c = study_strata(HOME / "work/corpus/train")
    rng = np.random.default_rng(0)
    q = df[df.key.isin(st_q.index)]
    top1 = q.nbr.map(lambda x: x[0])
    allc = np.array([k for k in st_c.index])
    rnd = rng.choice(allc, len(q))
    qa = st_q.loc[q.key, "abn"].to_numpy()
    t1 = st_c.reindex(top1).abn.to_numpy()
    rr = st_c.reindex(rnd).abn.to_numpy()
    m = ~pd.isna(t1)
    base_abn = float(np.nanmean(st_c.abn))
    res = {"n": int(m.sum()), "agree_abnormal_top1": round(float((qa[m] == t1[m].astype(bool)).mean()), 4),
           "agree_abnormal_random": round(float((qa == rr.astype(bool)).mean()), 4),
           "corpus_abnormal_rate": round(base_abn, 4),
           "P(top1 abnormal | query abnormal)": round(float(t1[m][qa[m]].astype(bool).mean()), 4),
           "P(top1 abnormal | query normal)": round(float(t1[m][~qa[m]].astype(bool).mean()), 4)}
    print("RETRIEVAL_SANITY", json.dumps(res))


def neighbour_texts(row, findings, k):
    seen, out = set(), []
    for nk in row:
        t = findings.get(nk, "")
        if not isinstance(t, str) or not t.strip():
            continue
        norm = " ".join(t.lower().split())
        if norm in seen:
            continue
        seen.add(norm); out.append(t.strip())
        if len(out) == k:
            break
    return out


def cmd_ret1(a):
    r = pd.read_parquet(HOME / f"work/retrieval/global_cls_{a.split}.parquet")
    f = pd.read_parquet(HOME / "work/manifest/rex_train.parquet").set_index("key").findings.to_dict()
    gen = [neighbour_texts(x, f, 1)[0] if neighbour_texts(x, f, 1) else "" for x in r.nbr]
    out = HOME / "work/gen" / f"ret1_cls_{a.split}.parquet"
    pd.DataFrame({"key": r.key, "generated": gen}).to_parquet(out, index=False)
    print(f"ret1 baseline -> {out.name} ({len(gen)} studies, empty={sum(not g for g in gen)})")


RAG_NOTE = ("Findings sections from {k} radiographs of OTHER patients that look similar to this image (retrieved "
            "automatically by image similarity; they may not apply to this patient):\n{refs}\n")


def cmd_gen(a):
    import gen_b0
    r = pd.read_parquet(HOME / f"work/retrieval/global_cls_{a.split}.parquet").set_index("key")
    man = pd.read_parquet(HOME / f"work/manifest/rex_{a.split}.parquet").set_index("key").loc[r.index]
    if a.limit:
        man = man.iloc[: a.limit]
    f = pd.read_parquet(HOME / "work/manifest/rex_train.parquet").set_index("key").findings.to_dict()
    name = f"e3rag_medgemma4b_k{a.k}_{a.split}{a.limit or ''}"
    dst = HOME / "work/gen" / f"{name}.parquet"
    g = gen_b0.pick_gpu()
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(g); print(f"using GPU {g}", flush=True)
    generate = gen_b0.make_vllm(a.model, a.gpu_util, a.max_len)
    rows, t0 = [], time.time()
    for s in range(0, len(man), a.batch):
        part = man.iloc[s:s + a.batch]
        msgs = []
        for key, p, ind in zip(part.index, part.frontal_path, part.indication):
            refs = neighbour_texts(r.at[key, "nbr"], f, a.k)
            m = gen_b0.build_messages(gen_b0.image_data_url(p), ind)
            txt = m[1]["content"][1]["text"]
            block = RAG_NOTE.format(k=len(refs), refs="\n".join(f"[{i + 1}] {t}" for i, t in enumerate(refs)))
            m[1]["content"][1]["text"] = txt.replace(gen_b0.INSTR, block + gen_b0.INSTR +
                                                     " Report only what is visible in THIS image.")
            msgs.append(m)
        for key, (text, fin, ntok) in zip(part.index, generate(msgs)):
            rows.append({"key": key, "generated": text, "finish_reason": fin, "n_tokens": ntok})
        print(f"  {min(s + a.batch, len(man))}/{len(man)}  {len(rows) / max(time.time() - t0, 1e-6):.2f} studies/s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    print(f"done -> {dst.name}: truncated={int((df.finish_reason == 'length').sum())}, "
          f"words p50={int(df.generated.str.split().str.len().median())}")
    print("E3_GEN_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieve", "ret1", "gen"])
    ap.add_argument("--split", default="valid")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--model", default="./models/Google/medgemma-4b-it")
    a = ap.parse_args(argv)
    {"retrieve": cmd_retrieve, "ret1": cmd_ret1, "gen": cmd_gen}[a.stage](a)


if __name__ == "__main__":
    main()
