#!/usr/bin/env python3
"""Budget B0 baseline: a VLM writes the FINDINGS section from the frontal image + indication, no retrieval.
Works for MedGemma-4B, Qwen2.5-VL-7B and Lingshu-7B through vLLM. Writes generations locally (contain text).

    python gen_b0.py --model ~/anchor/models/... --name medgemma4b --keys dev --limit 200
"""
import argparse
import base64
import io
import os
import pathlib
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image

HOME = pathlib.Path.home() / "anchor"
SYSTEM = "You are an expert thoracic radiologist."
INSTR = ("Write only the FINDINGS section of the report for this chest X-ray, in plain prose as a radiologist would. "
         "Describe pertinent positive and negative findings. Do not write an impression, recommendations or headings.")


def image_data_url(path, max_side=1024):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    lo, hi = np.percentile(a[::4, ::4], [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    img = Image.fromarray(np.ascontiguousarray((a * 255).astype(np.uint8))).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(img_url, indication):
    ind = indication if isinstance(indication, str) and indication.strip() else "not provided"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}},
                                         {"type": "text", "text": f"Indication: {ind}\n{INSTR}"}]}]


def pick_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):          # chain already chose a GPU with enough free memory
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", os.environ.get("ANCHOR_GPUS", "0,1"), "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return min((tuple(int(v) for v in l.split(",")) for l in out.strip().splitlines()), key=lambda r: r[1])[0]
    except Exception:
        return None


def make_vllm(model, gpu_util, max_len, n=1, temperature=0.0, seed=0):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=max_len,
              limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=0, max_tokens=320) if n == 1 else \
        SamplingParams(n=n, temperature=temperature, top_p=0.95, seed=seed, max_tokens=320)

    def generate(msgs):
        outs = llm.chat(msgs, sp, use_tqdm=False)
        if n == 1:
            return [(o.outputs[0].text.strip(), o.outputs[0].finish_reason, len(o.outputs[0].token_ids)) for o in outs]
        return [[(c.text.strip(), c.finish_reason, len(c.token_ids)) for c in o.outputs] for o in outs]
    return generate


def main(argv=None, generate=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, default="./models/Google/medgemma-4b-it")
    ap.add_argument("--name", default="medgemma4b")
    ap.add_argument("--keys", default="dev", help="dev | valid | test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default=str(HOME / "work/gen"))
    ap.add_argument("--n-samples", type=int, default=1, help=">1: stochastic samples for the self-consistency baseline (E12)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    mdir = HOME / "work/manifest"
    if a.keys == "dev":
        keys = pd.read_parquet(mdir / "dev_slice.parquet").key
        man = pd.read_parquet(mdir / "rex_train.parquet")
    else:
        man = pd.read_parquet(mdir / f"rex_{a.keys}.parquet")
        flags = pd.read_parquet(mdir / f"flags_{a.keys}.parquet")
        keys = flags[flags.cohort_primary].key
    man = man.set_index("key").loc[sorted(keys)]
    if a.limit:
        man = man.iloc[: a.limit]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"b0_{a.name}_{a.keys}{a.limit or ''}.parquet"
    print(f"{len(man)} studies -> {dst.name}", flush=True)
    if generate is None:
        g = pick_gpu()
        if g is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(g); print(f"using GPU {g}", flush=True)
        generate = make_vllm(a.model, a.gpu_util, a.max_len, a.n_samples, a.temperature, a.seed)
    if a.n_samples > 1:
        return sample_many(a, man, out, generate)
    rows, t0 = [], time.time()
    for s in range(0, len(man), a.batch):
        part = man.iloc[s:s + a.batch]
        msgs = [build_messages(image_data_url(p), ind) for p, ind in zip(part.frontal_path, part.indication)]
        for key, (text, fin, ntok) in zip(part.index, generate(msgs)):
            rows.append({"key": key, "generated": text, "finish_reason": fin, "n_tokens": ntok})
        print(f"  {min(s + a.batch, len(man))}/{len(man)}  {len(rows) / max(time.time() - t0, 1e-6):.2f} studies/s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    print(f"done: empty outputs={int((df.generated.str.len() == 0).sum())}, truncated={int((df.finish_reason == 'length').sum())}, "
          f"tokens p50={int(df.n_tokens.median())}, words p50={int(df.generated.str.split().str.len().median())}")
    print("GEN_B0_DONE", flush=True)


def sample_many(a, man, out, generate):
    """E12: N stochastic samples per study -> b0sc{s}_{name}_{keys}.parquet, s = 0..N-1."""
    rows, t0 = [[] for _ in range(a.n_samples)], time.time()
    for s in range(0, len(man), a.batch):
        part = man.iloc[s:s + a.batch]
        msgs = [build_messages(image_data_url(p), ind) for p, ind in zip(part.frontal_path, part.indication)]
        for key, outs in zip(part.index, generate(msgs)):
            for i, (text, fin, ntok) in enumerate(outs):
                rows[i].append({"key": key, "generated": text, "finish_reason": fin, "n_tokens": ntok})
        print(f"  {min(s + a.batch, len(man))}/{len(man)}  {len(rows[0]) / max(time.time() - t0, 1e-6):.2f} studies/s", flush=True)
    for i, r in enumerate(rows):
        pd.DataFrame(r).to_parquet(out / f"b0sc{i}_{a.name}_{a.keys}.parquet", index=False)
    print(f"done: {a.n_samples} samples x {len(rows[0])} studies (T={a.temperature}, top_p 0.95)")
    print("GEN_B0_DONE", flush=True)


if __name__ == "__main__":
    main()
