#!/usr/bin/env python3
"""ANCHOR E0: anatomy region cache.

For each eligible frontal image: CXAS (torchxrayvision UNetResNet50, 159 classes) -> 24 ANCHOR regions;
RAD-DINO patch tokens pooled under each region mask -> one 768-d vector per region, plus CLS and mean patch.
Writes resumable npz shards (local only, never share) + aggregate QA (safe to share) + optional overlays (local only).

    python region_cache.py --split test --limit 2000 --overlays 24          # pilot
    python region_cache.py --split train                                    # full split
"""
import argparse
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HOME = pathlib.Path.home() / "anchor"
REGIONS = {  # ANCHOR region -> CXAS target names (union)
    "R_apical": ["right apical zone lung"], "R_upper": ["right upper zone lung"],
    "R_mid": ["right mid zone lung"], "R_base": ["right lung base"],
    "L_apical": ["left apical zone lung"], "L_upper": ["left upper zone lung"],
    "L_mid": ["left mid zone lung"], "L_base": ["left lung base"],
    "R_upper_lobe": ["lung upper lobe right"], "R_middle_lobe": ["lung middle lobe right"],
    "R_lower_lobe": ["lung lower lobe right"], "L_upper_lobe": ["lung upper lobe left"],
    "L_lower_lobe": ["lung lower lobe left"],
    "R_lung": ["right lung"], "L_lung": ["left lung"],
    "heart": ["heart"], "upper_mediastinum": ["upper mediastinum"], "cardiomediastinum": ["cardiomediastinum"],
    "trachea": ["trachea"], "aortic_arch": ["aortic arch"],
    "R_hemidiaphragm": ["right hemidiaphragm"], "L_hemidiaphragm": ["left hemidiaphragm"],
    "spine": ["thoracic spine"], "clavicles": ["clavicles"],
}
RNAMES = list(REGIONS)
ZONES = ["R_apical", "R_upper", "R_mid", "R_base", "L_apical", "L_upper", "L_mid", "L_base"]
OVERLAY = {"R_apical": (230, 25, 75), "R_upper": (245, 130, 48), "R_mid": (255, 225, 25), "R_base": (60, 180, 75),
           "L_apical": (0, 130, 200), "L_upper": (145, 30, 180), "L_mid": (70, 240, 240), "L_base": (240, 50, 230),
           "heart": (255, 255, 255)}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def pick_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "-i", os.environ.get("ANCHOR_GPUS", "0,1"), "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
        idx, mem, util = min(rows, key=lambda r: (r[1], r[2]))
        return str(idx), mem, util
    except Exception:
        return None, None, None


def load_gray01(path):
    """Read any PNG (16-bit or 8-bit), robust min-max to [0, 1], centre-crop to a square."""
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    lo, hi = np.percentile(a[::4, ::4], [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = a.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return a[y0:y0 + s, x0:x0 + s], (h, w)


def resize(a, n):
    # contiguous float32 copy is required: a cropped view handed to PIL is read from the wrong memory
    im = Image.fromarray(np.ascontiguousarray(a, dtype=np.float32))
    assert im.mode == "F"
    out = np.asarray(im.resize((n, n), Image.BILINEAR), dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


class Images:
    def __init__(self, paths, dino_mean, dino_std):
        self.paths, self.m, self.s = paths, np.array(dino_mean, np.float32), np.array(dino_std, np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            a, hw = load_gray01(self.paths[i])
            x512 = resize(a, 512)[None]
            x518 = resize(a, 518)
            rgb = np.stack([(x518 - self.m[c]) / self.s[c] for c in range(3)])
            return {"i": i, "ok": 1, "x512": x512, "x518": rgb.astype(np.float32), "h": hw[0], "w": hw[1]}
        except Exception:
            return {"i": i, "ok": 0, "x512": np.zeros((1, 512, 512), np.float32),
                    "x518": np.zeros((3, 518, 518), np.float32), "h": 0, "w": 0}


def load_models(torch, device, dino_dir):
    import torchxrayvision as xrv
    from transformers import AutoModel
    cxas = xrv.baseline_models.chestx_anatomy.UNetResNet50(weights=True).eval().to(device)
    targets = list(cxas.targets)
    dino = AutoModel.from_pretrained(dino_dir, dtype=torch.bfloat16).eval().to(device)
    nreg = int(getattr(dino.config, "num_register_tokens", 0) or 0)

    def seg(x512):  # x in [0,1] -> xrv range [-1024, 1024]
        y = cxas((x512 * 2 - 1) * 1024)
        y = y[0] if isinstance(y, (tuple, list)) else y
        return y

    def embed(x518):
        hs = dino(pixel_values=x518.to(torch.bfloat16)).last_hidden_state.float()
        return hs[:, 0], hs[:, 1 + nreg:]
    return seg, embed, targets


def region_index(targets):
    missing = [t for v in REGIONS.values() for t in v if t not in targets]
    assert not missing, f"CXAS targets missing: {missing}"
    return [[targets.index(t) for t in REGIONS[r]] for r in RNAMES]


def process_batch(torch, F, seg, embed, ridx, x512, x518):
    logits = seg(x512)
    prob = torch.sigmoid(logits)
    masks = torch.stack([(prob[:, ix] > 0.5).any(dim=1) for ix in ridx], dim=1)  # [B,24,512,512] bool
    cls, patches = embed(x518)
    n = patches.shape[1]
    g = int(round(n ** 0.5))
    assert g * g == n, f"patch tokens {n} not square"
    w = F.adaptive_avg_pool2d(masks.float(), g).flatten(2)          # [B,24,g*g]
    denom = w.sum(-1, keepdim=True)
    emb = torch.bmm(w, patches) / denom.clamp_min(1e-6)              # [B,24,768]
    emb = torch.where(denom > 0, emb, torch.zeros_like(emb))
    area = masks.float().mean(dim=(2, 3))                            # [B,24]
    m64 = F.adaptive_avg_pool2d(masks.float(), 64) > 0.5
    B = masks.shape[0]
    rows_any, cols_any = masks.any(dim=3), masks.any(dim=2)          # [B,24,512]
    ar = torch.arange(512, device=masks.device).float()
    big, neg = torch.full_like(ar, 1e9), torch.full_like(ar, -1e9)
    y0 = torch.where(rows_any, ar, big).min(-1).values; y1 = torch.where(rows_any, ar, neg).max(-1).values
    x0 = torch.where(cols_any, ar, big).min(-1).values; x1 = torch.where(cols_any, ar, neg).max(-1).values
    bbox = torch.stack([x0, y0, x1, y1], -1) / 511.0
    bbox = torch.where(area[..., None] > 0, bbox, torch.zeros_like(bbox))
    # QA intensities on the [0,1] input
    img = x512[:, 0]
    def mean_in(mk):
        s = mk.float().sum(dim=(1, 2))
        return (img * mk).sum(dim=(1, 2)) / s.clamp_min(1), s
    lungs = masks[:, RNAMES.index("R_lung")] | masks[:, RNAMES.index("L_lung")]
    lung_mean, lung_n = mean_in(lungs)
    heart_mean, heart_n = mean_in(masks[:, RNAMES.index("heart")])
    def cx(mk):
        s = mk.float().sum(dim=(1, 2))
        return (mk.float() * ar[None, None, :]).sum(dim=(1, 2)) / s.clamp_min(1) / 511.0, s
    rcx, rn = cx(masks[:, RNAMES.index("R_lung")])
    lcx, ln = cx(masks[:, RNAMES.index("L_lung")])
    return {
        "emb": emb, "cls": cls, "pmean": patches.mean(1), "area": area, "bbox": bbox, "m64": m64,
        "lung_area": lungs.float().mean(dim=(1, 2)),
        "zones_nonempty": (area[:, [RNAMES.index(z) for z in ZONES]] > 0).sum(1),
        "invert_suspect": (heart_n > 0) & (lung_n > 0) & (lung_mean > heart_mean),
        "lat_swap": (rn > 0) & (ln > 0) & (rcx > lcx),
        "masks_for_overlay": masks,
    }


def save_overlay(path, gray512, masks, title):
    base = (np.clip(gray512, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(base, mode="L").convert("RGB")
    px = np.array(im)
    from scipy.ndimage import binary_erosion
    for r, color in OVERLAY.items():
        mk = masks[RNAMES.index(r)]
        if mk.any():
            edge = mk & ~binary_erosion(mk, iterations=2)
            px[edge] = color
    im = Image.fromarray(px)
    d = ImageDraw.Draw(im)
    d.text((6, 6), title, fill=(255, 255, 0))
    d.text((6, 490), "image left = patient R (expected)", fill=(255, 255, 0))
    y = 22
    for r, color in OVERLAY.items():
        d.text((6, y), r, fill=color); y += 12
    im.save(path)


def main(argv=None, models=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "mimic", "padchest"])
    ap.add_argument("--limit", type=int, default=0, help="random sample size (0 = all eligible)")
    ap.add_argument("--manifest", default=str(HOME / "work/manifest"))
    ap.add_argument("--out", default=str(HOME / "work/regions"))
    ap.add_argument("--dino", default=str(HOME / "models/microsoft/rad-dino"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--overlays", type=int, default=0)
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-cudnn", action="store_true", help="disable cuDNN (fallback if the library mix is broken)")
    a = ap.parse_args(argv)

    if not a.cpu:
        gid, mem, util = pick_gpu() if a.gpu == "auto" else (a.gpu, None, None)
        if gid is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gid
            log(f"using GPU {gid} (memory.used={mem} MiB, util={util}%)")
    import torch
    import torch.nn.functional as F
    device = "cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu"
    if device == "cuda":
        if a.no_cudnn:
            torch.backends.cudnn.enabled = False
        try:  # cuDNN self-test: a broken library mix fails here, not mid-run
            conv = torch.nn.Conv2d(3, 8, 3).to(device)
            conv(torch.randn(2, 3, 64, 64, device=device)); torch.cuda.synchronize()
            log(f"cuDNN self-test ok (enabled={torch.backends.cudnn.enabled}, version={torch.backends.cudnn.version()})")
        except RuntimeError as e:
            torch.backends.cudnn.enabled = False
            log(f"cuDNN self-test FAILED ({str(e)[:90]}); continuing with cuDNN disabled (slower)")

    man = pd.read_parquet(pathlib.Path(a.manifest) / f"rex_{a.split}.parquet")
    man = man[man.eligible & man.frontal_path.notna()].sort_values("key").reset_index(drop=True)
    if a.limit:
        man = man.sample(n=min(a.limit, len(man)), random_state=0).sort_values("key").reset_index(drop=True)
    tag = f"{a.split}_pilot{a.limit}" if a.limit else a.split
    out = pathlib.Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    ov_dir = HOME / "work/qa_overlays"
    if a.overlays:
        ov_dir.mkdir(parents=True, exist_ok=True)
    log(f"{tag}: {len(man)} images, device={device}")

    cfg = pathlib.Path(a.dino) / "preprocessor_config.json"
    pc = json.load(open(cfg)) if cfg.exists() else {}
    dmean, dstd = pc.get("image_mean", [0.485, 0.456, 0.406]), pc.get("image_std", [0.229, 0.224, 0.225])
    seg, embed, targets = models(torch, device) if models else load_models(torch, device, a.dino)
    ridx = region_index(targets)

    ds = Images(man.frontal_path.tolist(), dmean, dstd)
    t0, done, qa_rows = time.time(), 0, []
    for s0 in range(0, len(man), a.shard):
        sp = out / f"shard_{s0 // a.shard:05d}.npz"
        idx = list(range(s0, min(s0 + a.shard, len(man))))
        if sp.exists():
            z = np.load(sp)
            qa_rows.append(pd.DataFrame({k: z[k] for k in ["ok", "lung_area", "zones_nonempty", "invert_suspect", "lat_swap"]}
                                        | {"view": man.frontal_view.iloc[idx].to_numpy()}))
            log(f"skip existing {sp.name}"); continue
        sub = torch.utils.data.Subset(ds, idx)
        dl = torch.utils.data.DataLoader(sub, batch_size=a.batch, num_workers=a.workers, pin_memory=(device == "cuda"))
        acc = {k: [] for k in ["emb", "cls", "pmean", "area", "bbox", "m64", "lung_area", "zones_nonempty",
                               "invert_suspect", "lat_swap", "ok", "h", "w"]}
        with torch.inference_mode():
            for b in dl:
                x512, x518 = b["x512"].to(device), b["x518"].to(device)
                r = process_batch(torch, F, seg, embed, ridx, x512, x518)
                for k in ["emb", "cls", "pmean", "area", "bbox"]:
                    acc[k].append(r[k].float().cpu().numpy().astype(np.float16))
                acc["m64"].append(np.packbits(r["m64"].cpu().numpy().reshape(len(b["i"]), len(RNAMES), -1), axis=-1))
                for k in ["lung_area", "zones_nonempty", "invert_suspect", "lat_swap"]:
                    acc[k].append(r[k].float().cpu().numpy())
                for k in ["ok", "h", "w"]:
                    acc[k].append(np.asarray(b[k]))
                if a.overlays:
                    gi = b["i"].tolist()
                    for j, i in enumerate(gi):
                        if i < a.overlays:
                            save_overlay(ov_dir / f"overlay_{i:02d}_{man.frontal_view.iloc[i]}.png",
                                         x512[j, 0].float().cpu().numpy(),
                                         r["masks_for_overlay"][j].cpu().numpy(), f"#{i} {man.frontal_view.iloc[i]}")
        cat = {k: np.concatenate(v) for k, v in acc.items()}
        np.savez(sp, keys=man.key.iloc[idx].to_numpy().astype(str), regions=np.array(RNAMES), **cat)
        qa_rows.append(pd.DataFrame({k: cat[k] for k in ["ok", "lung_area", "zones_nonempty", "invert_suspect", "lat_swap"]}
                                    | {"view": man.frontal_view.iloc[idx].to_numpy()}))
        done += len(idx)
        el = time.time() - t0
        log(f"{sp.name}: {done}/{len(man)} images, {done / max(el, 1e-6):.1f} img/s")

    qa = pd.concat(qa_rows, ignore_index=True)
    qa["fail"] = (qa.ok == 0) | (qa.lung_area < 0.15) | (qa.lung_area > 0.70) | (qa.zones_nonempty < 8)
    summ = {"images": int(len(qa)), "read_errors": int((qa.ok == 0).sum()),
            "fail_rate": round(float(qa.fail.mean()), 4),
            "invert_suspect_rate": round(float(qa.invert_suspect.mean()), 4),
            "laterality_swap_rate": round(float(qa.lat_swap.mean()), 4),
            "lung_area_p5_p50_p95": np.round(np.percentile(qa.lung_area, [5, 50, 95]), 3).tolist(),
            "by_view": {v: {"n": int(len(g)), "fail_rate": round(float(g.fail.mean()), 4),
                            "invert_suspect_rate": round(float(g.invert_suspect.mean()), 4)}
                        for v, g in qa.groupby("view")}}
    if done:
        summ["throughput_img_s"] = round(done / (time.time() - t0), 1)
    if device == "cuda":
        summ["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        summ["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
    areas = []
    for sp in sorted(out.glob("shard_*.npz")):
        areas.append(np.load(sp)["area"].astype(np.float32))
    A = np.concatenate(areas)
    summ["region_area_pct_p5_p50_p95"] = {r: np.round(np.percentile(A[:, i], [5, 50, 95]) * 100, 1).tolist() for i, r in enumerate(RNAMES)}
    summ["region_empty_rate"] = {r: round(float((A[:, i] == 0).mean()), 4) for i, r in enumerate(RNAMES)}
    gate = {"fail<3%": summ["fail_rate"] < 0.03, "invert<1%": summ["invert_suspect_rate"] < 0.01,
            "lat_swap<1%": summ["laterality_swap_rate"] < 0.01,
            "throughput>=50/s": summ.get("throughput_img_s", 0) >= 50}
    summ["gate_G1_auto"] = gate
    json.dump(summ, open(out / "qa_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    log("REGION_CACHE_DONE" + (" (all automatic gate checks pass)" if all(gate.values()) else " (some automatic gate checks FAIL)"))


if __name__ == "__main__":
    main()
