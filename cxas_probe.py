#!/usr/bin/env python3
"""Probe the 159-class CXAS anatomy model in torchxrayvision on one IU X-ray image (CPU).
Prints: preprocessing hints from the library source, target names, output shape, per-structure areas."""
import glob
import inspect
import os
import re

import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv
import torchxrayvision.baseline_models.chestx_anatomy as ca

print("--- preprocessing hints in chestx_anatomy source ---")
for i, line in enumerate(inspect.getsource(ca).splitlines()):
    if re.search(r"normali|resize|512|mean|std|sigmoid|softmax|argmax|targets|def forward|transform|weights_url|url", line, re.I):
        print(f"{i:4}: {line.strip()[:140]}")

m = ca.UNetResNet50(weights=True).eval()
targets = getattr(m, "targets", None) or getattr(m, "classes", None)
print(f"\nn_targets={len(targets) if targets is not None else None}")
if targets is not None:
    print(list(targets))

f = sorted(glob.glob(os.path.expanduser("~/anchor/data/iu_xray/png/**/*.png"), recursive=True))[0]
img = skimage.io.imread(f)
img = img if img.ndim == 2 else img[..., 0]
img = xrv.datasets.normalize(img.astype(np.float32), 255)[None, ...]
img = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])(img)
x1 = torch.from_numpy(img)[None].float()
for name, x in [("1ch", x1), ("3ch", x1.repeat(1, 3, 1, 1))]:
    try:
        with torch.no_grad():
            y = m(x)
        y = y[0] if isinstance(y, (tuple, list)) else y
        print(f"\ninput {name} {tuple(x.shape)} -> output {tuple(y.shape)} range=({y.min().item():.2f},{y.max().item():.2f})")
        p = torch.sigmoid(y[0]) if (y.min() < 0 or y.max() > 1) else y[0]
        areas = (p > 0.5).float().mean(dim=(1, 2)).numpy() * 100
        names = list(targets) if targets is not None else [str(i) for i in range(len(areas))]
        order = np.argsort(-areas)
        print("top 15 by area %:", [(names[i], round(float(areas[i]), 1)) for i in order[:15]])
        lungish = [(n, round(float(a), 1)) for n, a in zip(names, areas) if re.search(r"lung|lobe|zone|heart|hilum|mediast|diaphragm|costophrenic", n, re.I)]
        print("lung/heart related:", lungish[:40])
        print(f"structures with area>0: {(areas > 0).sum()} / {len(areas)}")
        break
    except Exception as e:
        print(f"input {name} failed: {type(e).__name__}: {e}")
print("CXAS_PROBE_DONE")
