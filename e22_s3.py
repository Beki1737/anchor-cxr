#!/usr/bin/env python3
"""E22 S3 verdict (pre-registered): CRIMSON and GREEN, NRE-multi minus frozen FUS-multi on valid folds 2-3,
lower 95% study-bootstrap CI >= -0.005 for each. Also NRE minus CTRL and CTRL minus FUS-multi (descriptive).
Patient-cluster CI printed as a robustness check. ~/venvs/anchor_eval, CPU. Aggregates only."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import paired  # noqa: E402


def cluster(d, pat, reps=2000, seed=47):
    up, inv = np.unique(pat, return_inverse=True); s, c = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(seed); b = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)); b.append((w @ s) / (w @ c))
    return np.quantile(b, [0.025, 0.975])


def main():
    keys = set(pd.read_parquet(HOME / "work/e8/eval_keys_valid.parquet").key)
    man = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key")
    ok, out = True, {}
    for judge in ["crimson", "green"]:
        sc = pd.concat([pd.read_parquet(HOME / "work/judge" / t / f"{judge}_scores.parquet") for t in ["valid_e16", "valid_e22"]])
        W = sc[sc.key.isin(keys)].drop_duplicates(["cond", "key"]).pivot_table(index="key", columns="cond", values="score")
        for a, b in [("e22nremulti", "e16fusmulti"), ("e22ctrlmulti", "e16fusmulti"), ("e22nremulti", "e22ctrlmulti")]:
            x = W[[a, b]].dropna(); p = paired(x.rename(columns={a: "sys"}), "sys", b)
            cl = cluster((x[a] - x[b]).to_numpy(), man.loc[x.index, "patient"].astype(str).to_numpy())
            print(f"{judge.upper():<8} {a} minus {b}: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] "
                  f"(patient-cluster [{cl[0]:+.4f}, {cl[1]:+.4f}]) n {p['n']} | means {x[a].mean():.4f} vs {x[b].mean():.4f}")
            out[f"{judge}|{a}|{b}"] = p
            if (a, b) == ("e22nremulti", "e16fusmulti"):
                ok &= p["ci95"][0] >= -0.005
    print(f"S3 CRIMSON and GREEN non-inferiority (lower CI >= -0.005): {'PASS' if ok else 'FAIL'}")
    fz = HOME / "work/e22/freeze3.json"; z = json.load(open(fz)); z["valid_verdict"]["S3"] = bool(ok)
    json.dump(z, open(fz, "w"), indent=1, default=float)
    v = z["valid_verdict"]; allp = all(v[k] is True for k in ["S1", "S2", "S3", "S4"])
    print(f"E22 OVERALL (S1-S4): {'PASS: freeze3 eligible for one labelled test look' if allp else 'FAIL: paper keeps the current method'}")
    import hashlib
    print(f"freeze3.json md5 after S3 {hashlib.md5(open(fz, 'rb').read()).hexdigest()}")
    print("E22_S3_DONE")


if __name__ == "__main__":
    main()
