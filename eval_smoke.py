#!/usr/bin/env python3
"""E2 harness smoke test (run in ~/venvs/anchor_eval). Inspects the installed RadEval API, then scores
references against: themselves (ceiling), a shuffled reference (floor), one normal template, and any B0
generations given with --gen. One metric at a time, so a failing metric does not hide the others.
Prints aggregates only; per-study scores are saved locally."""
import argparse
import inspect
import json
import pathlib
import random
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
TEMPLATE = "The lungs are clear. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal."


def alias_radeval():
    """Some radeval submodules import the package as 'RadEval'; map that name to the installed 'radeval'."""
    import importlib
    import sys
    import radeval
    sys.modules.setdefault("RadEval", radeval)

    def _cnz(X, axis=None, sample_weight=None):  # sklearn>=1.9 count_nonzero only accepts sparse input
        return np.count_nonzero(np.asarray(X), axis=axis)
    for modname in ["radeval.metrics._chexbert_base", "RadEval.metrics._chexbert_base"]:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "count_nonzero"):
                mod.count_nonzero = _cnz
        except Exception:
            pass


def available_metrics():
    from radeval import RadEval
    try:
        RadEval(metrics=["__list__"])
    except Exception as e:
        return str(e)
    return "?"


def make_scorer(metric):
    alias_radeval()
    from radeval import RadEval
    params = inspect.signature(RadEval.__init__).parameters
    if "metrics" in params:
        ev = RadEval(metrics=[metric])
    else:
        ev = RadEval(**{f"do_{metric}": True})

    def score(preds, refs):
        if hasattr(ev, "evaluate"):
            try:
                return ev.evaluate(predictions=preds, references=refs)
            except TypeError:
                pass
        return ev(refs=refs, hyps=preds)
    return score


def flatten(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{pre}{k}."))
    elif isinstance(d, (int, float, np.floating)) and not isinstance(d, bool):
        out[pre.rstrip(".")] = float(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--metrics", default="radgraph,chexbert,ratescore,bleu,bertscore,green")
    ap.add_argument("--gen", default="", help="B0 generations parquet (key, generated)")
    ap.add_argument("--split", default="train", help="manifest split of the --gen keys")
    ap.add_argument("--skip-ceiling", action="store_true")
    ap.add_argument("--only-gen", action="store_true", help="score only the --gen system (no reference conditions)")
    ap.add_argument("--keys-file", default="", help="parquet with a key column: restrict scoring to these studies")
    a = ap.parse_args(argv)

    import radeval
    from radeval import RadEval
    print("radeval", getattr(radeval, "__version__", "?"), "| RadEval.__init__", inspect.signature(RadEval.__init__))
    for name in ["evaluate", "__call__"]:
        if hasattr(RadEval, name):
            print(f"  RadEval.{name}{inspect.signature(getattr(RadEval, name))}")
    import numpy
    print("numpy", numpy.__version__)
    print("available:", available_metrics())

    mdir = HOME / "work/manifest"
    if a.gen:
        g = pd.read_parquet(a.gen)
        if a.keys_file:
            g = g[g.key.isin(set(pd.read_parquet(a.keys_file).key))].reset_index(drop=True)
        man = pd.read_parquet(mdir / f"rex_{a.split}.parquet").set_index("key")
        refs = man.loc[g.key, "findings"].fillna("").tolist()
        conds = {"b0": g.generated.fillna("").tolist()}
    else:
        dev = pd.read_parquet(mdir / "dev_slice.parquet").key.tolist()[: a.n]
        man = pd.read_parquet(mdir / "rex_train.parquet").set_index("key")
        refs = man.loc[dev, "findings"].fillna("").tolist()
        conds = {}
    sh = refs[:]; random.Random(0).shuffle(sh)
    conds = {"ceiling_self": refs, "floor_shuffled": sh, "normal_template": [TEMPLATE] * len(refs), **conds}
    if a.skip_ceiling:
        conds.pop("ceiling_self")
    if a.only_gen:
        conds = {"b0": conds["b0"]}

    table, per = {}, {}
    for m in [x.strip() for x in a.metrics.split(",") if x.strip()]:
        t0 = time.time()
        try:
            score = make_scorer(m)
            for c, preds in conds.items():
                res = score(preds, refs)
                fl = flatten(res)
                table[(m, c)] = {k: round(v, 4) for k, v in fl.items() if not k.endswith("_list")}
                per[f"{m}|{c}"] = res if isinstance(res, dict) else str(res)
            print(f"[{m}] ok in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            import traceback
            print(f"[{m}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            tb = traceback.format_exc().splitlines()
            print("   traceback tail:\n   " + "\n   ".join(l.replace(str(pathlib.Path.home()), "~") for l in tb[-14:]), flush=True)
    print("\n=== metric x condition (aggregate scores) ===")
    for (m, c), v in table.items():
        print(f"{m:<10} {c:<16} {json.dumps(v)[:300]}")
    out = HOME / "work/eval"; out.mkdir(parents=True, exist_ok=True)
    tag = (pathlib.Path(a.gen).stem + ("_evalkeys" if a.keys_file else "")) if a.gen else "refs_only"
    mtag = "-".join(sorted(x.strip() for x in a.metrics.split(",") if x.strip()))
    json.dump({f"{m}|{c}": v for (m, c), v in table.items()}, open(out / f"smoke_{tag}__{mtag}.json", "w"), indent=1)
    print("EVAL_SMOKE_DONE")


if __name__ == "__main__":
    main()
