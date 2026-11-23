#!/usr/bin/env python3
"""E22 (round 47): NORMAL-REFERENCED REGION EVIDENCE (NRE) for critical-finding admission. Pre-registered in
e22_prereg.md (md5 printed in r47.log before this runs). VALIDATION ONLY: no test, no MIMIC in this round.
(~/venvs/anchor, one GPU of 0/1, about 8 GB.)

  python e22_nre.py              build normal banks (train = retrieval database only), NRE features for valid,
                                 critical scorer (fold 0), CRC + multi-group thresholds (fold 1), evaluation on folds 2-3
                                 vs frozen FUS-multi and CTRL, S1 / S2 / S4 verdicts, reports for judges, freeze3.json
Writes work/e22/{bank.npz, nre_valid.npz, fu_valid_nre.parquet, freeze3.json, results_valid.json},
work/gen/e22nremulti_valid.parquet, work/gen/e22ctrlmulti_valid.parquet. Aggregates only in the log."""
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import e4a_pool as E4  # noqa: E402
import e4b_verify as E4B  # noqa: E402
import e5_anchor as A5  # noqa: E402
import e8_final as F  # noqa: E402
import e11_ablate as X  # noqa: E402
import e16_fusion as FU  # noqa: E402

OUT = HOME / "work/e22"
CRIT = set(F.CRIT)
K, BANK, HOLD, SEED = 20, 20000, 2000, 22
NREF = ["nre_pop", "nre_asym", "nre_max"]
PAIR = {"R_apical": "L_apical", "R_upper": "L_upper", "R_mid": "L_mid", "R_base": "L_base", "R_upper_lobe": "L_upper_lobe",
        "R_lower_lobe": "L_lower_lobe", "R_lung": "L_lung", "R_hemidiaphragm": "L_hemidiaphragm"}
PAIR.update({v: k for k, v in list(PAIR.items())})
RN = E4.RN
RI = {r: i for i, r in enumerate(RN)}
CONTRA = np.array([RI.get(PAIR.get(r, ""), -1) for r in RN])
LEFT = np.array([r.startswith("L_") for r in RN]); RIGHT = np.array([r.startswith("R_") for r in RN])


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------ normal banks (train = retrieval database only)
def region_normal(keys, area, corp):
    """normal[n, r]: region visible and no positive claim (any finding) of the report is compatible with it."""
    N = np.zeros((len(keys), len(RN)), bool)
    for n, k in enumerate(keys):
        bad = set()
        for c, regs in corp.get(k, {}).items():
            for x in E4B.claim_regions(c, regs):
                bad.update(r for r in RN if (r, x) in E4.COMPAT)
        N[n] = (area[n] > 0) & np.array([r not in bad for r in RN])
    return N


def build_banks():
    t0 = time.time()
    E4.COMPAT = {(q, c) for q in RN for c in RN if E4.compat_region(q, c)}
    corp = E4.load_claims(HOME / "work/corpus_v2/train")
    flags = pd.read_parquet(HOME / "work/manifest/flags_train.parquet")
    prim = set(flags[flags.cohort_primary].key) & set(corp)
    shards = sorted((HOME / "work/regions/train").glob("shard_*.npz"))
    cand = {r: [] for r in range(len(RN))}
    for si, sp in enumerate(shards):
        z = np.load(sp); ks = z["keys"]; ar = z["area"]
        ok = np.array([k in prim for k in ks])
        if not ok.any():
            continue
        rows = np.where(ok)[0]
        Nm = region_normal([ks[i] for i in rows], ar[rows], corp)
        for j, i in enumerate(rows):
            for r in np.where(Nm[j])[0]:
                c = CONTRA[r]
                cand[r].append((si, int(i), bool(c >= 0 and Nm[j, c])))
    rng = np.random.default_rng(SEED)
    sel = {}
    for r, lst in cand.items():
        idx = rng.permutation(len(lst))[: BANK + HOLD]
        sel[r] = [lst[i] for i in idx]
    need = {}
    for r, lst in sel.items():
        for p, (si, i, bn) in enumerate(lst):
            need.setdefault(si, []).append((i, r, p, bn))
    bank = {r: np.zeros((len(sel[r]), 768), np.float32) for r in sel}
    contra = {r: np.full((len(sel[r]), 768), np.nan, np.float32) for r in sel}
    for si, lst in need.items():
        em = np.load(shards[si])["emb"]
        for i, r, p, bn in lst:
            bank[r][p] = em[i, r]
            if bn:
                contra[r][p] = em[i, CONTRA[r]]
    print(f"normal banks: {len(shards)} train shards, primary-cohort studies {len(prim)}; per-region normal candidates "
          f"min {min(len(v) for v in cand.values())} / median {int(np.median([len(v) for v in cand.values()]))}; "
          f"bank {BANK} + held-out {HOLD} per region ({time.time() - t0:.0f}s)", flush=True)
    return bank, contra


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def knn_dist(torch, Q, B, k=K, exclude_self=False):
    Bt = torch.from_numpy(unit(B)).half().cuda(); out = []
    for s in range(0, len(Q), 4096):
        q = torch.from_numpy(unit(Q[s:s + 4096])).half().cuda()
        sim = q @ Bt.T
        top = sim.topk(k + int(exclude_self), dim=1).values[:, int(exclude_self):]
        out.append((1 - top.float().mean(1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def nre_stats(torch, bank, contra):
    st = {}
    for r in range(len(RN)):
        nb = min(BANK, int(0.9 * len(bank[r])))
        B, H = bank[r][:nb], bank[r][nb:]
        if len(B) < K + 1 or len(H) < 50:
            st[r] = None; continue
        d = knn_dist(torch, H, B)
        hc = contra[r][nb:]; okc = ~np.isnan(hc).any(1)
        a = 1 - (unit(H[okc]) * unit(hc[okc])).sum(1) if okc.sum() >= 50 else np.array([])
        st[r] = {"pop_mu": float(d.mean()), "pop_sd": float(d.std() + 1e-6),
                 "asym_mu": float(a.mean()) if len(a) else None, "asym_sd": float(a.std() + 1e-6) if len(a) else None,
                 "n_bank": int(len(B)), "n_hold": int(len(H)), "n_hold_pairs": int(okc.sum())}
    return st


def region_features(torch, split, keys, bank, st):
    import e4c_probe as P
    ks, Em, Ar, _ = P.load_split(split, sorted(keys))
    Em = Em.astype(np.float32); vis = Ar > 0
    zpop = np.full((len(ks), len(RN)), np.nan, np.float32); zasy = np.full_like(zpop, np.nan)
    for r in range(len(RN)):
        if st[r] is None:
            continue
        m = vis[:, r]
        if m.any():
            zpop[m, r] = (knn_dist(torch, Em[m, r], bank[r][:st[r]["n_bank"]]) - st[r]["pop_mu"]) / st[r]["pop_sd"]
        c = CONTRA[r]
        if c >= 0 and st[r]["asym_mu"] is not None:
            mc = vis[:, r] & vis[:, c]
            a = 1 - (unit(Em[mc, r]) * unit(Em[mc, c])).sum(1)
            zasy[mc, r] = (a - st[r]["asym_mu"]) / st[r]["asym_sd"]
    return ks, zpop, zasy, vis


def add_nre(fu, probe, ks, zpop, zasy, vis):
    pos = {k: i for i, k in enumerate(ks)}
    ii = fu.key.map(pos).to_numpy()
    okr = ~pd.isna(ii); ii = np.where(okr, ii, 0).astype(int)
    pi = fu.key.map(probe["pos"]).fillna(-1).astype(int).to_numpy()
    fj = fu.finding.map(probe["F"]).fillna(-1).astype(int).to_numpy()
    W = np.ones((len(fu), len(RN)), np.float32)
    has = (pi >= 0) & (fj >= 0)
    W[has] = probe["Pr"][pi[has], :, fj[has]]
    V = vis[ii] & okr[:, None]
    lat = fu.lat.astype(str).to_numpy()
    side = np.ones_like(V)
    side[lat == "left"] = LEFT; side[lat == "right"] = RIGHT
    Wp = W * V * side
    zp, za = zpop[ii], zasy[ii]
    def wavg(Z, Wm):
        Wm = np.where(np.isnan(Z), 0, Wm); s = Wm.sum(1)
        return np.where(s > 0, (Wm * np.nan_to_num(Z)).sum(1) / np.maximum(s, 1e-9), 0.0)
    fu = fu.copy()
    fu["nre_pop"] = wavg(zp, Wp); fu["nre_asym"] = wavg(za, Wp)
    fu["nre_max"] = np.where(okr, np.nan_to_num(np.nanmax(np.where(V, zp, np.nan), 1), nan=0.0), 0.0)
    return fu


# ------------------------------------------------------------------ evaluation helpers
def crit_cov(adm, ref, keys):
    by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    cov, n, perf = np.zeros(len(keys)), np.zeros(len(keys)), {}
    for i, k in enumerate(keys):
        A = by.get(k, [])
        for r in ref.get(k, {}):
            if r[0] in CRIT:
                hit = E4.matches(r, A)
                n[i] += 1; cov[i] += hit
                c = perf.setdefault(r[0], [0, 0]); c[0] += hit; c[1] += 1
    return cov, n, perf


def cluster_ci(a, b, n, pat, reps=2000, seed=47):
    up, inv = np.unique(pat, return_inverse=True)
    A, B, Nn = np.bincount(inv, a), np.bincount(inv, b), np.bincount(inv, n)
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))
        d.append((w @ B - w @ A) / max(w @ Nn, 1))
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def evaluate(tag, d, th, keys, ref, probe, tau, groups):
    e = d[d.key.isin(set(keys))]
    adm = F.resolve_sides(e[e.score.to_numpy() >= e.key.map(th).to_numpy()], probe, tau, ref)
    ct = FU.claim_table(tag, adm, ref, keys)
    cov, n, perf = crit_cov(adm, ref, keys)
    fp = pd.Series(A5.fp_per_study(adm.assign(score=1.0), 0.5, keys), index=keys)
    worst = max(float((np.minimum(fp[g[g == v].index], 3) / 3).mean()) for g in groups.values() for v in g.unique() if (g == v).sum() >= 40)
    crit_recall = cov.sum() / max(n.sum(), 1)
    anymiss = float(((n - cov) > 0).mean())
    print(f"  {tag:<22} critical recall {crit_recall:.4f} | worst subgroup risk {worst:.4f} | studies with >=1 missed critical {anymiss:.3f}")
    return {"claims": ct, "crit_recall": float(crit_recall), "worst_subgroup": worst, "any_crit_missed": anymiss,
            "per_finding": {f: round(c[0] / c[1], 4) for f, c in perf.items()}, "_cov": cov, "_n": n, "_adm": adm}


def main():
    import torch
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fz = json.load(open(F.OUT / "freeze.json")); fz2 = json.load(open(FU.OUT / "freeze2.json"))
    prereg = CODE / "e22_prereg.md"
    print(f"pre-registration {prereg.name} md5 {md5(prereg)} | freeze.json {md5(F.OUT / 'freeze.json')} | freeze2.json {md5(FU.OUT / 'freeze2.json')}")
    bank, contra = build_banks()
    st = nre_stats(torch, bank, contra)
    np.savez_compressed(OUT / "bank.npz", **{f"r{r}": bank[r][:(st[r] or {}).get("n_bank", 0)].astype(np.float16) for r in bank})
    json.dump({RN[r]: v for r, v in st.items()}, open(OUT / "nre_stats.json", "w"), indent=1)
    # ---- valid features
    C = F.load_cands("valid"); ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    keys_all = sorted(C.key.unique())
    probe = F.probe_arrays("valid", keys_all)
    ks, zpop, zasy, vis = region_features(torch, "valid", keys_all, bank, st)
    np.savez_compressed(OUT / "nre_valid.npz", keys=np.array([str(k) for k in ks]), zpop=zpop, zasy=zasy)
    _, fu = FU.build("valid", C, probe, ref, fz2["tau"])
    fu = add_nre(fu, probe, ks, zpop, zasy, vis)
    fu["frozen"] = F.apply_scorer(fu, fz2["scorer_fus"], fz2["delta"])
    cm = fu.finding.isin(CRIT).to_numpy()
    for f in sorted(CRIT):
        fu[f"is_{f.replace(' ', '_')}"] = (fu.finding == f).astype(float)
    onehot = [f"is_{f.replace(' ', '_')}" for f in sorted(CRIT)][1:]
    dc = fu[cm]
    print(f"\nVALID FUS pool {len(fu)} rows, critical rows {cm.sum()} ({time.time() - t0:.0f}s)")
    print("feature sanity, critical rows of fold 0 (label AUROC): " + ", ".join(
        f"{c} {E4.auroc(dc[dc.fold == 0][c], dc[dc.fold == 0].label):.3f}" for c in ["probe_region", "probe_global", "frozen"] + NREF))
    base = X.drop_const(dc, list(FU.FFU)) + onehot
    m_ctrl = F.fit(dc, base); m_nre = F.fit(dc, base + NREF)
    print("NRE scorer weights (standardised): " + ", ".join(f"{a}={w:+.2f}" for a, w in zip(m_nre["feats"], m_nre["w"]) if a in NREF or a.startswith("probe")))
    sysd = {}
    for nm, m in [("nre", m_nre), ("ctrl", m_ctrl)]:
        s = fu.frozen.to_numpy().copy(); s[cm] = F.apply_scorer(fu[cm], m, fz2["delta"])
        sysd[nm] = fu.assign(score=s)
    ev_rows = fu[(fu.fold >= 2) & cm]
    print("critical claim AUROC on folds 2-3: frozen FUS " + f"{E4.auroc(ev_rows.frozen, ev_rows.label):.4f}, CTRL "
          f"{E4.auroc(sysd['ctrl'][(fu.fold >= 2) & cm].score, ev_rows.label):.4f}, NRE {E4.auroc(sysd['nre'][(fu.fold >= 2) & cm].score, ev_rows.label):.4f}")
    # ---- thresholds on fold 1 (freeze2 procedure)
    fz3 = {"alpha": F.ALPHA, "tau": fz2["tau"], "delta": fz2["delta"], "group_tops": fz2["group_tops"],
           "bank": {"k": K, "bank_per_region": BANK, "held_out": HOLD, "seed": SEED}, "nre_stats": {RN[r]: v for r, v in st.items()}}
    thr = {}
    for nm, d in sysd.items():
        grid = np.quantile(d.score, np.linspace(1, 0, 600)); cal = d[d.fold == 1]
        t = float(A5.crc_threshold(cal, F.fp_loss, F.ALPHA, grid))
        gt = FU.group_thr("valid", d, keys_all, None, t, tops=fz2["group_tops"], cal=cal, grid=grid)
        fz3[f"scorer_crit_{nm}"] = {**(m_nre if nm == "nre" else m_ctrl)}; fz3[f"t_{nm}"] = t; fz3[f"group_thresholds_{nm}"] = gt
        thr[nm] = (t, gt)
    keys = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key)
    G = F.groups_of("valid", keys, fz2["group_tops"])
    th_fus = FU.group_thr("valid", fu, keys, {"group_thresholds": fz2["group_thresholds_fus"]}, fz2["t_fus"], tops=fz2["group_tops"])
    print(f"\n=== VALID folds 2-3 ({len(keys)} studies), alpha {F.ALPHA}, multi-group ===")
    R = {"fusmulti": evaluate("frozen FUS-multi", fu.assign(score=fu.frozen), th_fus, keys, ref, probe, fz2["tau"], G)}
    for nm in ["ctrl", "nre"]:
        t, gt = thr[nm]
        th = FU.group_thr("valid", sysd[nm], keys, {"group_thresholds": gt}, t, tops=fz2["group_tops"])
        R[nm] = evaluate(f"{nm.upper()}-multi (t {t:.3f})", sysd[nm], th, keys, ref, probe, fz2["tau"], G)
    man = pd.read_parquet(HOME / "work/manifest/rex_valid.parquet").set_index("key").reindex(keys)
    pat = man.patient.astype(str).to_numpy()
    lo, hi = cluster_ci(R["fusmulti"]["_cov"], R["nre"]["_cov"], R["nre"]["_n"], pat)
    alo, ahi = cluster_ci(R["ctrl"]["_cov"], R["nre"]["_cov"], R["nre"]["_n"], pat)
    clo, chi = cluster_ci(R["fusmulti"]["_cov"], R["ctrl"]["_cov"], R["ctrl"]["_n"], pat)
    d1 = R["nre"]["crit_recall"] - R["fusmulti"]["crit_recall"]
    print("\nper-finding critical recall (FUS-multi / CTRL / NRE): " + "; ".join(
        f"{f} {R['fusmulti']['per_finding'].get(f, 0):.2f}/{R['ctrl']['per_finding'].get(f, 0):.2f}/{R['nre']['per_finding'].get(f, 0):.2f}" for f in sorted(CRIT)))
    s1 = d1 >= 0.020 and lo > 0
    s2 = R["nre"]["claims"]["risk"] <= F.ALPHA and R["nre"]["worst_subgroup"] <= F.ALPHA
    s4 = R["nre"]["claims"]["lenient"]["recall"] - R["fusmulti"]["claims"]["lenient"]["recall"] >= -0.010
    print(f"\nS1 critical recall NRE-multi minus FUS-multi {d1:+.4f} [{lo:+.4f}, {hi:+.4f}] (need >= +0.020 and lower > 0): {'PASS' if s1 else 'FAIL'}")
    print(f"S2 realised risk {R['nre']['claims']['risk']:.4f}, worst subgroup {R['nre']['worst_subgroup']:.4f} (need <= 0.10): {'PASS' if s2 else 'FAIL'}")
    print(f"S4 overall recall NRE {R['nre']['claims']['lenient']['recall']:.4f} vs FUS-multi {R['fusmulti']['claims']['lenient']['recall']:.4f} "
          f"(need >= -0.010): {'PASS' if s4 else 'FAIL'}")
    print(f"ATTRIBUTION NRE-multi minus CTRL-multi critical recall {R['nre']['crit_recall'] - R['ctrl']['crit_recall']:+.4f} [{alo:+.4f}, {ahi:+.4f}]; "
          f"CTRL minus FUS-multi {R['ctrl']['crit_recall'] - R['fusmulti']['crit_recall']:+.4f} [{clo:+.4f}, {chi:+.4f}]")
    print("S3 (CRIMSON / GREEN non-inferiority) is decided by the judge step that follows.")
    for nm in ["nre", "ctrl"]:
        by = {k: list(zip(g.finding, g.lat)) for k, g in R[nm]["_adm"].groupby("key")}
        pd.DataFrame({"key": keys, "generated": [A5.realize(by.get(k, [])) for k in keys]}).to_parquet(HOME / f"work/gen/e22{nm}multi_valid.parquet", index=False)
    fz3["code_md5"] = {f: md5(CODE / f) for f in ["e22_nre.py", "e22_prereg.md", "e16_fusion.py", "e8_final.py"]}
    fz3["bank_md5"] = md5(OUT / "bank.npz")
    fz3["valid_verdict"] = {"S1": bool(s1), "S2": bool(s2), "S4": bool(s4), "S3": "pending judges"}
    json.dump(fz3, open(OUT / "freeze3.json", "w"), indent=1, default=float)
    res = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in R.items()}
    res["S1"] = {"diff": d1, "ci": [lo, hi]}; res["attribution"] = {"nre_minus_ctrl": [alo, ahi], "ctrl_minus_fus": [clo, chi]}
    json.dump(res, open(OUT / "results_valid.json", "w"), indent=1, default=float)
    print(f"freeze3.json md5 {md5(OUT / 'freeze3.json')} (valid only; test not read) ({time.time() - t0:.0f}s)")
    print("E22_DONE")


if __name__ == "__main__":
    main()
