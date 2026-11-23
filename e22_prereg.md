# E22 pre-registration: normal-referenced region evidence (NRE) for critical-finding admission
Written 2026-11-10 before any E22 code was run. Its md5 is printed at the top of r47.log.
Author approval: gate 2 (2026-11-10), with these conditions: train and select on validation data only, keep the current
frozen FUS-multi as the comparison, criteria fixed in advance, a new validation-only freeze with checksums, at most one
labelled test look, and honest reporting if it fails. If it fails, the paper keeps the current method.

## Diagnosis this answers
At the frozen operating point FUS-multi states 26% of critical reference findings (folds 2-3 critical recall 0.259).
If every critical candidate were admitted, 91% would be covered (E21 ceiling 0.913). So the critical findings are
already among the candidates. What fails is the evidence that separates true critical candidates from false ones.
The current evidence is all supervised by weak report-text labels: probe heads and neighbour votes.

## The one idea
Each anatomical region is compared with what NORMAL looks like for that same region, using two references.
- **Population reference:** a bank of region embeddings (RAD-DINO, CXAS region masks) from ReXGradient TRAIN studies
  whose reports state nothing compatible with that region. This is the same database the frozen retrieval already uses.
  Score: mean cosine distance to the k = 20 nearest normal same-region embeddings, z-scored against held-out normal
  regions (cf. anatomy-partitioned memory banks, AMBER 2026).
- **Patient reference:** the patient's own contralateral region. Score: cosine distance between paired left and right
  regions, z-scored against pairs where both sides are normal.
- **Per critical candidate:** the probe's own region map for that finding (side-restricted when the claim has a side)
  says WHERE to look. The two deviations, averaged with those weights, say HOW abnormal it looks there. Features:
  nre_pop, nre_asym and nre_max (largest population deviation in the study).
- **The idea in one line:** anatomically matched neighbours no longer only PROPOSE claims (abnormal neighbours); they
  also TEST them (deviation from normal neighbours).

## What is fitted, and where
- No network is trained. The normal bank uses train studies only as a retrieval database, as frozen ANCHOR does.
- Critical scorer: logistic regression on FUS features + finding indicators + the 3 NRE features, fitted on valid
  fold 0, critical rows only. Non-critical rows keep their frozen FUS scores unchanged.
- Thresholds: marginal CRC and multi-group thresholds (same groups, n >= 40, alpha 0.10) calibrated on valid fold 1,
  same procedure as freeze2.
- Selection and judging: valid folds 2-3 (the same 3,330 evaluation studies as E16).
- Attribution control (CTRL): identical, but without the 3 NRE features. It tells whether a gain comes from the normal
  references or merely from giving critical findings their own scorer.
- Test: not read in this round.

## Success criteria (all must hold; candidate system = NRE-multi; comparator = frozen FUS-multi)
| | Criterion | Rule |
|---|---|---|
| S1 (primary) | Critical recall, folds 2-3 | NRE-multi minus FUS-multi >= +0.020 absolute, patient-cluster bootstrap 95% CI lower bound > 0 |
| S2 (safety) | Certified risk | realised capped false-finding risk <= 0.10 on folds 2-3, and worst subgroup (view / institution / vendor, n >= 40) <= 0.10 |
| S3 (quality) | CRIMSON and GREEN, folds 2-3 | NRE-multi minus FUS-multi: lower 95% CI >= -0.005 for each (non-inferiority) |
| S4 (no trade-off) | Overall claim recall | NRE-multi minus FUS-multi >= -0.010 |
- One primary endpoint (S1). S2 to S4 are gates, so no alpha is split.
- **Attribution (reported, not a gate):** NRE-multi minus CTRL-multi critical recall, with a patient-cluster 95% CI.
  If the CI includes 0, the paper may NOT claim that normal references are the reason for the gain.
- **Secondary (descriptive):** critical claim AUROC (frozen FUS vs CTRL vs NRE scorer); per-finding critical recall;
  share of studies with at least one missed critical finding.

## Failure
Any of S1 to S4 not met means failure. Then:
- no freeze3 test look;
- the paper keeps ANCHOR / FUS-multi;
- E22 is reported briefly as a negative result.

## If it succeeds
- freeze3.json is written from valid only, with code, bank and pre-registration checksums.
- freeze3 is applied ONCE to the ReXGradient test set, labelled "second look for freeze3 (selected on validation)".
  Note that the test split was already used once for the freeze2 second look.
- It is applied once to MIMIC (external).
- The headline result stays the one-shot frozen ANCHOR.
