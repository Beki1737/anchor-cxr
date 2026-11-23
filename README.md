# ANCHOR: certified claim admission for chest radiograph reporting

Code for the paper "Propose broadly, admit selectively: anatomy-grounded chest radiograph
reporting with certified control of false findings" (Gudeta and Jang, Medical Image Analysis, under review).

## Data
No data are included. Obtain the datasets under their own data use agreements:
ReXGradient-160K (Stanford AIMI), MIMIC-CXR and MIMIC-CXR-JPG (PhysioNet, credentialed access)
and PadChest-GR (BIMCV). Set the dataset paths in the configuration files before running.

## Contents
Scripts for region embedding, probe training, retrieval, claim extraction, scoring, conformal
calibration (CRC, multi-group CRC, LTT), monitoring and recalibration, and the analysis scripts
that produce the tables and figures of the paper.

## Licence
MIT, see LICENSE.
