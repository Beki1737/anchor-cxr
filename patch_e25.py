#!/usr/bin/env python3
"""r49 patch for e25_findings.py (MIMIC radiologist block crashed: KeyError 'Lung Opacity').
The MIMIC-CXR 2.1.0 test-set label file names the column 'Airspace Opacity'. This patch
(1) renames 'Lung Opacity' -> 'Airspace Opacity' in the script, (2) skips any label column
that is missing instead of crashing. Backup: e25_findings.py.bak_r48. Touches no data."""
import hashlib
import pathlib
import shutil

p = pathlib.Path.home() / "anchor/code/e25_findings.py"
s = p.read_text()
bak = p.with_name("e25_findings.py.bak_r48")
if not bak.exists():
    shutil.copy(p, bak)
n = s.count('"Lung Opacity"') + s.count("'Lung Opacity'")
s = s.replace('"Lung Opacity"', '"Airspace Opacity"').replace("'Lung Opacity'", "'Airspace Opacity'")
anchor = "            y = lab.loc[kk, L]; m = y.isin([0, 1]).to_numpy(); yy = (y.to_numpy() == 1)[m]\n"
guard = ("            if L not in lab.columns:\n"
         "                print(f\"  {L:<18} column missing in label file, skipped\"); continue\n")
if guard not in s:
    assert anchor in s, "anchor line not found: paste sed -n 100,110p e25_findings.py"
    s = s.replace(anchor, guard + anchor, 1)
p.write_text(s)
print(f"renamed {n} occurrence(s) | guard in place | md5 {hashlib.md5(s.encode()).hexdigest()}")
if n == 0:
    print("WARNING: no 'Lung Opacity' string found; paste: grep -n 'RAD' ~/anchor/code/e25_findings.py")
