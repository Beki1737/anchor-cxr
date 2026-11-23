# Reader guide: chest X-ray report error study

Thank you for helping. Expected time: about 4 to 5 hours, can be split over several sessions.

## Before you start
- You must have accepted the ReXGradient-160K data use terms. Do not copy, photograph or share any image or text outside this computer.
- Open `reader_sheet.csv` (Excel or LibreOffice) and the `images` folder side by side.

## What you see per item
- `image`: the frontal chest X-ray.
- `reference_findings`: the FINDINGS section written by the original radiologist. Treat it as the truth, but use the image when the reference is ambiguous.
- `report_A`, `report_B`: two computer-generated reports. Their order is random. You are not told which system wrote which.

## What to fill in (for report A and report B separately)
Count errors compared with the reference (and the image). Enter whole numbers, 0 if none.

| Column | Count one for each... |
|---|---|
| `_false_significant` | finding stated in the report that is not present, AND would change management or follow-up (e.g. pneumothorax, effusion, consolidation, mass/nodule, fracture, edema, misplaced line) |
| `_missed_significant` | clinically significant finding in the reference that the report does not mention |
| `_wrong_side` | finding that is correct but placed on the wrong side or wrong location |
| `_false_minor` | finding stated but not present, with no impact on management (e.g. mild degenerative change, tiny granuloma) |
| `_missed_minor` | minor finding in the reference not mentioned |

Then:
- `preferred`: A, B, or equal: which report you would rather sign after editing.
- `confidence_1to5`: how confident you are in your grading of this item (1 = guess, 5 = certain).
- `comment`: optional, short.

## Rules
- Wording and style do not count as errors. Only content.
- A normal statement ("no pleural effusion") that is correct is not an error. A normal statement that contradicts the reference IS a missed finding.
- If the reference itself looks wrong compared with the image, write "reference doubtful" in `comment` and grade against what you see.
- Do not look up items twice and do not change earlier grades after finishing a session.

## Second task (short): extractor check
`e1_rater2.csv` lists single report sentences and what our software extracted from them. For each row fill:
- `finding_and_polarity_ok`: 1 if every finding and its present/absent status is right, else 0
- `regions_ok`: 1 if the anatomical regions are right, else 0
- `laterality_ok`: 1 if right/left/bilateral is right, else 0
- `missed_finding`: 1 if the sentence mentions a finding the extraction missed, else 0
