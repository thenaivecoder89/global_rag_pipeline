# Synthetic POC Dataset - Project Helios

All data in this folder is synthetic and for proof-of-concept use only.

## Intended POC workflows
1. Completeness and gap analysis against the configured checklist.
2. Financial metric extraction and cross-document inconsistency detection.
3. Matching against the synthetic historical IC Q&A library.
4. K-means peer cost-profile clustering plus separate duplicate/outlier rules.

## Deliberately planted issues
- Total project cost differs between deck, memorandum and workbook.
- Equity IRR, COD, capacity factor, PPA price and annual generation differ across documents.
- Independent merchant curve and offtaker credit evidence are missing.
- Combined downside and BESS augmentation sensitivity are missing from the submitted pack.
- Cybersecurity and decommissioning funding are not addressed.
- Two pairs of advisory cost lines potentially overlap; corporate overhead, grid cost and contingency are high.

Use the files in `06_Synthetic_POC_Ground_Truth` to validate system outputs during development.
