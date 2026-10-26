# NSL-KDD Dataset

## Source
University of New Brunswick — Canadian Institute for Cybersecurity
https://www.unb.ca/cic/datasets/nsl.html

The UNB page identifies the original dataset files but currently states that
the dataset is no longer available there. The files in this folder were
retrieved from the Zenodo mirror below; their MD5 checksums match the values
published with that record.

- Zenodo record: https://zenodo.org/records/3768048
- Dataset DOI: https://doi.org/10.5281/zenodo.3768048

## Required Files
Download and place these files in this directory:
- `KDDTrain+.txt` — 125,973 training instances
- `KDDTest+.txt`  — 22,544 test instances

## Format
- 43 columns: 41 features + attack label + difficulty level
- Comma-separated, no header row
- 3 categorical features: protocol_type, service, flag
- 38 numeric features

## Classes
| Class   | Training | Testing |
|---------|----------|---------|
| Normal  | 67,343   | 9,711   |
| Anomaly | 58,630   | 12,833  |

## Checksums
- `KDDTrain+.txt` — MD5 `f5592a95d1d1428348dfa6ca9652a800`
- `KDDTest+.txt` — MD5 `7c6d1b1af246690766394920d6b4c751`

## Citation
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009). “A Detailed
Analysis of the KDD CUP 99 Data Set.” Second IEEE Symposium on Computational
Intelligence for Security and Defense Applications (CISDA).

## Feature-overlap audit
The official KDDTest+ file is preserved unchanged for results comparable with
published NSL-KDD benchmarks. An audit found test rows whose 41 feature values
also occur in KDDTrain+. To support a leakage-reduced evaluation, run
`python -m src.deduplicate_splits`. This writes
`KDDTest+_no_train_feature_overlap.txt`, a separate 43-column test file with
all such test rows removed, and `split_overlap_audit.json` with the counts and
label breakdown. The cleaned split is an additional evaluation set; its scores
should not be compared directly with results on the official KDDTest+ file.

The Weka runner saves its prepared feature exports as gzip-compressed CSV files
under `data/weka_exports/` to keep each exported artifact below common hosting
limits. Open the `.csv.gz` files with a tool that supports gzip CSV input, or
decompress them before loading into software that requires plain CSV.

