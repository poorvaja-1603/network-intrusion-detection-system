# NSL-KDD Dataset

## Source
University of New Brunswick — Canadian Institute for Cybersecurity
https://www.unb.ca/cic/datasets/nsl.html

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

