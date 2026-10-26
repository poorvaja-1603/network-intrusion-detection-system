# NSL-KDD Error and Vulnerability Analysis (Task 22)

## 1. Executive Summary
- **Model:** 100-tree entropy Random Forest trained for the five attack categories (seed 42).
- **Test split note:** This run uses official KDDTest+ for comparability. The overlap audit found 664 test rows with exact 41-feature matches in KDDTrain+; the separate overlap-clean file removes those rows.
- **Five-category accuracy:** 74.21%
- **Binary attack/normal accuracy:** 75.31%
- **Detection Rate on Known (Train-Seen) Attacks:** 75.69%
- **Detection Rate on Novel (Test-Only Zero-Day) Attacks:** 17.41%
- **The Generalization Drop:** Detection rate drops by **58.28%** on previously unseen attack vectors.

## 2. Category-Level Performance Breakdown

| Category | Total Test Samples | Binary Detection Recall | Exact Category Recall |
|---|---|---|---|
| **NORMAL** | 9,711 | 97.32% | 97.32% |
| **DOS** | 7,460 | 78.62% | 77.57% |
| **PROBE** | 2,421 | 66.42% | 59.64% |
| **R2L** | 2,885 | 1.77% | 1.63% |
| **U2R** | 67 | 5.97% | 1.49% |

### Observations:
The table reports measured recall for this run. Low recall in a category is a model limitation on this fixed benchmark split; the small U2R training count (52 rows) is one plausible contributing factor, not a demonstrated cause.

## 3. Attack-Level Granular Breakdown (All 39 Types)

| Attack Name | Category | Status | Count | Detection Rate (%) | Exact Category Rate (%) |
|---|---|---|---|---|---|
| `neptune` | dos | Seen in Train | 4657 | 99.9% | 98.3% |
| `apache2` | dos | Novel (Test Only) | 737 | 19.4% | 19.4% |
| `processtable` | dos | Novel (Test Only) | 685 | 0.0% | 0.0% |
| `smurf` | dos | Seen in Train | 665 | 100.0% | 100.0% |
| `back` | dos | Seen in Train | 359 | 97.5% | 97.5% |
| `mailbomb` | dos | Novel (Test Only) | 293 | 0.0% | 0.0% |
| `pod` | dos | Seen in Train | 41 | 92.7% | 92.7% |
| `teardrop` | dos | Seen in Train | 12 | 100.0% | 100.0% |
| `land` | dos | Seen in Train | 7 | 28.6% | 28.6% |
| `udpstorm` | dos | Novel (Test Only) | 2 | 100.0% | 0.0% |
| `worm` | dos | Novel (Test Only) | 2 | 0.0% | 0.0% |
| `normal` | normal | Seen in Train | 9711 | 97.3% | 97.3% |
| `mscan` | probe | Novel (Test Only) | 996 | 18.8% | 2.3% |
| `satan` | probe | Seen in Train | 735 | 100.0% | 100.0% |
| `saint` | probe | Novel (Test Only) | 319 | 98.8% | 98.8% |
| `portsweep` | probe | Seen in Train | 157 | 100.0% | 100.0% |
| `ipsweep` | probe | Seen in Train | 141 | 100.0% | 100.0% |
| `nmap` | probe | Seen in Train | 73 | 100.0% | 100.0% |
| `guess_passwd` | r2l | Seen in Train | 1231 | 0.0% | 0.0% |
| `warezmaster` | r2l | Seen in Train | 944 | 4.9% | 4.9% |
| `snmpguess` | r2l | Novel (Test Only) | 331 | 0.0% | 0.0% |
| `snmpgetattack` | r2l | Novel (Test Only) | 178 | 0.0% | 0.0% |
| `httptunnel` | r2l | Novel (Test Only) | 133 | 1.5% | 0.0% |
| `multihop` | r2l | Seen in Train | 18 | 0.0% | 0.0% |
| `named` | r2l | Novel (Test Only) | 17 | 5.9% | 0.0% |
| `sendmail` | r2l | Novel (Test Only) | 14 | 0.0% | 0.0% |
| `xlock` | r2l | Novel (Test Only) | 9 | 0.0% | 0.0% |
| `xsnoop` | r2l | Novel (Test Only) | 4 | 25.0% | 0.0% |
| `ftp_write` | r2l | Seen in Train | 3 | 0.0% | 0.0% |
| `phf` | r2l | Seen in Train | 2 | 50.0% | 50.0% |
| `imap` | r2l | Seen in Train | 1 | 0.0% | 0.0% |
| `buffer_overflow` | u2r | Seen in Train | 20 | 10.0% | 0.0% |
| `ps` | u2r | Novel (Test Only) | 15 | 6.7% | 0.0% |
| `rootkit` | u2r | Seen in Train | 13 | 0.0% | 0.0% |
| `xterm` | u2r | Novel (Test Only) | 13 | 7.7% | 7.7% |
| `loadmodule` | u2r | Seen in Train | 2 | 0.0% | 0.0% |
| `perl` | u2r | Seen in Train | 2 | 0.0% | 0.0% |
| `sqlattack` | u2r | Novel (Test Only) | 2 | 0.0% | 0.0% |

## 4. Deep Dive: 10 Representative False Negatives

The table shows up to 10 false negatives from distinct attack types. The listed fields describe the examples and do not establish why the model made each prediction.

| # | Attack Type | Category | Test-only type? | Service | Count | Dst Host Srv Count | Observed attributes |
|---|---|---|---|---|---|---|---|
| 1 | `mscan` | probe | Yes | telnet | 1 | 86 | test-only attack type `mscan`; count=1 |
| 2 | `guess_passwd` | r2l | No | telnet | 1 | 255 | flag=SF; serror_rate=0 and rerror_rate=0; count=1 |
| 3 | `apache2` | dos | Yes | http | 12 | 238 | test-only attack type `apache2`; logged_in=1 |
| 4 | `buffer_overflow` | u2r | No | ftp | 1 | 27 | flag=SF; serror_rate=0 and rerror_rate=0; logged_in=1; count=1 |
| 5 | `warezmaster` | r2l | No | ftp | 1 | 37 | flag=SF; serror_rate=0 and rerror_rate=0; logged_in=1; count=1 |
| 6 | `snmpgetattack` | r2l | Yes | private | 2 | 9 | test-only attack type `snmpgetattack`; flag=SF; serror_rate=0 and rerror_rate=0; count=2 |
| 7 | `processtable` | dos | Yes | telnet | 1 | 217 | test-only attack type `processtable`; flag=SF; serror_rate=0 and rerror_rate=0; count=1 |
| 8 | `httptunnel` | r2l | Yes | other | 1 | 13 | test-only attack type `httptunnel`; count=1 |
| 9 | `ps` | u2r | Yes | telnet | 1 | 32 | test-only attack type `ps`; flag=SF; serror_rate=0 and rerror_rate=0; logged_in=1; count=1 |
| 10 | `snmpguess` | r2l | Yes | private | 5 | 254 | test-only attack type `snmpguess`; flag=SF; serror_rate=0 and rerror_rate=0 |

## 5. Conclusions & Strategic Recommendations
1. **Report class-level metrics:** Overall accuracy can hide weak recall on minority attack families, so operational evaluation should include per-class recall and false-positive cost.
2. **Treat this as benchmark evidence:** The inspected NSL-KDD connection fields do not include packet payloads, so these results do not establish performance on live traffic or payload-based attacks.
3. **Evaluate mitigations separately:** Class weighting, resampling, and additional features are candidate follow-up experiments; this report does not claim they will improve performance.
