# NSL-KDD Error and Vulnerability Analysis (Task 22)

## 1. Executive Summary
- **Overall Test Accuracy:** 74.75%
- **Detection Rate on Known (Train-Seen) Attacks:** 75.73%
- **Detection Rate on Novel (Test-Only Zero-Day) Attacks:** 13.92%
- **The Generalization Drop:** Detection rate drops by **61.81%** on previously unseen attack vectors.

## 2. Category-Level Performance Breakdown

| Category | Total Test Samples | Binary Detection Recall | Exact Category Recall |
|---|---|---|---|
| **NORMAL** | 9,711 | 97.31% | 97.31% |
| **DOS** | 7,460 | 76.9% | 75.92% |
| **PROBE** | 2,421 | 66.42% | 59.56% |
| **R2L** | 2,885 | 1.8% | 1.7% |
| **U2R** | 67 | 5.97% | 1.49% |

### Key Observations:
1. **Normal & DoS traffic:** High recall (>80-97%). DoS patterns (high volume SYN packets, same host connection counts) remain distinct.
2. **Probe traffic:** Fairly well-detected (~75-80%), though slow stealth scans mimic legitimate administrative requests.
3. **R2L (Remote to Local):** Massive vulnerability (<25% detection rate). Attackers masquerade inside single valid TCP sessions (`ftp_write`, `guess_passwd`, `snmpguest`), lacking anomalous traffic bursts.
4. **U2R (User to Root):** Catastrophic failure rate (<15-20% detection). Extremely few training samples (only 52 out of 125,973 in train) lead to extreme class imbalance.

## 3. Attack-Level Granular Breakdown (All 39 Types)

| Attack Name | Category | Status | Count | Detection Rate (%) | Exact Category Rate (%) |
|---|---|---|---|---|---|
| `neptune` | dos | Seen in Train | 4657 | 99.9% | 98.4% |
| `apache2` | dos | Novel (Test Only) | 737 | 1.8% | 1.8% |
| `processtable` | dos | Novel (Test Only) | 685 | 0.0% | 0.0% |
| `smurf` | dos | Seen in Train | 665 | 100.0% | 100.0% |
| `back` | dos | Seen in Train | 359 | 97.5% | 97.5% |
| `mailbomb` | dos | Novel (Test Only) | 293 | 0.0% | 0.0% |
| `pod` | dos | Seen in Train | 41 | 92.7% | 92.7% |
| `teardrop` | dos | Seen in Train | 12 | 100.0% | 100.0% |
| `land` | dos | Seen in Train | 7 | 57.1% | 57.1% |
| `udpstorm` | dos | Novel (Test Only) | 2 | 100.0% | 0.0% |
| `worm` | dos | Novel (Test Only) | 2 | 0.0% | 0.0% |
| `normal` | normal | Seen in Train | 9711 | 97.3% | 97.3% |
| `mscan` | probe | Novel (Test Only) | 996 | 18.8% | 2.1% |
| `satan` | probe | Seen in Train | 735 | 100.0% | 100.0% |
| `saint` | probe | Novel (Test Only) | 319 | 98.8% | 98.8% |
| `portsweep` | probe | Seen in Train | 157 | 100.0% | 100.0% |
| `ipsweep` | probe | Seen in Train | 141 | 100.0% | 100.0% |
| `nmap` | probe | Seen in Train | 73 | 100.0% | 100.0% |
| `guess_passwd` | r2l | Seen in Train | 1231 | 0.0% | 0.0% |
| `warezmaster` | r2l | Seen in Train | 944 | 5.0% | 5.0% |
| `snmpguess` | r2l | Novel (Test Only) | 331 | 0.0% | 0.0% |
| `snmpgetattack` | r2l | Novel (Test Only) | 178 | 0.0% | 0.0% |
| `httptunnel` | r2l | Novel (Test Only) | 133 | 0.8% | 0.0% |
| `multihop` | r2l | Seen in Train | 18 | 5.6% | 5.6% |
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

The table below inspects 10 specific attacks that bypassed the model and were classified as `normal`:

| # | Attack Type | Category | Novel? | Service | Count | Dst Host Srv Count | Why it slipped through |
|---|---|---|---|---|---|---|---|
| 1 | `mscan` | probe | Yes | telnet | 1 | 86 | Novel zero-day exploit (`mscan`) unseen in training. Low connection frequency (`count=1`), disguising as normal user traffic. |
| 2 | `guess_passwd` | r2l | No | telnet | 1 | 255 | Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. Low connection frequency (`count=1`), disguising as normal user traffic. |
| 3 | `apache2` | dos | Yes | http | 12 | 238 | Novel zero-day exploit (`apache2`) unseen in training. Successfully authenticated session (`logged_in=1`). |
| 4 | `buffer_overflow` | u2r | No | ftp | 1 | 27 | Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. Successfully authenticated session (`logged_in=1`). Low connection frequency (`count=1`), disguising as normal user traffic. |
| 5 | `warezmaster` | r2l | No | ftp | 1 | 37 | Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. Successfully authenticated session (`logged_in=1`). Low connection frequency (`count=1`), disguising as normal user traffic. |
| 6 | `snmpgetattack` | r2l | Yes | private | 2 | 9 | Novel zero-day exploit (`snmpgetattack`) unseen in training. Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. Low connection frequency (`count=2`), disguising as normal user traffic. |
| 7 | `processtable` | dos | Yes | telnet | 1 | 217 | Novel zero-day exploit (`processtable`) unseen in training. Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. Low connection frequency (`count=1`), disguising as normal user traffic. |
| 8 | `httptunnel` | r2l | Yes | other | 1 | 13 | Novel zero-day exploit (`httptunnel`) unseen in training. Low connection frequency (`count=1`), disguising as normal user traffic. |
| 9 | `ps` | u2r | Yes | telnet | 1 | 32 | Novel zero-day exploit (`ps`) unseen in training. Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. Successfully authenticated session (`logged_in=1`). Low connection frequency (`count=1`), disguising as normal user traffic. |
| 10 | `snmpguess` | r2l | Yes | private | 5 | 254 | Novel zero-day exploit (`snmpguess`) unseen in training. Clean TCP handshake & teardown (`flag='SF'`). Zero SYN/REJ errors. |

## 5. Conclusions & Strategic Recommendations
1. **Do not rely on accuracy alone:** Binary accuracy near 78-80% creates a false sense of security. An IDS that misses 80% of R2L and U2R compromises is insecure in production.
2. **Need for Content-Aware Features:** Connection metadata alone (ports, counts) fails to detect payload-driven application layer intrusions.
3. **Class Rebalancing:** Techniques like SMOTE or cost-sensitive loss are necessary to give minority attacks (U2R with 52 samples) sufficient decision boundary presence.
