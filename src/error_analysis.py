"""
error_analysis.py — Task 22: Comprehensive Error Analysis on KDDTest+

Evaluates predictions across:
1. Per-attack-type recall for all 39 attack types (train-seen vs test-novel)
2. Confusion breakdown across the 5 categories (Normal, DoS, Probe, R2L, U2R)
3. Identification of hardest attack categories and zero-day attacks
4. Extraction of 10 representative false negatives with feature inspection

Generates:
  reports/error_analysis.md
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data import load_nsl_kdd, FEATURE_COLUMNS, ALL_COLUMNS, DATA_DIR
from src.preprocess import build_preprocessor, preprocess_data
from src.attack_map import ATTACK_MAP, TEST_ONLY_ATTACKS, TRAIN_ATTACKS

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def run_error_analysis():
    print("Running error analysis...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data with attack types
    raw_test = pd.read_csv(DATA_DIR / "KDDTest+.txt", header=None, names=ALL_COLUMNS)
    raw_test["attack_type"] = raw_test["attack_type"].str.strip()
    
    X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat = load_nsl_kdd()
    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)

    # Train a strong multiclass classifier (RandomForest) and J48
    rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42, n_jobs=-1)
    rf.fit(X_train_proc, y_train_cat)
    y_pred_cat = rf.predict(X_test_proc)
    y_proba = rf.predict_proba(X_test_proc)
    classes = list(rf.classes_)

    # Binary evaluation
    y_pred_bin = (y_pred_cat != "normal").astype(int)

    # 1. Per-attack type analysis
    raw_test["pred_cat"] = y_pred_cat
    raw_test["pred_bin"] = y_pred_bin
    raw_test["true_cat"] = y_test_cat
    raw_test["true_bin"] = y_test_bin
    raw_test["is_novel"] = raw_test["attack_type"].isin(TEST_ONLY_ATTACKS.keys())

    attack_stats = []
    for attack, grp in raw_test.groupby("attack_type"):
        total = len(grp)
        cat = ATTACK_MAP.get(attack, "unknown")
        is_novel = attack in TEST_ONLY_ATTACKS
        
        # If normal, recall is normal correctly predicted
        if attack == "normal":
            correct = (grp["pred_cat"] == "normal").sum()
            rec = correct / total
        else:
            # For attack, binary detection recall (detected as any attack)
            detected = (grp["pred_bin"] == 1).sum()
            rec = detected / total
            # Also exact category recall
            exact = (grp["pred_cat"] == cat).sum()
            rec_exact = exact / total

        attack_stats.append({
            "attack": attack,
            "category": cat,
            "total_samples": total,
            "is_novel": is_novel,
            "detected_as_attack_rate": round(rec, 4),
            "exact_category_rate": round(rec_exact if attack != "normal" else rec, 4),
        })

    df_attack_stats = pd.DataFrame(attack_stats).sort_values(by=["category", "total_samples"], ascending=[True, False])

    # 2. Category level breakdown
    cat_summary = []
    for cat in ["normal", "dos", "probe", "r2l", "u2r"]:
        grp = raw_test[raw_test["true_cat"] == cat]
        total = len(grp)
        exact = (grp["pred_cat"] == cat).sum()
        as_attack = (grp["pred_bin"] == 1).sum() if cat != "normal" else (grp["pred_bin"] == 0).sum()
        cat_summary.append({
            "Category": cat.upper(),
            "Total Samples": total,
            "Exact Category Recall (%)": round(exact / total * 100, 2),
            "Binary Detection Recall (%)": round(as_attack / total * 100, 2),
        })
    df_cat_summary = pd.DataFrame(cat_summary)

    # 3. Novel vs Seen Attack Detection
    seen_attacks = raw_test[(raw_test["true_bin"] == 1) & (~raw_test["is_novel"])]
    novel_attacks = raw_test[(raw_test["true_bin"] == 1) & (raw_test["is_novel"])]

    seen_recall = (seen_attacks["pred_bin"] == 1).mean() * 100
    novel_recall = (novel_attacks["pred_bin"] == 1).mean() * 100

    # 4. Extract 10 distinct, representative False Negatives across categories
    fn_df = raw_test[(raw_test["true_bin"] == 1) & (raw_test["pred_bin"] == 0)]
    # Pick diverse attacks: pick 1 sample from up to 10 distinct attack types
    selected_indices = []
    seen_types = set()
    for idx, row in fn_df.iterrows():
        atk = row["attack_type"]
        if atk not in seen_types:
            seen_types.add(atk)
            selected_indices.append(idx)
        if len(selected_indices) >= 10:
            break
    fn_examples = fn_df.loc[selected_indices]

    def describe_fn(r):
        """List observable features without claiming they caused the error."""
        observations = []
        if r["is_novel"]:
            observations.append(f"test-only attack type `{r['attack_type']}`")
        if r["flag"] == "SF":
            observations.append("flag=SF")
        if r["serror_rate"] == 0.0 and r["rerror_rate"] == 0.0:
            observations.append("serror_rate=0 and rerror_rate=0")
        if r["logged_in"] == 1:
            observations.append("logged_in=1")
        if r["count"] <= 2:
            observations.append(f"count={int(r['count'])}")
        if not observations:
            observations.append("no listed indicator among the inspected fields")
        return "; ".join(observations)

    # Write Markdown Report
    report_path = REPORTS_DIR / "error_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NSL-KDD Error and Vulnerability Analysis (Task 22)\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("- **Model:** 100-tree entropy Random Forest trained for the five attack categories (seed 42).\n")
        f.write("- **Test split note:** This run uses official KDDTest+ for comparability. The overlap audit found 664 test rows with exact 41-feature matches in KDDTrain+; the separate overlap-clean file removes those rows.\n")
        f.write(f"- **Five-category accuracy:** {(raw_test['pred_cat'] == raw_test['true_cat']).mean() * 100:.2f}%\n")
        f.write(f"- **Binary attack/normal accuracy:** {(raw_test['pred_bin'] == raw_test['true_bin']).mean() * 100:.2f}%\n")
        f.write(f"- **Detection Rate on Known (Train-Seen) Attacks:** {seen_recall:.2f}%\n")
        f.write(f"- **Detection Rate on Novel (Test-Only Zero-Day) Attacks:** {novel_recall:.2f}%\n")
        f.write(f"- **The Generalization Drop:** Detection rate drops by **{seen_recall - novel_recall:.2f}%** on previously unseen attack vectors.\n\n")

        f.write("## 2. Category-Level Performance Breakdown\n\n")
        f.write("| Category | Total Test Samples | Binary Detection Recall | Exact Category Recall |\n")
        f.write("|---|---|---|---|\n")
        for _, row in df_cat_summary.iterrows():
            f.write(f"| **{row['Category']}** | {row['Total Samples']:,} | {row['Binary Detection Recall (%)']}% | {row['Exact Category Recall (%)']}% |\n")
        f.write("\n")

        f.write("### Observations:\n")
        f.write("The table reports measured recall for this run. Low recall in a category is a model limitation on this fixed benchmark split; the small U2R training count (52 rows) is one plausible contributing factor, not a demonstrated cause.\n\n")

        f.write("## 3. Attack-Level Granular Breakdown (All 39 Types)\n\n")
        f.write("| Attack Name | Category | Status | Count | Detection Rate (%) | Exact Category Rate (%) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in df_attack_stats.iterrows():
            status = "Novel (Test Only)" if row["is_novel"] else "Seen in Train"
            f.write(f"| `{row['attack']}` | {row['category']} | {status} | {row['total_samples']} | {row['detected_as_attack_rate']*100:.1f}% | {row['exact_category_rate']*100:.1f}% |\n")
        f.write("\n")

        f.write("## 4. Deep Dive: 10 Representative False Negatives\n\n")
        f.write("The table shows up to 10 false negatives from distinct attack types. The listed fields describe the examples and do not establish why the model made each prediction.\n\n")
        f.write("| # | Attack Type | Category | Test-only type? | Service | Count | Dst Host Srv Count | Observed attributes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        for idx, (_, row) in enumerate(fn_examples.iterrows()):
            novel_str = "Yes" if row["is_novel"] else "No"
            observations = describe_fn(row)
            f.write(f"| {idx+1} | `{row['attack_type']}` | {row['true_cat']} | {novel_str} | {row['service']} | {int(row['count'])} | {int(row['dst_host_srv_count'])} | {observations} |\n")

        f.write("\n## 5. Conclusions & Strategic Recommendations\n")
        f.write("1. **Report class-level metrics:** Overall accuracy can hide weak recall on minority attack families, so operational evaluation should include per-class recall and false-positive cost.\n")
        f.write("2. **Treat this as benchmark evidence:** The inspected NSL-KDD connection fields do not include packet payloads, so these results do not establish performance on live traffic or payload-based attacks.\n")
        f.write("3. **Evaluate mitigations separately:** Class weighting, resampling, and additional features are candidate follow-up experiments; this report does not claim they will improve performance.\n")

    print(f"Error analysis report generated at: {report_path}")


if __name__ == "__main__":
    run_error_analysis()

