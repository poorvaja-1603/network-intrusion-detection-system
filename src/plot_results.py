"""Task 30: generate report figures and a comparison table from results/*.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


def read_json(name: str):
    return json.loads((RESULTS / name).read_text())


def save_full_vs_reduced() -> None:
    full = read_json("exp1_full_features.json")
    reduced = read_json("exp7_kbest.json")
    model_names = ["NaiveBayes", "J48", "RandomForest"]
    settings = [("Full", "full", None), ("PCA 5", "pca", 5),
                ("Gaussian RP 5", "rp_gaussian", 5), ("K-best 5", "kbest", 5)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, model_name in zip(axes, model_names):
        values = []
        for _, setting, dims in settings:
            source = full if setting == "full" else reduced
            match = next(
                row for row in source
                if row["model"] == model_name and row["setting"] == setting
                and (dims is None or row["n_dims"] == dims)
            )
            values.append(match["accuracy"])
        bars = ax.bar([s[0] for s in settings], values, color=["#355C7D", "#6C9A8B", "#F2A65A", "#B56576"])
        ax.set_title(model_name)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    fig.suptitle("Full features vs 5-dimensional representations")
    fig.tight_layout()
    fig.savefig(FIGURES / "full_vs_reduced.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrices() -> None:
    full = read_json("exp1_full_features.json")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, row in zip(axes.flat, full):
        cm = np.asarray(row["confusion_matrix"], dtype=int)
        image = ax.imshow(cm, cmap="Blues")
        for (row_index, col_index), value in np.ndenumerate(cm):
            ax.text(col_index, row_index, f"{value:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.set_title(row["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    axes.flat[-1].axis("off")
    fig.suptitle("Full-feature binary confusion matrices (KDDTest+)")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices_full.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_dimensions() -> None:
    rp = read_json("exp3_rp_gaussian_sweep.json")
    kbest = read_json("exp7_kbest.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for model_name, color in zip(
        ["NaiveBayes", "J48", "RandomForest"], ["#355C7D", "#C06C84", "#6C9A8B"]
    ):
        rows = sorted((r for r in rp if r["model"] == model_name), key=lambda r: r["n_dims"])
        axes[0].plot([r["n_dims"] for r in rows], [r["accuracy"] for r in rows],
                     marker="o", label=model_name, color=color)
    axes[0].set_title("Gaussian RP sweep")
    axes[0].set_xlabel("Projected dimensions")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(sorted(set(r["n_dims"] for r in rp)))
    axes[0].legend()

    model_name = "J48"
    for setting, label, color in [
        ("pca", "PCA", "#6C9A8B"),
        ("rp_gaussian", "Gaussian RP", "#F2A65A"),
        ("kbest", "Mutual-information K-best", "#B56576"),
    ]:
        rows = sorted(
            (r for r in kbest if r["model"] == model_name and r["setting"] == setting),
            key=lambda r: r["n_dims"],
        )
        axes[1].plot([r["n_dims"] for r in rows], [r["accuracy"] for r in rows],
                     marker="o", label=label, color=color)
    axes[1].set_title("J48 representation comparison")
    axes[1].set_xlabel("Dimensions / selected features")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks([5, 10, 20])
    axes[1].legend(fontsize=8)
    fig.suptitle("Accuracy by representation size")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_dimensions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _paper_and_ours() -> tuple[dict, dict]:
    paper_data = read_json("paper_reported.json")
    ours_full = read_json("exp1_full_features.json")
    ours_rp = read_json("exp3_rp_gaussian_sweep.json")
    weka = read_json("weka_results.json")
    paper = {}
    for setting in ("exp1_full", "exp3_rp_gaussian"):
        experiment = next(row for row in paper_data if row["experiment"] == setting)
        paper[setting] = {row["model"]: row for row in experiment["results"]}
    ours = {"exp1_full": {r["model"]: r for r in ours_full}}
    ours["exp1_full"].update({r["model"]: r for r in weka if r["setting"] == "full"})
    ours["exp3_rp_gaussian"] = {
        r["model"]: r for r in ours_rp if r["n_dims"] == 5
    }
    ours["exp3_rp_gaussian"].update(
        {r["model"]: r for r in weka if r["setting"] == "rp_gaussian"}
    )
    return paper, ours


def save_paper_comparison() -> None:
    paper, ours = _paper_and_ours()
    models = ["BayesNet", "NaiveBayes", "J48", "PART", "RandomForest"]
    settings = [("exp1_full", "Full features"), ("exp3_rp_gaussian", "Gaussian RP (5)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
    for col, (setting, title) in enumerate(settings):
        for row, metric in enumerate(["accuracy", "fpr"]):
            pvals = [paper[setting][model][metric] for model in models]
            ovals = [ours[setting][model][metric] for model in models]
            x = np.arange(len(models))
            width = 0.36
            axes[row, col].bar(x - width / 2, pvals, width, label="Paper", color="#355C7D")
            axes[row, col].bar(x + width / 2, ovals, width, label="This project", color="#F2A65A")
            axes[row, col].set_xticks(x, models, rotation=20)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].set_title(f"{title}: {metric.upper() if metric == 'fpr' else 'Accuracy'}")
            axes[row, col].set_ylabel(metric.upper() if metric == "fpr" else "Accuracy")
            axes[row, col].legend(fontsize=8)
    fig.suptitle("Paper results vs this project's results")
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_vs_ours.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Paper vs project comparison", "",
        "The paper results are from the transcribed tables. This project uses Weka for PART/BayesNet and scikit-learn for the other models; see the report for implementation differences.",
        "", "| Setting | Model | Paper accuracy | Project accuracy | Paper FPR | Project FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for setting, label in settings:
        for model in models:
            p, o = paper[setting][model], ours[setting][model]
            lines.append(
                f"| {label} | {model} | {p['accuracy']:.4f} | {o['accuracy']:.4f} | {p['fpr']:.4f} | {o['fpr']:.4f} |"
            )
    (FIGURES / "paper_vs_ours.md").write_text("\n".join(lines) + "\n")


def save_timing() -> None:
    data = read_json("timing.json")["summary"]
    settings = list(data)
    means = [data[s]["total_fit"]["mean_seconds"] for s in settings]
    stds = [data[s]["total_fit"]["std_seconds"] for s in settings]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(settings, means, xerr=stds, capsize=4,
                   color=["#355C7D", "#6C9A8B", "#F2A65A", "#B56576"])
    ax.set_xlabel("Mean fit time (seconds; ±1 SD)")
    ax.set_title("Five-run fit-time benchmark (J48)")
    ax.bar_label(bars, fmt="%.2f s", padding=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "timing_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_full_vs_reduced()
    save_confusion_matrices()
    save_accuracy_dimensions()
    save_paper_comparison()
    save_timing()
    print(f"Generated figures and tables in {FIGURES}")


if __name__ == "__main__":
    main()
