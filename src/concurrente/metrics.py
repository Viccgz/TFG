import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
MAPPED_EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "love", "neutral"]

SENTIMENT_MAPPING = {
    "+": "positive",
    "-": "negative",
    "=": "neutral",
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral"
}


def _normalize_label(value, mapping=None):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if mapping is not None:
        return mapping.get(value, value)
    return value


def normalize_sentiment_series(series):
    return series.apply(lambda x: _normalize_label(x, mapping=SENTIMENT_MAPPING))


def _clean_series(series):
    return series.astype(str).str.strip().str.lower().replace({"nan": np.nan})


def _filter_labels(y_true, y_pred, labels):
    mask = y_true.isin(labels) & y_pred.isin(labels)
    return y_true[mask], y_pred[mask]


def _classification_metrics(y_true, y_pred, labels):
    metrics = {}
    if len(y_true) == 0:
        for label in labels:
            metrics[label] = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        return {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "per_class": pd.DataFrame.from_dict(metrics, orient="index"),
            "confusion_matrix": pd.DataFrame(np.zeros((len(labels), len(labels)), dtype=int), index=labels, columns=labels)
        }

    accuracy = accuracy_score(y_true, y_pred)
    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    per_class = pd.DataFrame({
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1": f1_per_class
    }, index=labels)

    conf = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(conf, index=labels, columns=labels)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_df
    }


def evaluate_sentiment(df, pred_column, gt_column="emotion", labels=None):
    """Compute sentiment metrics for positive/negative/neutral classification."""
    if labels is None:
        labels = SENTIMENT_LABELS

    y_true = normalize_sentiment_series(df[gt_column])
    y_pred = normalize_sentiment_series(df[pred_column])
    y_true, y_pred = _filter_labels(y_true, y_pred, labels)

    result = _classification_metrics(y_true, y_pred, labels)
    result["task"] = "sentiment"
    result["pred_column"] = pred_column
    result["gt_column"] = gt_column
    return result


def evaluate_mapped_emotions(df, pred_column, gt_column="emotion_gt_mapped", labels=None):
    """Compute mapped-emotion metrics for the normalized emotion classes."""
    if labels is None:
        labels = MAPPED_EMOTION_LABELS

    y_true = _clean_series(df[gt_column])
    y_pred = _clean_series(df[pred_column])
    y_true, y_pred = _filter_labels(y_true, y_pred, labels)

    result = _classification_metrics(y_true, y_pred, labels)
    result["task"] = "mapped_emotions"
    result["pred_column"] = pred_column
    result["gt_column"] = gt_column
    return result


def evaluate_fine_grained_emotions(df, pred_column, gt_column, labels=None):
    """Compute fine-grained metrics for raw emotion labels, preferably GoEmotions."""
    y_true = _clean_series(df[gt_column])
    y_pred = _clean_series(df[pred_column])
    if labels is None:
        labels = sorted(set(y_true.dropna()).union(set(y_pred.dropna())))

    y_true, y_pred = _filter_labels(y_true, y_pred, labels)
    metrics = _classification_metrics(y_true, y_pred, labels)
    metrics["task"] = "fine_grained_emotions"
    metrics["pred_column"] = pred_column
    metrics["gt_column"] = gt_column
    return metrics


def _ensure_plot_backend():
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate plots. Install it with `pip install matplotlib`."
        )


def plot_per_class_metrics(result, prefix, show=False):
    _ensure_plot_backend()
    df = result["per_class"]
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_title(f"Per-class metrics for {result.get('task')}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    fig.tight_layout()
    filename = f"{prefix}_per_class.png"
    fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(result, prefix, show=False):
    _ensure_plot_backend()
    cm = result["confusion_matrix"]
    labels = list(cm.index)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm.values, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion matrix for {result.get('task')}")
    fig.colorbar(cax, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm.iat[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    filename = f"{prefix}_confusion_matrix.png"
    fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary_metrics(result, prefix, show=False):
    _ensure_plot_backend()
    summary = {
        "accuracy": result.get("accuracy"),
        "macro_f1": result.get("macro_f1"),
        "weighted_f1": result.get("weighted_f1")
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.keys(), summary.values(), color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Summary metrics for {result.get('task')}")
    ax.set_ylabel("Score")
    for i, value in enumerate(summary.values()):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    filename = f"{prefix}_summary.png"
    fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_plots(result, prefix, show=False):
    plot_summary_metrics(result, prefix, show=show)
    plot_per_class_metrics(result, prefix, show=show)
    plot_confusion_matrix(result, prefix, show=show)


def build_metrics_summary(result):
    return pd.DataFrame([{
        "task": result.get("task"),
        "pred_column": result.get("pred_column"),
        "gt_column": result.get("gt_column"),
        "accuracy": result.get("accuracy"),
        "macro_f1": result.get("macro_f1"),
        "weighted_f1": result.get("weighted_f1")
    }])


def print_metrics(result):
    print(f"Task: {result.get('task')}")
    print(f"Prediction column: {result.get('pred_column')}")
    print(f"Ground truth column: {result.get('gt_column')}")
    print(f"Accuracy: {result.get('accuracy'):.4f}")
    print(f"Macro F1: {result.get('macro_f1'):.4f}")
    print(f"Weighted F1: {result.get('weighted_f1'):.4f}")
    print("\nPer-class metrics:")
    print(result.get("per_class"))
    print("\nConfusion matrix:")
    print(result.get("confusion_matrix"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate classification metrics for sentiment or emotion results.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--pred", required=True, help="Prediction column name")
    parser.add_argument("--gt", required=True, help="Ground truth column name")
    parser.add_argument("--task", choices=["sentiment", "mapped_emotions", "fine_grained_emotions"], default="sentiment")
    parser.add_argument("--sep", default=None, help="CSV separator, e.g. ',' or ';' (auto-detect if omitted)")
    parser.add_argument("--output-dir", default=None, help="Directory where plots will be saved. Defaults to data/results/concurrente/metrics/<dataset_name>.")
    parser.add_argument("--output-prefix", default="metrics", help="Prefix for saved plot files")
    parser.add_argument("--show", action="store_true", help="Display the generated plots interactively after saving them")
    args = parser.parse_args()

    if args.sep:
        df = pd.read_csv(args.csv, sep=args.sep)
    else:
        try:
            df = pd.read_csv(args.csv)
        except pd.errors.ParserError:
            df = pd.read_csv(args.csv, sep=';')

    if args.task == "sentiment":
        result = evaluate_sentiment(df, args.pred, args.gt)
    elif args.task == "mapped_emotions":
        result = evaluate_mapped_emotions(df, args.pred, args.gt)
    else:
        result = evaluate_fine_grained_emotions(df, args.pred, args.gt)

    csv_path = Path(args.csv).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "results" / "concurrente" / "metrics" / csv_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_dir / args.output_prefix)

    save_plots(result, prefix, show=args.show)
    print(f"Saved plots in: {output_dir}")
