import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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


def save_metrics(result, prefix):
    metrics_summary = {
        "task": result.get("task"),
        "pred_column": result.get("pred_column"),
        "gt_column": result.get("gt_column"),
        "accuracy": result.get("accuracy"),
        "macro_f1": result.get("macro_f1"),
        "weighted_f1": result.get("weighted_f1")
    }
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{prefix}_summary.csv", index=False)
    result["per_class"].to_csv(f"{prefix}_per_class.csv")
    result["confusion_matrix"].to_csv(f"{prefix}_confusion_matrix.csv")
    return summary_df


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
    parser.add_argument("--output-prefix", default="metrics", help="Prefix for saved metric CSV files")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.task == "sentiment":
        result = evaluate_sentiment(df, args.pred, args.gt)
    elif args.task == "mapped_emotions":
        result = evaluate_mapped_emotions(df, args.pred, args.gt)
    else:
        result = evaluate_fine_grained_emotions(df, args.pred, args.gt)

    print_metrics(result)
    save_metrics(result, args.output_prefix)
