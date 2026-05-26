import pandas as pd
import ast
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from concurrente.utils import sanitize_text
from concurrente.dataset_preprocessing.data_loader import descargar_datasets

GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral"
]

EMOTION_MAP = {
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "pride": "joy",
    "joy": "joy",
    "optimism": "joy",
    "relief": "joy",

    "love": "love",
    "caring": "love",

    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",

    "fear": "fear",
    "nervousness": "fear",

    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",
    "shame": "sadness",
    "guilt": "sadness",

    "disgust": "disgust",

    "surprise": "surprise",
    "realization": "surprise",

    "confusion": "neutral",
    "curiosity": "neutral",
    "neutral": "neutral",
    "desire": "neutral"
}
def normalize_emotion_label(emotion_raw):
    """
    ES: Mapea una etiqueta de emoción raw a una emoción normalizada usando EMOTION_MAP
    EN: Maps a raw emotion label to a normalized emotion using EMOTION_MAP
    
    Args:
        emotion_raw: La emoción raw a mapear
    
    Returns:
        La emoción mapeada. Si no existe en el mapa, devuelve "neutral"
    """
    return EMOTION_MAP.get(emotion_raw.lower(), "neutral")

def normalize_go_emotions_labels(df):
    """
    ES: Normaliza las etiquetas de emoción del dataset GoEmotions usando EMOTION_MAP
    EN: Normalizes the emotion labels from the GoEmotions dataset using EMOTION_MAP
    Args:
        df: DataFrame con las columnas "id", "text" y "labels" (donde "labels" es una lista de IDs de emociones)
    Returns:
        DataFrame con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """

    emotion_gts = []
    emotion_gts_mapped = []
    ids_list = []
    texts = []

    for index, row in df.iterrows():
        ids = ast.literal_eval(row["labels"])
        emotion_gt = GO_EMOTIONS_LABELS[ids[0]]
        emotion_gt_mapped = EMOTION_MAP.get(emotion_gt, "neutral")
        emotion_gts.append(emotion_gt)
        emotion_gts_mapped.append(emotion_gt_mapped)
        ids_list.append(row["id"])
        texts.append(sanitize_text(row["text"]))

    return pd.DataFrame({
        "id": ids_list,
        "text": texts,
        "emotion_gt": emotion_gts,
        "emotion_gt_mapped": emotion_gts_mapped
    })

def normalize_isear_labels(df):
    """
    ES: Normaliza las etiquetas de emoción del dataset ISEAR usando EMOTION_MAP
    EN: Normalizes the emotion labels from the ISEAR dataset using EMOTION_MAP
    Args:
        df: DataFrame con las columnas "ID", "content" y "sentiment" (donde "sentiment" es la etiqueta de emoción original)
    Returns:         DataFrame con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """
    
    return pd.DataFrame({
        "id": df["ID"],
        "text": df["content"].apply(sanitize_text),
        "emotion_gt": df["sentiment"],
        "emotion_gt_mapped": df["sentiment"].apply(lambda x: EMOTION_MAP.get(x, "neutral"))
    })

def normalize_kaggle_emotions_labels(df):
    """
    ES: Normaliza las etiquetas de emoción del dataset Kaggle Emotions usando EMOTION_MAP
    EN: Normalizes the emotion labels from the Kaggle Emotions dataset using EMOTION_MAP
    Args:
        df: DataFrame con las columnas "id", "text" y "emotion" (donde "emotion" es la etiqueta de emoción original)
    Returns:         DataFrame con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """

    ids = []
    texts = []    
    emotion_gts = []
    emotion_gts_mapped = []
    for index, row in df.iterrows():
        ids.append(index)
        text = row["text"]
        emotion_gt = row["emotion"]
        emotion_gt_mapped = EMOTION_MAP.get(emotion_gt, "neutral")
        texts.append(sanitize_text(text))
        emotion_gts.append(emotion_gt)
        emotion_gts_mapped.append(emotion_gt_mapped)
    return pd.DataFrame({
        "id": ids,
        "text": texts,
        "emotion_gt": emotion_gts,
        "emotion_gt_mapped": emotion_gts_mapped
    })

def normalize_datasets(goemotions_df, kaggle_emotions_df, isear_emotions_df):
    """
    ES: Normaliza las etiquetas de emoción de los datasets GoEmotions, Kaggle Emotions e ISEAR usando EMOTION_MAP
    EN: Normalizes the emotion labels from the GoEmotions, Kaggle Emotions, and ISEAR datasets using EMOTION_MAP
    Args:
        goemotions_df: DataFrame del dataset GoEmotions con las columnas "id", "text" y "labels" (donde "labels" es una lista de IDs de emociones)
        kaggle_emotions_df: DataFrame del dataset Kaggle Emotions con las columnas "id", "text" y "emotion" (donde "emotion" es la etiqueta de emoción original)
        isear_emotions_df: DataFrame del dataset ISEAR con las columnas "ID", "content" y "sentiment" (donde "sentiment" es la etiqueta de emoción original)
    Returns:
         Tres DataFrames normalizados con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """
    goemotions_normalized = normalize_go_emotions_labels(goemotions_df)
    kaggle_emotions_normalized = normalize_kaggle_emotions_labels(kaggle_emotions_df)
    isear_emotions_normalized = normalize_isear_labels(isear_emotions_df)

    return goemotions_normalized, kaggle_emotions_normalized, isear_emotions_normalized

if __name__ == "__main__":
    goemotions_df, kaggle_emotions_df, isear_emotions_df = descargar_datasets()

    goemotions_normalized, kaggle_emotions_normalized, isear_emotions_normalized = normalize_datasets(
        goemotions_df, kaggle_emotions_df, isear_emotions_df
    )

    goemotions_normalized.to_csv("data/processed/goemotions_normalized.csv", index=False)
    kaggle_emotions_normalized.to_csv("data/processed/kaggle_emotions_normalized.csv", index=False)
    isear_emotions_normalized.to_csv("data/processed/isear_emotions_normalized.csv", index=False)