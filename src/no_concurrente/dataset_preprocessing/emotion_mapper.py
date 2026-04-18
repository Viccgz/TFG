import pandas as pd
import ast

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
    emotions = []
    ids_list = []
    texts = []

    for index, row in df.iterrows():
        ids = ast.literal_eval(row["labels"])
        emotion = GO_EMOTIONS_LABELS[ids[0]]
        mapped_emotion = EMOTION_MAP.get(emotion, "neutral")
        emotions.append(mapped_emotion)
        ids_list.append(row["id"])
        texts.append(row["text"])

    return pd.DataFrame({ "id": ids_list, "text": texts, "emotion": emotions })

def normalize_isear_labels(df):
    return pd.DataFrame({
        "id": df["ID"],
        "text": df["content"],
        "emotion": df["sentiment"].apply(lambda x: EMOTION_MAP.get(x, "neutral"))
    })

def normalize_kaggle_emotions_labels(df):
    id = 0
    texts = []    
    emotions = []
    for index, row in df.iterrows():
        text = row["text"]
        emotion = row["emotion"]
        mapped_emotion = EMOTION_MAP.get(emotion, "neutral")
        texts.append(text)
        emotions.append(mapped_emotion)
        id += 1
    return pd.DataFrame({"id": range(id), "text": texts, "emotion": emotions})

def normalize_datasets(goemotions_df, kaggle_emotions_df, isear_emotions_df):
    goemotions_normalized = normalize_go_emotions_labels(goemotions_df)
    kaggle_emotions_normalized = normalize_kaggle_emotions_labels(kaggle_emotions_df)
    isear_emotions_normalized = normalize_isear_labels(isear_emotions_df)

    return goemotions_normalized, kaggle_emotions_normalized, isear_emotions_normalized

if __name__ == "__main__":
    goemotions_df = pd.read_csv("data/raw/goemotions/goemotions.csv")
    kaggle_emotions_df = pd.read_csv("data/raw/kaggle/kaggle_dataset.csv")
    isear_emotions_df = pd.read_csv("data/raw/isear/isear_dataset.csv")

    goemotions_normalized, kaggle_emotions_normalized, isear_emotions_normalized = normalize_datasets(
        goemotions_df, kaggle_emotions_df, isear_emotions_df
    )

    goemotions_normalized.to_csv("data/processed/goemotions_normalized.csv", index=False)
    kaggle_emotions_normalized.to_csv("data/processed/kaggle_emotions_normalized.csv", index=False)
    isear_emotions_normalized.to_csv("data/processed/isear_emotions_normalized.csv", index=False)