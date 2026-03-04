from datasets import load_dataset
import pandas as pd
import os

def descargar_goemotions(save_path="data/raw/goemotions.csv"):
    dataset = load_dataset("go_emotions")
    df = pd.concat([
        pd.DataFrame(dataset["train"]),
        pd.DataFrame(dataset["validation"]),
        pd.DataFrame(dataset["test"])
    ])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def descargar_kaggle_emotions(path="data/raw/emotions.csv"):
    df = pd.read_csv(path)
    return df


def descargar_isear_emotions(path="data/raw/isear.csv"):
    df = pd.read_csv(path)
    return df

def descargar_datasets():
    goemotions_df = descargar_goemotions()
    kaggle_emotions_df = descargar_kaggle_emotions()
    isear_emotions_df = descargar_isear_emotions()
    return goemotions_df, kaggle_emotions_df, isear_emotions_df
    