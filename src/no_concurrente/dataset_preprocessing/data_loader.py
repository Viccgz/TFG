from pydoc import text

from datasets import load_dataset
import pandas as pd
import os
import kagglehub

def descargar_goemotions(save_path="data/raw/goemotions/goemotions.csv"):

    if os.path.exists(save_path):
        print("GoEmotions ya existe. Cargando dataset...")
        return pd.read_csv(save_path)
    
    dataset = load_dataset("go_emotions")
    df = pd.concat([
        pd.DataFrame(dataset["train"]),
        pd.DataFrame(dataset["validation"]),
        pd.DataFrame(dataset["test"])
    ])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def descargar_kaggle_emotions(save_path="data/raw/kaggle/kaggle_dataset.csv"):
    if os.path.exists(save_path):
        print("Kaggle Emotions ya existe. Cargando dataset...")
        return pd.read_csv(save_path)
    
    save_path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")
    print("Path to dataset files:", save_path)


def descargar_isear_emotions(save_path="data/raw/isear/isear_dataset.csv"):
    if os.path.exists(save_path):
        print("ISEAR Emotions ya existe. Cargando dataset...")
        return pd.read_csv(save_path)

    save_path = kagglehub.dataset_download("faisalsanto007/isear-dataset")
    print("Path to dataset files:", save_path)


def descargar_datasets():
    goemotions_df = descargar_goemotions()
    kaggle_emotions_df = descargar_kaggle_emotions()
    isear_emotions_df = descargar_isear_emotions()
    return goemotions_df, kaggle_emotions_df, isear_emotions_df

if __name__ == "__main__":
    goemotions_dataset, kaggle_emotions_dataset, isear_emotions_dataset = descargar_datasets()