from datasets import load_dataset
import pandas as pd
import os
import kagglehub

def download_goemotions(save_path="data/raw/goemotions/goemotions.csv"):
    """
    ES: Descarga el dataset GoEmotions y lo guarda como un archivo CSV. Si el archivo ya existe, lo carga desde el disco.
    EN: Downloads the GoEmotions dataset and saves it as a CSV file. If the file already exists, it loads it from disk.
    Args:
        save_path: Ruta donde se guardará el archivo CSV del dataset GoEmotions
    Returns:        
        DataFrame con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """

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

def download_kaggle_emotions(save_path="data/raw/kaggle/kaggle_dataset.csv"):
    """
    ES: Descarga el dataset Kaggle Emotions y lo guarda como un archivo CSV. Si el archivo ya existe, lo carga desde el disco.
    EN: Downloads the Kaggle Emotions dataset and saves it as a CSV file. If the file already exists, it loads it from disk.
    Args:
        save_path: Ruta donde se guardará el archivo CSV del dataset Kaggle Emotions
    Returns:        
        DataFrame con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """
    if os.path.exists(save_path):
        print("Kaggle Emotions ya existe. Cargando dataset...")
        return pd.read_csv(save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")
    print("Path to dataset files:", save_path)


def download_isear_emotions(save_path="data/raw/isear/isear_dataset.csv"):
    """
    ES: Descarga el dataset ISEAR Emotions y lo guarda como un archivo CSV. Si el archivo ya existe, lo carga desde el disco.
    EN: Downloads the ISEAR Emotions dataset and saves it as a CSV file. If the file already exists, it loads it from disk.
    Args:
        save_path: Ruta donde se guardará el archivo CSV del dataset ISEAR Emotions
    Returns:
        DataFrame con las columnas "id", "text", "emotion_gt" (la emoción original) y "emotion_gt_mapped" (la emoción mapeada)
    """
    if os.path.exists(save_path):
        print("ISEAR Emotions ya existe. Cargando dataset...")
        return pd.read_csv(save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path = kagglehub.dataset_download("faisalsanto007/isear-dataset")
    print("Path to dataset files:", save_path)


def download_datasets():
    """
    ES: Descarga los tres datasets (GoEmotions, Kaggle Emotions e ISEAR Emotions) y los guarda como archivos CSV. Si los archivos ya existen, los carga desde el disco.
    EN: Downloads the three datasets (GoEmotions, Kaggle Emotions, and ISEAR Emotions) and saves them as CSV files. If the files already exist, it loads them from disk.
    """
    goemotions_df = download_goemotions()
    kaggle_emotions_df = download_kaggle_emotions()
    isear_emotions_df = download_isear_emotions()
    return goemotions_df, kaggle_emotions_df, isear_emotions_df
