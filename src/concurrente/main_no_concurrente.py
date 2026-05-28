import os
import time as timer
import pandas as pd
from pymongo import MongoClient
from dataset_preprocessing.emotion_mapper import normalize_emotion_label
import sentiment_analysis.llm_call as llm_call
import utils
import sentiment_analysis.libraries_call as libraries_call
import chardet

def process_csv(input_csv, output_csv, llm_chosen, justification, dataset):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se guarda en otro equipo
                                                  # EN: Change to the PC's which has the database IP if saving in another computer
    DATABASE_NAME = 'TFG_Results_EmotionalAnalysis'
    COLLECTION_NAME = dataset + '_' + llm_chosen + '_results_no_concurrente'
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    with open(input_csv, "rb") as f:
        raw = f.read()
        result = chardet.detect(raw)
    detected_encoding = result.get('encoding')
    
    if not detected_encoding:
        detected_encoding = 'utf-8'

    try:
        df = pd.read_csv(input_csv, sep=None, engine='python', encoding=detected_encoding)
    except UnicodeDecodeError as err:
        fallback_encodings = [enc for enc in ["latin-1", "cp1252", "utf-8"] if enc != detected_encoding]
        df = None
        for enc in fallback_encodings:
            try:
                df = pd.read_csv(input_csv, sep=None, engine='python', encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            df = pd.read_csv(input_csv, sep=None, engine='python', encoding='utf-8', errors='replace')

    string_columns = [
        "sentiment_" + llm_chosen,
        "justification_sentiment_" + llm_chosen,
        "certainty_sentiment_" + llm_chosen,
        "emotion_" + llm_chosen,
        "justification_emotion_" + llm_chosen,
        "certainty_emotion_" + llm_chosen,
        "processing_date_sentiment",
        "processing_hour_sentiment",
        "processing_date_emotion",
        "processing_hour_emotion"
    ]
    float_columns = [
        "total_process_time_seconds_" + llm_chosen
    ]
    for column in string_columns:
        if column not in df.columns:
            df[column] = pd.Series([None] * len(df), dtype="string")
        elif df[column].dtype not in ["object", "string"]:
            df[column] = df[column].astype("string")
    for column in float_columns:
        if column not in df.columns:
            df[column] = pd.Series([pd.NA] * len(df), dtype="Float64")
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    print("llm_chosen: ", llm_chosen)
    for index, row in df.iterrows():
        id = row["id"]
        message = row["text"]
        emotion_raw_gt = row["emotion_gt"]
        emotion_mapped_gt = row["emotion_gt_mapped"] 

        # ES:  procesar si los campos son nulos
        # EN:  process if fields are null
        if pd.isnull(row["sentiment_" + llm_chosen]) or pd.isnull(row["certainty_sentiment_" + llm_chosen]) or pd.isnull(row["justification_sentiment_" + llm_chosen]):
            function = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
            sentiment, certainty, justification_sentiment, date, time = function(message, justification, "sentiment_analysis", dataset, id)

            # ES: Actualizar el dataframe creando una nueva columna
            # EN: Update the dataframe creating a new column
            df.at[index, "sentiment_" + llm_chosen] = sentiment
            df.at[index, "certainty_sentiment_" + llm_chosen] = certainty
            df.at[index, "justification_sentiment_" + llm_chosen] = justification_sentiment
            df.at[index, "processing_date_sentiment"] = date
            df.at[index, "processing_hour_sentiment"] = time

            #ES: Realizar análisis de emociones con LLMs para cada mensaje, y guardar el resultado en el dataframe
            #EN: Perform emotion analysis with LLMs for each message, and save the result
            func = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
            emotion_raw_llm, emotion_mapped, certainty_emotion, justification_emotion, date_emotion, time_emotion = func(message, justification, "emotion_analysis", dataset, id)
            
            df.at[index, "emotion_" + llm_chosen] = emotion_mapped
            df.at[index, "certainty_emotion_" + llm_chosen] = certainty_emotion
            df.at[index, "justification_emotion_" + llm_chosen] = justification_emotion
            df.at[index, "processing_date_emotion"] = date_emotion
            df.at[index, "processing_hour_emotion"] = time_emotion

            # ES: Guardar en MongoDB
            # EN: Save in MongoDB
            utils.save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification_sentiment, date, time, emotion_mapped, certainty_emotion, justification_emotion, date_emotion, time_emotion, emotion_raw_llm, id, emotion_raw_gt, emotion_mapped_gt)

    # Guardar el CSV actualizado
    df.to_csv(output_csv, index=False)
    print("CSV updated and stored in : ", output_csv)


if __name__ == "__main__":
    
    process_start_time = timer.perf_counter()
    #TODO: cambiar el input para que sea el nombre del CSV a analizar, y no el origen de los datos, ya que se pueden analizar CSVs de ambos orígenes indistintamente
    dataset = input("Which dataset would you like to get analyzed? (ISEAR/GOEMOTIONS/KAGGLE/DEFAULT): ")
    while dataset.upper() != 'ISEAR' and dataset.upper() != 'GOEMOTIONS' and dataset.upper() != 'KAGGLE' and dataset.upper() != 'DEFAULT':
        dataset = input("Incorrect format, from which dataset would you like to get the data? (ISEAR/GOEMOTIONS/KAGGLE/DEFAULT): ")

    MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se ejecuta desde otro equipo
                                                  # EN: Change to the PC's which has the database IP if running from another computer
    DATABASE_NAME = 'Messages'
    COLLECTION_NAME = dataset + '_messages'
    client = MongoClient(MONGO_URI)
    data_base = client[DATABASE_NAME]
    colection = data_base[COLLECTION_NAME]

    cursor = colection.find()
    documents = list(cursor)   
    df = pd.DataFrame(documents)

    # ES: Guardar el CSV en el directorio actual
    # EN: Save the CSV in the current directory
    if dataset.upper() == 'ISEAR':
        csv_filename = "./data/processed/isear_emotions_normalized.csv"
    elif dataset.upper() == 'GOEMOTIONS':
        csv_filename = "./data/processed/goemotions_normalized.csv"
    elif dataset.upper() == 'KAGGLE':
        csv_filename = "./data/processed/kaggle_emotions_normalized.csv"
    else:
        print("Using default CSV file (mini go emotions dataset).")
        csv_filename = "./data/processed/default_emotions_normalized.csv"

    stringJustification = input("Would you like a justification of the sentiment analysis? (Y/N): ")
    valid_format = False
    while not valid_format:
        if stringJustification.upper() == 'Y' or stringJustification.upper() == 'N':
            valid_format = True
            if stringJustification.upper() == 'Y':
                justification = True
            else:
                justification = False
        else:
            stringJustification = input("Incorrect format, would you like to get the sentiment classified? (Y/N): ")
    
    llm_chosen = input(f"Which LLM model would you like to use? ({'/'.join(llm_call.AVAILABLE_LLMS)}): ")
    valid_format = False
    while not valid_format:
        if llm_chosen.upper() in llm_call.AVAILABLE_LLMS:
            valid_format = True
        else:
            llm_chosen = input(f"Incorrect format, which LLM model would you like to use? ({'/'.join(llm_call.AVAILABLE_LLMS)}): ")

    output_csv = "./data/results/no_concurrente/" + llm_chosen + "_"+ dataset + "_results.csv"
    process_csv(csv_filename, output_csv, llm_chosen, justification, dataset)
    
    # ES: Realizar análisis de sentimientos y emociones con bibliotecas, y guardar los resultados en el CSV
    # EN: Load the CSV with LLM results
    df = pd.read_csv(output_csv)
    
    # ES: Realizar análisis de sentimientos con bibliotecas
    # EN: Perform sentiment analysis with libraries
    df = libraries_call.TextBlob_sentiment_analysis(df)
    df = libraries_call.vader_sentiment_analysis(df)
    df = libraries_call.BERT_sentiment_analysis(df)
    

    # EN: Save sentiment analysis results
    sentiment_csv = "./data/results/no_concurrente/" + llm_chosen + "_"+ dataset + "_sentiment_analysis_results.csv"
    os.makedirs(os.path.dirname(sentiment_csv), exist_ok=True)
    df.to_csv(sentiment_csv, index=False)
    print(f"Sentiment analysis results saved to {sentiment_csv}")
    
    # ES: Realizar análisis de emociones con librerías
    # EN: Perform emotion analysis with libraries
    df = libraries_call.NRCLex_emotion_analysis(df)
    df = libraries_call.GoEmotions_EmoRoBERTa_emotion_analysis(df, "EmoRoBERTa")
    df = libraries_call.GoEmotions_EmoRoBERTa_emotion_analysis(df, "GoEmotions")
    
    # Map raw emotions from each library to normalized emotions using normalize_emotion_label
    df["emotion_mapped_NRCLex"] = df["emotion_raw_NRCLex"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
    df["emotion_mapped_EmoRoBERTa"] = df["emotion_raw_EmoRoBERTa"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
    df["emotion_mapped_GoEmotions"] = df["emotion_raw_GoEmotions"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
    
    # ES: Guardar los resultados del análisis de emociones con librerías
    # EN: Save emotion analysis performed with libraries results 
    emotion_csv = "./data/results/no_concurrente/" + llm_chosen + "_"+ dataset + "_sentiment_emotion_analysis_results.csv"
    os.makedirs(os.path.dirname(emotion_csv), exist_ok=True)

    total_process_time_seconds = round(timer.perf_counter() - process_start_time, 4)
    df["total_process_time_seconds_" + llm_chosen ] = total_process_time_seconds
    df.to_csv(emotion_csv, index=False)
    print(f"Emotion analysis results saved to {emotion_csv}")
    print(f"Total process time: {total_process_time_seconds} seconds")
    print("All processes completed successfully.")
    
