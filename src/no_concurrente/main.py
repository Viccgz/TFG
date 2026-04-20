import time as timer
import pandas as pd
from pymongo import MongoClient
from dataset_preprocessing.emotion_mapper import normalize_emotion_label
import sentiment_analysis.llm_call as llm_call
import utils
import sentiment_analysis.libraries_call as libraries_call
import metrics


def process_csv(input_csv, output_csv, llm_chosen, justification, dataset):
    process_start_time = timer.perf_counter()

    MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se guarda en otro equipo
                                                  # EN: Change to the PC's which has the database IP if saving in another computer
    DATABASE_NAME = 'TFG_Results_EmotionalAnalysis'
    COLLECTION_NAME = dataset + '_' + llm_chosen + '_results'
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    df = pd.read_csv(input_csv, sep=None, engine='python', encoding="utf-8")
    for column in ["sentiment_" + llm_chosen, "certainty_" + llm_chosen, "justification_" + llm_chosen, "total_process_time_seconds"]:
        if column not in df.columns:
            df[column] = None
    print("llm_chosen: ", llm_chosen)
    for index, row in df.iterrows():
        id = row["id"]
        message = row["text"]
        emotion_raw_gt = row["emotion_gt"]
        emotion_mapped_gt = row["emotion_gt_mapped"] 

        # ES:  procesar si los campos son nulos
        # EN:  process if fields are null
        if pd.isnull(row["sentiment_" + llm_chosen]) or pd.isnull(row["certainty_" + llm_chosen]) or pd.isnull(row["justification_" + llm_chosen]):
            function = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
            sentiment, certainty, justification, date, time = function(message, justification, "sentiment_analysis", dataset)

            # ES: Actualizar el dataframe creando una nueva columna
            # EN: Update the dataframe creating a new column
            df.at[index, "sentiment_" + llm_chosen] = sentiment
            df.at[index, "certainty_sentiment_" + llm_chosen] = certainty
            df.at[index, "justification_sentiment_" + llm_chosen] = justification
            df.at[index, "processing_date_sentiment"] = date
            df.at[index, "processing_hour_sentiment"] = time

            #ES: Realizar análisis de emociones con LLMs para cada mensaje, y guardar el resultado en el dataframe
            #EN: Perform emotion analysis with LLMs for each message, and save the result
            function = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
            emotion, certainty_emotion, justification_emotion, date_emotion, time_emotion = function(message, justification, "emotion_analysis", dataset)
            
            df.at[index, "emotion_" + llm_chosen] = emotion
            df.at[index, "certainty_emotion_" + llm_chosen] = certainty_emotion
            df.at[index, "justification_emotion_" + llm_chosen] = justification_emotion
            df.at[index, "processing_date_emotion"] = date_emotion
            df.at[index, "processing_hour_emotion"] = time_emotion

            # ES: Guardar en MongoDB
            # EN: Save in MongoDB
            utils.save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification, date, time, emotion, certainty_emotion, justification_emotion, date_emotion, time_emotion, emotion_raw, id, emotion_raw_gt, emotion_mapped_gt)

    # Guardar el CSV actualizado
    df.to_csv(output_csv, index=False)
    print("CSV updated and stored in : ", output_csv)
    return process_start_time
    


if __name__ == "__main__":
    
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
        csv_filename = "./data/processed/goemotions_emotions_normalized.csv"
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

    output_csv = "./data/results/" + llm_chosen + "_"+ dataset + "_results.csv"
    process_start_time = process_csv(csv_filename, output_csv, llm_chosen, justification, dataset)
    
    # ES: Realizar análisis de sentimientos y emociones con bibliotecas, y guardar los resultados en el CSV
    # EN: Load the CSV with LLM results
    df = pd.read_csv(output_csv)
    
    # ES: Realizar análisis de sentimientos con bibliotecas
    # EN: Perform sentiment analysis with libraries
    df = libraries_call.TextBlob_sentiment_analysis(df)
    df = libraries_call.vader_sentiment_analysis(df)
    df = libraries_call.BERT_sentiment_analysis(df)
    

    # EN: Save sentiment analysis results
    sentiment_csv = "./data/results/" + llm_chosen + "_"+ dataset + "_sentiment_analysis_results.csv"
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
    emotion_csv = "./data/results/" + llm_chosen + "_"+ dataset + "_sentiment_emotion_analysis_results.csv"
    df.to_csv(emotion_csv, index=False)
    print(f"Emotion analysis results saved to {emotion_csv}")
    
    #new_out_csv = metrics.calculate_majority(new_out_csv)
    #metrics.calculate_accuracy(new_out_csv)
    #metrics.interrated(new_out_csv)
    #metrics.calculateStatisticalDiff(new_out_csv)
    #metrics.calculateSummarySentimentLLMs(new_out_csv)
    #metrics.calculateStatisticsSentiment(new_out_csv)
    #metrics.carryOutTextAnalysis(new_out_csv)
    
    total_process_time_seconds = round(timer.perf_counter() - process_start_time, 4)
    df["total_process_time_seconds"] = total_process_time_seconds
    print(f"Total process time: {total_process_time_seconds} seconds")
    print("All processes completed successfully.")
    
