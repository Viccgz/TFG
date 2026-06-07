import time as timer
import pandas as pd
from pymongo import MongoClient
import sentiment_analysis.llm_call as llm_call
import utils
import sentiment_analysis.libraries_call as libraries_call
import concurrent.futures
import threading
from dataset_preprocessing.emotion_mapper import normalize_emotion_label
from pathlib import Path
import os
import chardet


def process_csv(input_csv, output_csv, llm_chosen, justification, dataset, num_threads=8):
    """
    ES: Procesa un CSV de entrada con mensajes, realiza análisis de sentimiento y emoción utilizando LLMs de forma concurrente, y guarda los resultados en un nuevo CSV y en MongoDB.
    EN: Processes an input CSV with messages, performs sentiment and emotion analysis using LLMs concurrently, and saves the results to a new CSV and MongoDB.
    Args:
        input_csv (str): Ruta al CSV de entrada con los mensajes a analizar.
        output_csv (str): Ruta al CSV de salida donde se guardarán los resultados.
        llm_chosen (str): El nombre del LLM a utilizar para el análisis.
        justification (bool): Si se debe incluir una justificación en el análisis.
        dataset (str): El nombre del dataset, utilizado para nombrar la colección de MongoDB.
        num_threads (int): El número de threads a utilizar para el procesamiento concurrente.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se guarda en otro equipo
                                                  # EN: Change to the PC's which has the database IP if saving in another computer
    DATABASE_NAME = 'TFG_Results_EmotionalAnalysis'
    COLLECTION_NAME = dataset + '_' + llm_chosen + '_results_concurrente'
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
        "emotion_raw_" + llm_chosen,
        "emotion_mapped_" + llm_chosen,
        "justification_emotion_" + llm_chosen,
        "processing_date_sentiment",
        "processing_hour_sentiment",
        "processing_date_emotion",
        "processing_hour_emotion",
        "emotion_raw_with_sentiment_" + llm_chosen,
        "emotion_mapped_with_sentiment_" + llm_chosen,
        "justification_emotion_with_sentiment_" + llm_chosen,
        "processing_date_emotion_with_sentiment",
        "processing_hour_emotion_with_sentiment"
    ]
    numeric_columns = [
        "certainty_sentiment_" + llm_chosen,
        "certainty_emotion_" + llm_chosen,
        "certainty_emotion_with_sentiment_" + llm_chosen
    ]

    for column in string_columns:
        if column not in df.columns:
            df[column] = pd.Series([None] * len(df), dtype="string")
        elif df[column].dtype not in ["object", "string"]:
            df[column] = df[column].astype("string")

    for column in numeric_columns:
        if column not in df.columns:
            df[column] = pd.Series([pd.NA] * len(df), dtype="Float64")
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Float64")
    print("llm_chosen: ", llm_chosen)

    # Lock para acceso exclusivo al DataFrame
    df_lock = threading.Lock()

    def to_string_value(value):
        if pd.isna(value):
            return pd.NA
        return str(value)

    def to_numeric_value(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            value = value.strip().rstrip('%')
        try:
            return float(value)
        except (ValueError, TypeError):
            return pd.NA

    def process_row(index, row, justification):
        """
        ES: Procesa una fila del DataFrame, realizando análisis de sentimiento y emoción con LLMs si los campos correspondientes están vacíos o son nulos. Actualiza el DataFrame con los resultados y guarda en MongoDB.
        EN: Processes a row of the DataFrame, performing sentiment and emotion analysis with LLMs if the corresponding fields are empty or null. Updates the DataFrame with the results and saves to MongoDB.

        Args:
            index (int): El índice de la fila en el DataFrame.
            row (pd.Series): La fila del DataFrame a procesar.
            justification (bool): Si se debe incluir una justificación en el análisis.
        Returns:
            None
        """
        id = row["id"]
        message = row["text"]
        emotion_raw_gt = row["emotion_gt"]
        emotion_mapped_gt = row["emotion_gt_mapped"] 

        # ES:  procesar si los campos son nulos
        # EN:  process if fields are null
        if (pd.isnull(row["sentiment_" + llm_chosen]) or pd.isnull(row["emotion_raw_" + llm_chosen]) or pd.isnull(row["emotion_raw_with_sentiment_" + llm_chosen])
        or row["sentiment_" + llm_chosen] == "NA" or row["emotion_raw_" + llm_chosen] == "NA" or row["emotion_raw_with_sentiment_" + llm_chosen] == "NA"):
            sentiment, certainty, justification_sentiment, date, time = func(message, justification, "sentiment_analysis", dataset, id)
            
            #ES: Realizar análisis de emociones con LLMs para cada mensaje, y guardar el resultado en el dataframe
            #EN: Perform emotion analysis with LLMs for each message, and save the result
            func = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
            emotion_raw_llm, emotion_mapped, certainty_emotion, justification_emotion, date_emotion, time_emotion = func(message, justification, "emotion_analysis", dataset, id)
            
            func = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
            emotion_with_sentiment_raw_llm, emotion_with_sentiment_mapped, certainty_emotion_with_sentiment, justification_emotion_with_sentiment, date_emotion_with_sentiment, time_emotion_with_sentiment = func(message, justification, "emotion_with_sentiment_analysis", dataset, id, sentiment=sentiment)
            # Usar lock para actualizar el DataFrame
            with df_lock:
                df.at[index, "text"] = to_string_value(message)
                df.at[index, "sentiment_" + llm_chosen] = to_string_value(sentiment)
                df.at[index, "certainty_sentiment_" + llm_chosen] = to_numeric_value(certainty)
                if justification:
                    df.at[index, "justification_sentiment_" + llm_chosen] = to_string_value(justification_sentiment)
                df.at[index, "processing_date_sentiment"] = to_string_value(date)
                df.at[index, "processing_hour_sentiment"] = to_string_value(time)
                df.at[index, "emotion_raw_" + llm_chosen] = to_string_value(emotion_raw_llm)
                df.at[index, "emotion_mapped_" + llm_chosen] = to_string_value(emotion_mapped)
                df.at[index, "certainty_emotion_" + llm_chosen] = to_numeric_value(certainty_emotion)
                if justification:
                    df.at[index, "justification_emotion_" + llm_chosen] = to_string_value(justification_emotion)
                df.at[index, "processing_date_emotion"] = to_string_value(date_emotion)
                df.at[index, "processing_hour_emotion"] = to_string_value(time_emotion)

                df.at[index, "emotion_raw_with_sentiment_" + llm_chosen] = to_string_value(emotion_with_sentiment_raw_llm)
                df.at[index, "emotion_mapped_with_sentiment_" + llm_chosen] = to_string_value(emotion_with_sentiment_mapped)
                df.at[index, "certainty_emotion_with_sentiment_" + llm_chosen] = to_numeric_value(certainty_emotion_with_sentiment)
                if justification:
                    df.at[index, "justification_emotion_with_sentiment_" + llm_chosen] = to_string_value(justification_emotion_with_sentiment)
                df.at[index, "processing_date_emotion_with_sentiment"] = to_string_value(date_emotion_with_sentiment)
                df.at[index, "processing_hour_emotion_with_sentiment"] = to_string_value(time_emotion_with_sentiment)

            # ES: Guardar en MongoDB (pymongo es thread-safe)
            # EN: Save in MongoDB (pymongo is thread-safe)
            utils.save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification_sentiment, date, time, emotion_mapped, certainty_emotion, justification_emotion, date_emotion, time_emotion, emotion_raw_llm, id, emotion_raw_gt, emotion_mapped_gt, emotion_with_sentiment_raw_llm, emotion_mapped_with_sentiment, certainty_emotion_with_sentiment, justification_emotion_with_sentiment, date_emotion_with_sentiment, time_emotion_with_sentiment)

    # Procesar filas concurrentemente
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_row, index, row, justification) for index, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Esperar a que termine, aunque no hay excepciones manejadas

    # Guardar el CSV actualizado
    df.to_csv(output_csv, index=False)
    print("CSV updated and stored in : ", output_csv)


def get_available_datasets(processed_dir="./data/processed"):
    """
    ES: Obtiene la lista de conjuntos de datos disponibles en el directorio especificado.
    EN: Gets the list of available datasets in the specified directory.

    Args:
        processed_dir (str): El directorio donde se encuentran los CSV procesados.
    Returns:
        list of tuples: Una lista de tuplas, donde cada tupla contiene el nombre del dataset (sin extensión) y la ruta al archivo CSV correspondiente.
    """
    processed_path = Path(processed_dir)
    datasets = []
    if not processed_path.exists():
        return datasets

    for file in processed_path.glob("*.csv"):
        if file.is_file():
            datasets.append((file.stem, str(file)))

    datasets.sort(key=lambda pair: pair[0].lower())
    return datasets

    


if __name__ == "__main__":

    try:
        process_start_time = timer.perf_counter()
        #TODO: cambiar el input para que sea el nombre del CSV a analizar, y no el origen de los datos, ya que se pueden analizar CSVs de ambos orígenes indistintamente
        available_datasets = get_available_datasets("./data/processed")
        if not available_datasets:
            raise FileNotFoundError("No dataset CSV files found in ./data/processed")

        print("Available datasets:")
        for index, (name, path) in enumerate(available_datasets, start=1):
            print(f"  {index}) {name}")

        selected = None
        while selected is None:
            choice = input("Select dataset by number or exact name: ").strip()
            if choice.strip() != "" and choice.isdigit():
                index = int(choice)
                if 1 <= index <= len(available_datasets):
                    selected = available_datasets[index - 1]
                    break
            else:
                for name, path in available_datasets:
                    if name.lower() == choice.lower():
                        selected = (name, path)
                        break

            if selected is None:
                print("Invalid selection. Please choose dataset number or exact name from the list.")

        dataset, csv_filename = selected
        print(f"Selected dataset: {dataset}")

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
        
        MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se ejecuta desde otro equipo
                                                    # EN: Change to the PC's which has the database IP if running from another computer
        DATABASE_NAME = 'TFG_Results_EmotionalAnalysis'
        COLLECTION_NAME = dataset + '_' + llm_chosen + '_results_concurrente'
        print(f"Connecting to MongoDB at {MONGO_URI}, database: {DATABASE_NAME}, collection: {COLLECTION_NAME}")
        client = MongoClient(MONGO_URI)
        data_base = client[DATABASE_NAME]
        colection = data_base[COLLECTION_NAME]

        cursor = colection.find()
        documents = list(cursor)   
        df = pd.DataFrame(documents)

        output_csv = "./data/results/concurrente/" + llm_chosen + "_"+ dataset + "_results.csv"
        num_threads = 8  # Sugerido para I/O bound operations como llamadas a LLM
        print(f"Using {num_threads} threads for concurrent processing.")
        llm_start_time = timer.perf_counter()
        process_csv(csv_filename, output_csv, llm_chosen, justification, dataset, num_threads)
        total_llm_time_seconds = round(timer.perf_counter() - llm_start_time, 4)
        print(f"LLM processing time: {total_llm_time_seconds} seconds")
        # Load the CSV with LLM results
        df = pd.read_csv(output_csv)
        libraries_start_time = timer.perf_counter()
        # Perform sentiment analysis with libraries
        df = libraries_call.TextBlob_sentiment_analysis(df)
        df = libraries_call.vader_sentiment_analysis(df)
        df = libraries_call.BERT_sentiment_analysis(df)
        
        # Save sentiment analysis results
        sentiment_csv = "./data/results/concurrente/" + llm_chosen + "_"+ dataset + "_sentiment_analysis_results.csv"
        os.makedirs(os.path.dirname(sentiment_csv), exist_ok=True)
        df.to_csv(sentiment_csv, index=False)
        print(f"Sentiment analysis results saved to {sentiment_csv}")
        
        # Perform emotion analysis with libraries
        df = libraries_call.NRCLex_emotion_analysis(df)
        df = libraries_call.GoEmotions_EmoRoBERTa_emotion_analysis(df, "EmoRoBERTa")
        df = libraries_call.GoEmotions_EmoRoBERTa_emotion_analysis(df, "GoEmotions")
        
        # Map raw emotions from each library to normalized emotions using normalize_emotion_label
        df["emotion_mapped_NRCLex"] = df["emotion_raw_NRCLex"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
        df["emotion_mapped_EmoRoBERTa"] = df["emotion_raw_EmoRoBERTa"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
        df["emotion_mapped_GoEmotions"] = df["emotion_raw_GoEmotions"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
        
        total_libraries_time_seconds = round(timer.perf_counter() - libraries_start_time, 4)
        print(f"Libraries processing time: {total_libraries_time_seconds} seconds")
 
        # ES: Añadir columnas de tiempo al dataframe
        # EN: Add timing columns to the dataframe
        df["total_LLM_time_seconds_" + llm_chosen] = total_llm_time_seconds
        df["total_libraries_time_seconds"] = total_libraries_time_seconds
        # Save emotion analysis results
        emotion_csv = "./data/results/concurrente/" + llm_chosen + "_"+ dataset + "_sentiment_emotion_analysis_results.csv"
        os.makedirs(os.path.dirname(emotion_csv), exist_ok=True)
        df.to_csv(emotion_csv, index=False)
        print(f"Emotion analysis results saved to {emotion_csv}")
        
        
        # Reordenar columnas en el orden especificado
        order_base_columns = ["id", "text",
                    "sentiment_" + llm_chosen, 
                    "certainty_sentiment_" + llm_chosen, 
                    "justification_sentiment_" + llm_chosen,
                    "processing_date_sentiment", "processing_hour_sentiment",
                    "emotion_gt", "emotion_gt_mapped",
                    "emotion_raw_" + llm_chosen,
                    "emotion_mapped_" + llm_chosen,
                    "certainty_emotion_" + llm_chosen,
                    "justification_emotion_" + llm_chosen,
                    "processing_date_emotion", "processing_hour_emotion"]
        time_columns = [
            "total_LLM_time_seconds_" + llm_chosen,
            "total_libraries_time_seconds",
        ]

        order_remaining_cols = [
            col for col in df.columns
            if col not in order_base_columns
            and col not in time_columns
            and col != "_id"
        ]
 
        # ES: Orden final: base + librerías + tiempos
        # EN: Final order: base + libraries + timings
        final_cols = order_base_columns + order_remaining_cols + time_columns
        final_cols = [col for col in final_cols if col in df.columns]
        df = df[final_cols]
 
        df.to_csv(emotion_csv, index=False)
        
        print(f"Total process time: {total_process_time_seconds} seconds")
        print("All processes completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
    