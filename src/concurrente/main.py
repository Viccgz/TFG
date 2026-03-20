import pandas as pd
from pymongo import MongoClient
import sentiment_analysis.llm_call as llm_call
import utils
import sentiment_analysis.libraries_call as libraries_call
import metrics

n_items_used = 1

def process_csv(input_csv, output_csv, llm_chosen, justification, source):
    MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se guarda en otro equipo
                                                  # EN: Change to the PC's which has the database IP if saving in another computer
    DATABASE_NAME = llm_chosen + 'Results_EmotionalAnalysis'
    COLLECTION_NAME = source + '_results'
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    global n_items_used

    df = pd.read_csv(input_csv, delimiter=";", encoding="utf-8")
    for column in ["sentiment_" + llm_chosen, "certainty_" + llm_chosen, "justification_" + llm_chosen]:
        if column not in df.columns:
            df[column] = ""

    for index, row in df.iterrows():
        date = row["date"]
        time = row["time"]
        source = row["source"]
        message = row["message"]

        # ES:  procesar si los campos son nulos
        # EN:  process if fields are null
        if pd.isnull(row["sentiment_" + llm_chosen]) or pd.isnull(row["certainty_" + llm_chosen]) or pd.isnull(row["justification_" + llm_chosen]):
            if llm_chosen.upper() == 'CHATGPT':
                sentiment, certainty, justification, date, time = llm_call.send_to_chatgpt(message, justification)
            elif llm_chosen.upper() == 'GEMINI':
                sentiment, certainty, justification, date, time = llm_call.send_to_gemini(message, justification)
            elif llm_chosen.upper() == 'DEEPSEEK':
                sentiment, certainty, justification, date, time = llm_call.send_to_deepseek(message, justification)
            # ES: Actualizar el dataframe creando una nueva columna
            # EN: Update the dataframe creating a new column
            df.at[index, "sentiment_" + llm_chosen] = sentiment
            df.at[index, "certainty_" + llm_chosen] = certainty
            df.at[index, "justification_" + llm_chosen] = justification
            df.at[index, "processing_date"] = date
            df.at[index, "processing_hour"] = time

        # ES: Guardar en MongoDB
        # EN: Save in MongoDB
            utils.save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification, date, time)
    # Guardar el CSV actualizado
    df.to_csv(output_csv, index=False)
    print("CSV updated and stored in : ", output_csv)


if __name__ == "__main__":
    
    #TODO: cambiar el input para que sea el nombre del CSV a analizar, y no el origen de los datos, ya que se pueden analizar CSVs de ambos orígenes indistintamente
    source = input("Which dataset would you like to get analyzed?: ")
    while source.upper() != 'X' and source.upper() != 'TELEGRAM':
        source = input("Incorrect format, from which social network would you like to get the data? (X/Telegram): ")

    MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar a la IP del PC con la base de datos si se ejecuta desde otro equipo
                                                  # EN: Change to the PC's which has the database IP if running from another computer
    DATABASE_NAME = 'Messages'
    COLLECTION_NAME = source + '_messages'
    client = MongoClient(MONGO_URI)
    data_base = client[DATABASE_NAME]
    colection = data_base[COLLECTION_NAME]

    cursor = colection.find()
    documents = list(cursor)   
    df = pd.DataFrame(documents)

    # ES: Guardar el CSV en el directorio actual
    # EN: Save the CSV in the current directory
    csv_filename = f"{COLLECTION_NAME}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"CSV stored in: {csv_filename}")

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
    
    llm_chosen = input("Which LLM model would you like to use? (ChatGPT/Gemini/Deepseek): ")
    valid_format = False
    while not valid_format:
        if llm_chosen.upper() == 'CHATGPT' or llm_chosen.upper() == 'GEMINI' or llm_chosen.upper() == 'DEEPSEEK':
            valid_format = True
        else:
            llm_chosen = input("Incorrect format, which LLM model would you like to use? (ChatGPT/Gemini/Deepseek): ")

    input_csv = csv_filename
    output_csv = llm_chosen+"_results_"+source+"_emontional_analysis_results.csv"
    process_csv(input_csv, output_csv, llm_chosen, justification, source)
    new_out_csv = libraries_call.TextBlob_sentiment_analysis(output_csv)
    new_out_csv = libraries_call.vader_sentiment_analysis(new_out_csv)
    new_out_csv = libraries_call.BERT_sentiment_analysis(new_out_csv)
    new_out_csv = metrics.calculate_majority(new_out_csv)
    metrics.calculate_accuracy(new_out_csv)
    metrics.interrated(new_out_csv)
    metrics.calculateStatisticalDiff(new_out_csv)
    metrics.calculateSummarySentimentLLMs(new_out_csv)
    metrics.calculateStatisticsSentiment(new_out_csv)
    metrics.carryOutTextAnalysis(new_out_csv)
    print("All processes completed successfully.")
    
