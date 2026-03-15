import os
import pandas as pd
from pymongo import MongoClient
import llm_call
import utils
import asyncio
import libraries_call
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
        id_source = row["id_source"]
        created_at = row["created_at"]
        author_id = row["author_id"]
        date = row["date"]
        time = row["time"]
        source = row["source"]
        n_items = row["n_items"]
        n_likes = row["n_likes"]
        n_retweets = row["n_retweets"]
        n_impressions = row["n_impressions"]
        n_replies = row["n_replies"]
        n_quotes = row["n_quotes"]
        n_bookmarks = row["n_bookmarks"]
        message = row["message"]
        n_media = row.get("n_media", 0)
        media = row.get("media", [])
        id_message = row.get("id_message", "")
        chat = row.get("chat", "")
        sender_id = row.get("sender_id", "")
        dateCreated = row.get("dateCreated", "")
        lang = row.get("lang", "")
        
  
        # ES:  procesar si los campos son nulos
        # EN:  process if fields are null
        if pd.isnull(row["sentiment_" + llm_chosen]) or pd.isnull(row["certainty_" + llm_chosen]) or pd.isnull(row["justification_" + llm_chosen]):
            if llm_chosen.upper() == 'CHATGPT':
                sentiment, certainty, justification, n_items_used = llm_call.send_to_chatgpt(message, media, justification, n_media, n_items_used )
            elif llm_chosen.upper() == 'GEMINI':
                sentiment, certainty, justification, n_items_used = llm_call.send_to_gemini(message, media, justification, n_media, n_items_used )
            elif llm_chosen.upper() == 'DEEPSEEK':
                sentiment, certainty, justification, n_items_used = llm_call.send_to_deepseek(message, media, justification, n_media, n_items_used )
            # ES: Actualizar el dataframe creando una nueva columna
            # EN: Update the dataframe creating a new column
            df.at[index, "sentiment_" + llm_chosen] = sentiment
            df.at[index, "certainty_" + llm_chosen] = certainty
            df.at[index, "justification_" + llm_chosen] = justification
            df.at[index, "n_items_decision"] = n_items_used

        # ES: Guardar en MongoDB
        # EN: Save in MongoDB
        if source.upper() == 'X':
            utils.save_in_mongodb_final_csv_X(collection, message, sentiment, certainty, justification, n_media, n_items_used, id_source, created_at, author_id, date, time, source, n_items, n_likes, n_retweets, n_impressions, n_replies, n_quotes, n_bookmarks)
        else:
            utils.save_in_mongodb_final_csv_Telegram (collection, id_message, chat, message,  media, n_items, n_media, sender_id, dateCreated, lang, justification, n_items_used, sentiment, certainty)
        n_items_used = 1
    # Guardar el CSV actualizado
    df.to_csv(output_csv, index=False)
    print("CSV updated and stored in : ", output_csv)


if __name__ == "__main__":
    
    source = input("Which dataset would you like to get analyzed? (X/Telegram): ")
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

    stringJustification = input("Would you like to get the sentiment classified? (Y/N): ")
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
    
