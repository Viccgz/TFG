from datetime import datetime
from logging import log
import re
import json

LOG_FILE_CHATGPT = "chatgpt_log.txt"
LOG_FILE_GEMINI = "gemini_log.txt"
LOG_FILE_DEEPSEEK = "deepseek_log.txt"

def log_message(msg, llm):
    
    # ES: Imprimir mensaje y guardar en archivo de log
    # EN: Print message and save to log file
    if llm == "chatgpt":
        LOG_FILE = LOG_FILE_CHATGPT
    elif llm == "gemini":
        LOG_FILE = LOG_FILE_GEMINI
    elif llm == "deepseek":
        LOG_FILE = LOG_FILE_DEEPSEEK
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(msg + "\n")

# ES: Función para guardar en MongoDB el resultado final
# EN: Function to save the final result in MongoDB
def save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification, processing_date, processing_hour):

    #ES: Porcesar la justificacion de la respuesta y cambiar caracteres especiales
    #EN: Process the justification of the response and change special characters
    if justification:
        justification = justification.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and ").replace("&", "and").replace(";", " ")


    collection.insert_one({
            "message": message,
            "sentiment": sentiment,
            "certainty": certainty,
            "justification": justification,
            "date": processing_date,
            "time": processing_hour
        })

def sanitize_json(raw_json):
    try:
        # ES: Eliminar marcadores de bloque de código circundantes. Este problema surgio porque se devolvia el json con formato ```json y ``` alrededor, lo cual hacia que json.loads fallara
        # EN: Remove surrounding code block markers. This issue arose because the json was being returned with ```json and ``` around it which caused json.loads to fail
        sanitized = re.sub(r'^```json|```$', '', raw_json.strip(), flags=re.MULTILINE).strip()
        return sanitized
    except Exception as e:
        print(f"Error sanitizing JSON: {e}")
        return raw_json

def sanitize_text(text):
    #ES: Procesar texto y cambiar caracteres especiales
    #EN: Process text and change special characters
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and ").replace("&", "and").replace(";", " ").replace(":", " ")
    return text

def parse_response(response):
    try:
        sanitized_response = sanitize_json(response)
        data = json.loads(sanitized_response)
        emotion = data.get("emotion", "=")
        log_message(f"Emotion: {emotion}\n")
        certainty = data.get("certainty", 0.0)
        log_message(f"Certainty: {certainty}\n")
        justification = data.get("justification", "")
        log_message(f"Justification: {justification}\n")
        return emotion, certainty, justification
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON: {e}")
        return "unknown", 0.0, "Error parsing response"
    
def process_response(response, source):
    try:
        sanitize_response = sanitize_json(response)
        log_message(f"ChatGPT's sanitized response:\n{sanitize_response}", source)
        sentiment, certainty, justification = parse_response(sanitize_response)
        processing_date = datetime.now().strftime("%Y-%m-%d")
        processing_hour = datetime.now().strftime("%H:%M:%S")
        log_message(f"Parsed response - Sentiment: {sentiment}, Certainty: {certainty}, Justification: {justification}, Processing_date: {processing_date}, Processing_hour: {processing_hour}\n", source)  
        return sentiment, certainty, justification, processing_date, processing_hour
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON: {e}")
        return 'NA', 'NA', 'NA'