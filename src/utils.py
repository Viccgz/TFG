import re
import datetime
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
def save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification):

    #ES: Porcesar la justificacion de la respuesta y cambiar caracteres especiales
    #EN: Process the justification of the response and change special characters
    if justification:
        justification = justification.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and ").replace("&", "and").replace(";", " ")

    processing_date = datetime.datetime.now().strftime("%Y-%m-%d")
    processing_hour = datetime.datetime.now().strftime("%H:%M:%S")

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

def parse_chatgpt_response(response):
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