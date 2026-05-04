from datetime import datetime
from logging import log
import os
import re
import json

# ----------------LOGGING UTILS----------------

LOG_PLANTILLAS = {
    "chatgpt": "LOGS/{dataset}_chatgpt_log.txt",
    "gemini": "LOGS/{dataset}_gemini_log.txt",
    "deepseek": "LOGS/{dataset}_deepseek_log.txt"
}

def log_message(msg, llm, dataset, id):
    template = LOG_PLANTILLAS.get(llm.lower())
    if not template:
        print(f"Error: LLM '{llm}' invvalido con identificador {id}. No se registrará el mensaje: {msg}")
        return
    log_file_path = template.format(dataset=dataset.upper())
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    print(msg)
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} (ID: {id})\n")
    except Exception as e:
        print(f"No se pudo escribir en el log: {e}")


# ----------------DB UTILS----------------

# ES: Función para guardar en MongoDB el resultado final
# EN: Function to save the final result in MongoDB
def save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification, processing_date, processing_hour, emotion_mapped_llm, certainty_emotion, justification_emotion, date_emotion, time_emotion, emotion_raw_llm, id, emotion_raw_gt, emotion_mapped_gt):

    #ES: Porcesar la justificacion de la respuesta y cambiar caracteres especiales
    #EN: Process the justification of the response and change special characters
    if justification:
        justification = sanitize_text(justification)

    if message:
        message = sanitize_text(message)

    collection.insert_one({
            "id": id,
            "message": message,
            "emotion_raw_gt": emotion_raw_gt,
            "emotion_mapped_gt": emotion_mapped_gt,
            
            "sentiment": sentiment,
            "certainty": certainty,
            "justification": justification,
            "date": processing_date,
            "time": processing_hour,
            
            "emotion_raw_llm": emotion_raw_llm,
            "emotion_mapped_llm": emotion_mapped_llm,
            "certainty_emotion": certainty_emotion,
            "justification_emotion": justification_emotion,
            "date_emotion": date_emotion,
            "time_emotion": time_emotion
        })



# ----------------RESPONSE PROCESSING UTILS----------------
def sanitize_json(raw_json):
    try:
        # ES: Eliminar marcadores de bloque de código circundantes. Este problema surgio porque se devolvia el json con formato ```json y ``` alrededor, lo cual hacia que json.loads fallara
        # EN: Remove surrounding code block markers. This issue arose because the json was being returned with ```json and ``` around it which caused json.loads to fail
        # sanitized = re.sub(r'^```json|```$', '', raw_json.strip(), flags=re.MULTILINE).strip()
        sanitized = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_json.strip(), flags=re.IGNORECASE).strip()
        return sanitized
    except Exception as e:
        print(f"Error sanitizing JSON: {e}")
        return raw_json

def sanitize_text(text):
    #ES: Procesar texto y cambiar caracteres especiales
    #EN: Process text and change special characters
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and ").replace("&", "and").replace(";", " ").replace(":", " ")
    return text

def parse_response(response, evaluation_mode, dataset, source, id):
    try:
        if not response.strip():
            log_message("Empty response after sanitization", source, dataset, id)
            return "unknown", 0.0, "Empty response after sanitization"
        data = json.loads(response)
        if evaluation_mode == "sentiment_analysis":
            sentiment = data.get("sentiment", "unknown")
            log_message(f"Sentiment: {sentiment}\n", source, dataset, id)
        else:
            emotion = data.get("emotion", "=")
            log_message(f"Emotion: {emotion}\n", source, dataset, id)

        certainty = data.get("certainty", 0.0)
        log_message(f"Certainty: {certainty}\n", source, dataset, id)

        justification = data.get("justification", "")
        log_message(f"Justification: {justification}\n", source, dataset, id)
        if evaluation_mode == "sentiment_analysis":
            return sentiment, certainty, justification
        else:
            return emotion, certainty, justification

        
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON: {e}", source, dataset, id)
        return "unknown", 0.0, "Error parsing response"
    
def process_response(response, source, evaluation_mode, dataset, id):
    try:
        if not response.strip():
            log_message("Empty response from llm", source, dataset, id)
            return "unknown", 0.0, "Empty response from llm", "NA", "NA"
        sanitize_response = sanitize_json(response)
        log_message(f"{source}'s sanitized response:\n{sanitize_response}", source, dataset, id)
        processing_date = datetime.now().strftime("%Y-%m-%d")
        processing_hour = datetime.now().strftime("%H:%M:%S")

        if evaluation_mode == "sentiment_analysis":
            sentiment, certainty, justification = parse_response(sanitize_response, evaluation_mode, dataset, source, id)
            log_message(f"Parsed response - Sentiment: {sentiment}, Certainty: {certainty}, Justification: {justification}, Processing_date: {processing_date}, Processing_hour: {processing_hour}\n", source, dataset, id)  
            return sentiment, certainty, justification, processing_date, processing_hour 
        else:
            emotion, certainty, justification = parse_response(sanitize_response, evaluation_mode, dataset, source, id)
            log_message(f"Parsed response - Emotion: {emotion}, Certainty: {certainty}, Justification: {justification}, Processing_date: {processing_date}, Processing_hour: {processing_hour}\n", source, dataset, id)  
            return emotion, certainty, justification, processing_date, processing_hour  
    
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON: {e}", source, dataset, id)
        return 'NA', 'NA', 'NA', 'NA', 'NA'