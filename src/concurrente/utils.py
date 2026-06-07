from datetime import datetime
from logging import log
import os
import re
import json

# ----------------LOGGING UTILS----------------

# ES: Plantillas de rutas para los archivos de log de cada LLM. Se formatean con el nombre del dataset (goemotions, kaggle o isear)
#     Para añadir un nuevo LLM, simplemente hay que añadir una nueva entrada a este diccionario con el nombre del LLM en minúsculas como clave y la plantilla de ruta como valor. La plantilla de ruta debe contener "{dataset}" donde se quiera que aparezca el nombre del dataset en el log.

# EN: Path templates for the log files of each LLM. They are formatted with the name of the dataset (goemotions, kaggle, or isear)
#     To add a new LLM, simply add a new entry to this dictionary with the name of the LLM in lowercase as the key and the path template as the value. The path template should contain "{dataset}" where you want the dataset name to appear in the log.
LOG_PLANTILLAS = {
    "chatgpt": "LOGS/{dataset}_chatgpt_log.txt",
    "gemini": "LOGS/{dataset}_gemini_log.txt",
    "deepseek": "LOGS/{dataset}_deepseek_log.txt",
    "mistral": "LOGS/{dataset}_mistral_log.txt"
}

def log_message(msg, llm, dataset, id):
    """ES: Función para registrar un mensaje en el log correspondiente al LLM y dataset. El mensaje se formatea con la fecha y hora actual, el mensaje, y el ID de la muestra procesada. Si el archivo de log no existe, se crea automáticamente.
    EN: Function to log a message in the log corresponding to the LLM and dataset. The message is formatted with the current date and time, the message, and the ID of the processed sample. If the log file does not exist, it is created automatically.
    Args:
        msg: El mensaje a registrar en el log
        llm: El nombre del LLM (en minúsculas) que procesó la muestra (ej. "chatgpt", "gemini", "deepseek", "mistral")
        dataset: El nombre del dataset (en minúsculas) que se está procesando (ej. "goemotions", "kaggle", "isear")
        id: El ID de la muestra que se está procesando, para poder rastrear el mensaje en el log
    """
    template = LOG_PLANTILLAS.get(llm.lower())
    if not template:
        print(f"Error: LLM '{llm}' invalido con identificador {id}. No se registrará el mensaje: {msg}")
        return
    log_file_path = template.format(dataset=dataset.upper())
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} (ID: {id})\n")
    except Exception as e:
        print(f"No se pudo escribir en el log: {e}")


# ----------------DB UTILS----------------

# ES: Función para guardar en MongoDB el resultado final
# EN: Function to save the final result in MongoDB
def save_in_mongodb_final_csv(collection, message, sentiment, certainty, justification, processing_date, processing_hour, emotion_mapped_llm, certainty_emotion, justification_emotion, date_emotion, time_emotion, emotion_raw_llm, id, emotion_raw_gt, emotion_mapped_gt, emotion_with_sentiment_raw_llm, emotion_mapped_with_sentiment, certainty_emotion_with_sentiment, justification_emotion_with_sentiment, date_emotion_with_sentiment, time_emotion_with_sentiment):

    #ES: Porcesar la justificacion de la respuesta y cambiar caracteres especiales
    #EN: Process the justification of the response and change special characters
    if justification:
        justification = sanitize_text(justification)

    if message:
        message = sanitize_text(message)

    try:
        collection.insert_one({
                "id": id,
                "text": message,
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
                "time_emotion": time_emotion,
                "emotion_with_sentiment_raw_llm": emotion_with_sentiment_raw_llm,
                "emotion_mapped_with_sentiment": emotion_mapped_with_sentiment,
                "certainty_emotion_with_sentiment": certainty_emotion_with_sentiment,
                "justification_emotion_with_sentiment": justification_emotion_with_sentiment,
                "date_emotion_with_sentiment": date_emotion_with_sentiment,
                "time_emotion_with_sentiment": time_emotion_with_sentiment
            })
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")



# ----------------TEXT PROCESSING UTILS----------------
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
    """
    ES: Procesa el texto de entrada, eliminando saltos de línea, tabulaciones y reemplazando caracteres especiales como "&" por "and". Esto es útil para limpiar el texto antes de procesarlo o guardarlo en la base de datos.
    EN: Processes the input text, removing line breaks, tabs, and replacing special characters like "&" with "and". This is useful for cleaning the text before processing it or saving it to the database.
    Args:
        text: El texto a procesar
    Returns:
            El texto procesado, con saltos de línea y tabulaciones eliminados, y caracteres especiales reemplazados
    """
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and ").replace("&", "and").replace(";", " ").replace(":", " ")
    return text

# ----------------RESPONSE PROCESSING UTILS----------------
def parse_response(response, evaluation_mode, dataset, source, id):
    """
    ES: Parsea la respuesta del LLM, extrayendo la emoción o sentimiento, la certeza y la justificación. Si la respuesta no es un JSON válido o no contiene los campos esperados, se registrará un mensaje de error en el log y se devolverán valores por defecto.
    EN: Parses the LLM response, extracting the emotion or sentiment, certainty, and justification. If the response is not valid JSON or does not contain the expected fields, an error message will be logged and default values will be returned.
    Args:
        response: La respuesta del LLM a parsear, que se espera que sea un JSON con los campos "sentiment" o "emotion", "certainty" y "justification"
        evaluation_mode: El modo de evaluación, que puede ser "sentiment_analysis" o "emotion_recognition". Esto determina si se espera un campo "sentiment" o "emotion" en la respuesta.
        dataset: El nombre del dataset que se está procesando, para registrar en el log
        source: El nombre del LLM que generó la respuesta, para registrar en el log
        id: El ID de la muestra que se está procesando, para registrar en el log
    Returns:
        Si evaluation_mode es "sentiment_analysis", devuelve una tupla (sentiment, certainty, justification) donde sentiment es la etiqueta de sentimiento extraída de la respuesta, certainty es el valor de certeza extraído, y justification es la justificación extraída.
        Si evaluation_mode es "emotion_recognition", devuelve una tupla (emotion, certainty, justification) donde emotion es la etiqueta de emoción extraída de la respuesta, certainty es el valor de certeza extraído, y justification es la justificación extraída.
        En caso de error, devuelve ("unknown", 0.0, "Error parsing response") para "sentiment_analysis" o ("unknown", 0.0, "Error parsing response") para "emotion_recognition"
    """
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