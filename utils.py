import re
import datetime
import json

LOG_FILE_CLAUDE = "claude_log.txt"
LOG_FILE_CHATGPT = "chatgpt_log.txt"
LOG_FILE_GEMINI = "gemini_log.txt"
LOG_FILE_DEEPSEEK = "deepseek_log.txt"

def log_message(msg, llm):
    # ES: Imprimir mensaje y guardar en archivo de log
    # EN: Print message and save to log file
    if llm == "claude":
        LOG_FILE = LOG_FILE_CLAUDE
    elif llm == "chatgpt":
        LOG_FILE = LOG_FILE_CHATGPT
    elif llm == "gemini":
        LOG_FILE = LOG_FILE_GEMINI
    elif llm == "deepseek":
        LOG_FILE = LOG_FILE_DEEPSEEK
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(msg + "\n")

# ES: Función para guardar en MongoDB
# EN: Function to save in MongoDB
def save_in_mongodb_from_X(colection, tweet, includes):
    # ES: Obtener la fecha y hora actual
    # EN: Get the current date and time
    now = datetime.datetime.now()
    formatted_date = now.strftime("%d/%m/%y")
    formatted_time = now.strftime("%H:%M:%S") 

    media = []
    if 'attachments' in tweet.data and 'media_keys' in tweet.data['attachments']:
        media_keys = tweet.data['attachments']['media_keys']  # ES: Lista de archivos multimedia
                                                              # EN: List of multimedia files
        media_objects = includes.get('media', [])
        for media_file in media_keys:
            media_url = None
            for media in media_objects:
                if media['media_key'] == media_file:
                    media_url = media['url']
                    if media_url is not None:
                        media.append(media_url) 

    # ES: si tiene algun archivo multimedia lo añade a la lista
    # EN: if it has any multimedia file it adds it to the list
    if tweet != None:
     if includes != None: 
        if 'media' in includes:
            if 'attachments' in tweet.data and 'media_keys' in tweet.attachments:
                for obj in includes['media']:
                    if obj['media_key'] in tweet.attachments['media_keys']:
                        media.append(obj)

    n_items = 1
    media_types = []
    media_keys = []
    if 'attachments' in tweet.data:
        n_items += len(tweet.attachments["media_keys"])
        media_keys = tweet.attachments['media_keys']


    # ES: Calcular tipos de media distintos
    # EN: Calculate different types of media
    if includes != None:
        if 'media' in includes:
            for media_item in includes['media']:
                # Asegurarse de que las claves están en el mismo formato (ambas como strings)
                if str(media_item['media_key']) in map(str, media_keys):
                    media_types.append(media_item['type'])  # Guardar el tipo de media

    # ES: Crear el documento con todos los campos, verificando la existencia de cada uno
    # EN: Create the document with all fields, checking the existence of each one
    message = tweet.text if 'text' in tweet.data else "None"
    message = sanitize_text(message)
    document = {
        'id_source': tweet.id if 'id' in tweet.data else "None",
        'message': message,
        'created_at': tweet.created_at if 'created_at' in tweet.data else "None",
        'language': tweet.lang if 'lang' in tweet.data else "None",
        'evaluation_date': formatted_date,  
        'evaluated_time': formatted_time,
        'source': "X" ,
        'n_items': n_items,
        'media_items': media,
        'n_media': len(media_types),  # ES: Número de tipos de archivos multimedia distintos
                                        # EN: Number of different types of multimedia files
        
    }
    colection.insert_one(document)

# ES: Función para guardar en MongoDB
# EN: Function to save to MongoDB
async def save_to_mongodb_from_Telegram(colection, id_message, chat, message,  media, justify, n_items, n_media, sender_id, dateCreated, lang):
   
    
    # ES: Establecer fecha y hora actuales
    # EN: Set current date and time
    now = datetime.datetime.now()
    formatted_date = now.strftime("%d/%m/%y")
    formatted_time = now.strftime("%H:%M:%S") 

    #ES: Procesar texto y cambiar caracteres especiales
    #EN: Process text and change special characters
    message = message.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and").replace("&", "and").replace(";", " ")

    #ES: Procesar justificación y cambiar caracteres especiales
    #EN: Process justification and change special characters

    document = {
        'id_source': id_message,
        'chat': chat,
        'message': message,
        'lang': lang,
        'evaluation_date': formatted_date,
        'evaluation_time': formatted_time,
        'created_at': dateCreated,
        'author_id': sender_id,
        'source': "Telegram",
        'n_items': n_items,
        'n_media': n_media,

    }

    # ES: Insertar el documento en la colección de MongoDB
    # EN: Insert the document into the MongoDB collection
    colection.insert_one(document)

# ES: Función para guardar en MongoDB el resultado final
# EN: Function to save the final result in MongoDB
def save_in_mongodb_final_csv_X(collection, tweet, sentiment, certainty, justification, n_media, n_items_used, id_source, created_at, author_id, date, time, source, n_items, n_likes, n_retweets, n_impressions, n_replies, n_quotes, n_bookmarks, language):

    #ES: Procesar texto y cambiar caracteres especiales
    #EN: Process text and change special characters
    tweet = tweet.replace("\n", " ")
    tweet = tweet.replace("\r", " ")
    tweet = tweet.replace("\t", " ")
    tweet = tweet.replace("&amp", "and ")
    tweet = tweet.replace("&", "and")
    tweet = tweet.replace(";", " ")
    tweet = tweet.replace(":", " ")

    #ES: Porcesar la justificacion de la respuesta y cambiar caracteres especiales
    #EN: Process the justification of the response and change special characters
    if justification:
        justification = justification.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and ").replace("&", "and").replace(";", " ")

   
    collection.insert_one({
            "id_source": id_source,
            "message": tweet,
            "created_at": created_at,
            "lang": language,
            "author_id": author_id,
            "sentiment": sentiment,
            "certainty": certainty,
            "justification": justification,
            "date": date,
            "time": time,
            "source": source,
            "n_items": n_items,
            "n_media": n_media,
            "n_items_decision": n_items_used,
            "n_likes": n_likes,
            "n_retweets": n_retweets,
            "n_impressions": n_impressions,
            "n_replies": n_replies,
            "n_quotes": n_quotes,
            "n_bookmarks": n_bookmarks
        })

def save_in_mongodb_final_csv_Telegram(colection, id_message, chat, message,  media, n_items, n_media, sender_id, dateCreated, lang, justification, n_items_used, sentiment, certainty):
    #ES: Procesar texto y cambiar caracteres especiales
    #EN: Process text and change special characters
    message = message.replace("\n", " ")
    message = message.replace("\r", " ")
    message = message.replace("\t", " ")    
    message = message.replace("&amp", "and ")
    message = message.replace("&", "and")
    message = message.replace(";", " ")
    message = message.replace(":", " ")

    # ES: Establecer fecha y hora actuales
    # EN: Set current date and time
    now = datetime.datetime.now()
    formatted_date = now.strftime("%d/%m/%y")
    formatted_time = now.strftime("%H:%M:%S") 
    
    if justification:
        justification = justification.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("&amp", "and").replace("&", "and").replace(";", " ")

    #ES: Procesar justificación y cambiar caracteres especiales
    #EN: Process justification and change special characters

    document = {
        'id_source': id_message,
        'chat': chat,
        'message': message,
        'lang': lang,
        'evaluation_date': formatted_date,
        'evaluation_time': formatted_time,
        'created_at': dateCreated,
        'author_id': sender_id,
        'source': "Telegram",
        'n_items': n_items,
        'n_media': n_media,
        'sentiment': sentiment,
        'certainty': certainty,
        'justification': justification,
        'n_items_decision': n_items_used

    }

    # ES: Insertar el documento en la colección de MongoDB
    # EN: Insert the document into the MongoDB collection
    colection.insert_one(document)
    

  

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