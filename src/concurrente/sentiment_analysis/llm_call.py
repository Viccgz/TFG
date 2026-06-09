import json
import threading
import openai
import utils
import google.generativeai as genai
from openai import OpenAI
from dataset_preprocessing.emotion_mapper import GO_EMOTIONS_LABELS, normalize_emotion_label
from mistralai.client import Mistral
import time

mistral_semaphore = threading.Semaphore(1)

with open('config.json') as config_file:
    config = json.load(config_file)

# ES: Credenciales de la API de OpenAI
# EN: OpenAI API credentials
OPEN_AI_KEY_SECRET = config['OPEN_AI_KEY_SECRET']

clientMistral = Mistral(api_key=config["mistral_api_key"])

# ES: Credenciales de la API de Gemini
# EN: Gemini API credentials
genai.configure(api_key= config['genai_api_key'])
generation_config = {"temperature": 0.0, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}
modelGemini = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)


# ES: Credenciales de la API de Deepseek
# EN: Deepseek API credentials
clientDeepseek = OpenAI(api_key= config['deepseek_api_key'], base_url="https://api.deepseek.com")

# ES: Pompts analisis de sentimiento
# EN: Sentiment analysis prompts
sentiment_prompt = "From the data provided, classify the sentiment of the text as positive, negative, or neutral. You must use the following values for the sentiment key: positive, negative, or neutral. The third key IS certainty, NOT certainly.\n "
sentiment_justify_prompt = "Return the result as a JSON object with the following keys: sentiment, justification, and certainty. Format example: {\"sentiment\": \"negative\", \"justification\": \"The announcement ...\", \"certainty\": \"90%\"}. The text to analyze is: \n "
sentiment_not_justify_prompt = "Return the result as a JSON object with the following keys: sentiment and certainty. Format example: {\"sentiment\": \"positive\", \"certainty\": \"90%\"}\n Avoid returning anyyhing else than the json object. The text to analyze is: \n "

# ES: Pompts analisis de emociones
# EN: Emotion analysis prompts
goemotions_labels = sorted(GO_EMOTIONS_LABELS)
emotions_str = ", ".join(goemotions_labels)

emotion_prompt = f"From the data provided, you MUST choose ONLY one emotion of the following categories: [{emotions_str}]. DO NOT INVENT new emotions, that is forbidden. The third key IS certainty, NOT certainly.\n "
emotion_justify_prompt = "Return the result as a JSON object with the following keys: emotion, justification, and certainty. Format example: {\"emotion\": \"anger\", \"justification\": \"The announcement ...\", \"certainty\": \"90%\"}. The text to analyze is: \n "
emotion_not_justify_prompt = "Return the result as a JSON object with the following keys: emotion and certainty. Format example: {\"emotion\": \"anger\", \"certainty\": \"90%\"}\n Avoid returning anyyhing else than the json object. The text to analyze is: \n"

emotion_and_sentiment_justify_prompt = "Taking into account the sentiment retrieved from the previous analysis you must do the following:\n The sentiment was :"



# ES: Función para enviar los tweets a la API de OpenAI para generar una respuesta positiva o negativa
# EN: Function to send tweets to OpenAI API to generate a positive or negative response
def send_to_chatgpt(text, justify, evaluation_mode, dataset, id, sentiment=None):
    """
    ES: Envía el texto a la API de ChatGPT para obtener una clasificación de sentimiento o emoción, junto con una justificación y un nivel de certeza.
    EN: Sends the text to the ChatGPT API to obtain a sentiment or emotion classification, along with a justification and a certainty level.
    Args:        
    text (str): El texto a analizar.
        justify (bool): Si se debe incluir una justificación en la respuesta.
        evaluation_mode (str): El modo de evaluación (sentiment_analysis o emotion_analysis).
        dataset (str): El nombre del dataset.
        id (str): El ID del mensaje.
        sentiment (str): El sentimiento previamente obtenido (solo para análisis de emoción con sentimiento).
    Returns:
        Si evaluation_mode es "sentiment_analysis": (sentiment, certainty, justification, processing_date, processing_hour)"
        Si evaluation_mode es "emotion_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
        Si evaluation_mode es "emotion_with_sentiment_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
    """
    # ES: Credenciales de la API de ChatGPT
    # EN: ChatGPT API credentials
    openai.api_key = OPEN_AI_KEY_SECRET
    
    # ES: Generar respuesta inicial de ChatGPT
    # EN: Generate initial ChatGPT response
    user_prompt = ""
    if evaluation_mode == "sentiment_analysis":
        user_prompt = sentiment_prompt + (sentiment_justify_prompt if justify else sentiment_not_justify_prompt) + text
        target_key = "sentiment"
        allowed_labels = ["positive", "negative", "neutral"]
    elif evaluation_mode == "emotion_with_sentiment_analysis":
        user_prompt = emotion_and_sentiment_justify_prompt + sentiment + "\n" + emotion_prompt + (emotion_justify_prompt if justify else emotion_not_justify_prompt) + text
        target_key = "emotion"
        allowed_labels = goemotions_labels
    else:
        user_prompt = emotion_prompt + (emotion_justify_prompt if justify else emotion_not_justify_prompt) + text
        target_key = "emotion"
        allowed_labels = goemotions_labels

    required_keys = [target_key, "certainty"]
    if justify:
        required_keys.append("justification")
        
    system_instruction = f"You are an expert data classification API. You MUST return a valid JSON object with these exact keys: {required_keys}. For the '{target_key}' key, you are strictly restricted to these values: {allowed_labels}."

    response = openai.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            stream=False
        )
    response_json = response.choices[0].message.content.strip()
    utils.log_message(f"ChatGPT's raw response:\n{response_json}", "chatgpt", dataset, id)

    # ES: Verifico si la respuesta JSON no está vacía
    # EN: Check if the JSON response is not empty
    if not response_json:
        utils.log_message("Null response received.", "chatgpt", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'
        
    try:
        utils.log_message(f"ChatGPT's sanitized response:\n{response_json}", "chatgpt", dataset, id)
        sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(response_json, "chatgpt", evaluation_mode, dataset, id)
        
        # ES: Si es análisis de emociones, mapear la emoción y devolver ambas
        # EN: If it's emotion analysis, map the emotion and return both
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            emotion_raw = sentiment
            emotion_mapped = normalize_emotion_label(emotion_raw)
            return emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour
        else:
            return sentiment, certainty, justification, processing_date, processing_hour


    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "chatgpt", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'
 
def send_to_deepseek(text, justify, evaluation_mode, dataset, id, sentiment=None):
    """
    ES: Envía el texto a la API de Deepseek para obtener una clasificación de sentimiento o emoción, junto con una justificación y un nivel de certeza.
    EN: Sends the text to the Deepseek API to obtain a sentiment or emotion classification, along with a justification and a certainty level.
    Args:        
    text (str): El texto a analizar.
        justify (bool): Si se debe incluir una justificación en la respuesta.
        evaluation_mode (str): El modo de evaluación (sentiment_analysis o emotion_analysis).
        dataset (str): El nombre del dataset.
        id (str): El ID del mensaje.
        sentiment (str): El sentimiento previamente obtenido (solo para análisis de emoción con sentimiento).
    Returns:
        Si evaluation_mode es "sentiment_analysis": (sentiment, certainty, justification, processing_date, processing_hour)"
        Si evaluation_mode es "emotion_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
        Si evaluation_mode es "emotion_with_sentiment_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
    """
    # ES: Generar la respuesta inicial con Deepseek
    # EN: Generate the initial response with Deepseek
    user_prompt = ""
    if evaluation_mode == "sentiment_analysis":
        user_prompt = sentiment_prompt + (sentiment_justify_prompt if justify else sentiment_not_justify_prompt) + text
        target_key = "sentiment"
        allowed_labels = ["positive", "negative", "neutral"]
    elif evaluation_mode == "emotion_with_sentiment_analysis":
        user_prompt = emotion_and_sentiment_justify_prompt + sentiment + "\n" + emotion_prompt + (emotion_justify_prompt if justify else emotion_not_justify_prompt) + text
        target_key = "emotion"
        allowed_labels = goemotions_labels
    else:
        user_prompt = emotion_prompt + (emotion_justify_prompt if justify else emotion_not_justify_prompt) + text
        target_key = "emotion"
        allowed_labels = goemotions_labels

    required_keys = [target_key, "certainty"]
    if justify:
        required_keys.append("justification")
        
    system_instruction = f"You are an expert data classification API. You MUST return a valid JSON object with these exact keys: {required_keys}. For the '{target_key}' key, you are strictly restricted to these values: {allowed_labels}."

    response = clientDeepseek.chat.completions.create(
            model="deepseek-v4-flash", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            stream=False
        )
    output = response.choices[0].message.content.strip()
    
    # ES: Verifico si la respuesta JSON no está vacía
    # EN: Check if the JSON response is not empty
    if not output:
        utils.log_message("Null response received.", "deepseek", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'
    try:
        # ES: Procesar la respuesta de Deepseek        
        # # EN: Process the Deepseek response
        utils.log_message(f"Deepseek's raw response:\n{output}", "deepseek", dataset, id)
        sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(output, "deepseek", evaluation_mode, dataset, id)
        
        # ES: Si es análisis de emociones, mapear la emoción y devolver ambas
        # EN: If it's emotion analysis, map the emotion and return both
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            emotion_raw = sentiment
            emotion_mapped = normalize_emotion_label(emotion_raw)
            return emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour
        else:
            return sentiment, certainty, justification, processing_date, processing_hour
    
    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "deepseek", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'
    except Exception as e:
        utils.log_message(f"Error processing response: {e}", "deepseek", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
                return 'NA', 'NA', 'NA', 'NA', 'NA'


def send_to_gemini(text, justify, evaluation_mode, dataset, id, sentiment=None):
    """
    ES: Envía el texto a la API de Gemini para obtener una clasificación de sentimiento o emoción, junto con una justificación y un nivel de certeza.
    EN: Sends the text to the Gemini API to obtain a sentiment or emotion classification, along with a justification and a certainty level.
    Args:        
    text (str): El texto a analizar.
        justify (bool): Si se debe incluir una justificación en la respuesta.
        evaluation_mode (str): El modo de evaluación (sentiment_analysis o emotion_analysis).
        dataset (str): El nombre del dataset.
        id (str): El ID del mensaje.
        sentiment (str): El sentimiento previamente obtenido (solo para análisis de emoción con sentimiento).
    Returns:
        Si evaluation_mode es "sentiment_analysis": (sentiment, certainty, justification, processing_date, processing_hour)"
        Si evaluation_mode es "emotion_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
        Si evaluation_mode es "emotion_with_sentiment_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
    """
    # ES: Generar la respuesta inicial con Gemini
    # EN: Generate the initial response with Gemini
    try:
        if evaluation_mode == "sentiment_analysis":
            prompt = ( sentiment_prompt + sentiment_justify_prompt + text) if justify else (sentiment_prompt + sentiment_not_justify_prompt + text)
        elif evaluation_mode == "emotion_analysis":
            prompt = ( emotion_prompt + emotion_justify_prompt + text) if justify else (emotion_prompt + emotion_not_justify_prompt + text )
        elif evaluation_mode == "emotion_with_sentiment_analysis":
            prompt = ( emotion_and_sentiment_justify_prompt + sentiment + "\n" + emotion_prompt + emotion_justify_prompt + text) if justify else ( emotion_and_sentiment_justify_prompt + sentiment + "\n" + emotion_prompt + emotion_not_justify_prompt + text )

        response = modelGemini.generate_content(prompt).text.strip()
        utils.log_message(f"Gemini's raw response:\n{response}", "gemini", dataset, id)
        if not response:
            utils.log_message("Null response received.", "gemini", dataset, id)
            if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
                return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
            else:
                return 'NA', 'NA', 'NA', 'NA', 'NA'
        
        sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(response, "gemini", evaluation_mode, dataset, id)
        
        # ES: Si es análisis de emociones, mapear la emoción y devolver ambas
        # EN: If it's emotion analysis, map the emotion and return both
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            emotion_raw = sentiment
            emotion_mapped = normalize_emotion_label(emotion_raw)
            return emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour
        else:
            return sentiment, certainty, justification, processing_date, processing_hour

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "gemini", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'
    except Exception as e:
        utils.log_message(f"Error processing response: {e}", "gemini", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'

def send_to_mistral(text, justify, evaluation_mode, dataset, id, sentiment=None):
    """
    ES: Envía el texto a la API de Mistral para obtener una clasificación de sentimiento o emoción, junto con una justificación y un nivel de certeza.
    EN: Sends the text to the Mistral API to obtain a sentiment or emotion classification, along with a justification and a certainty level.
    Args:        
    text (str): El texto a analizar.
        justify (bool): Si se debe incluir una justificación en la respuesta.
        evaluation_mode (str): El modo de evaluación (sentiment_analysis o emotion_analysis).
        dataset (str): El nombre del dataset.
        id (str): El ID del mensaje.
        sentiment (str): El sentimiento previamente obtenido (solo para análisis de emoción con sentimiento).
    Returns:
        Si evaluation_mode es "sentiment_analysis": (sentiment, certainty, justification, processing_date, processing_hour)"
        Si evaluation_mode es "emotion_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
        Si evaluation_mode es "emotion_with_sentiment_analysis": (emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour)
    """
    max_retries = 6
    for attempt in range(max_retries):
        try:
            with mistral_semaphore:
                time.sleep(1)  # Espera fija de 1 segundo entre solicitudes para evitar el rate limit --> daba problemas por ir demasiado rápido, aunque el rate limit es de 20 rpm, con 1.5s entre solicitudes debería ser suficiente para no recibir 429.
                response = clientMistral.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": ((sentiment_prompt + sentiment_justify_prompt + text) if justify else (sentiment_prompt + sentiment_not_justify_prompt + text)) 
                           if evaluation_mode == "sentiment_analysis" else ((emotion_prompt + emotion_justify_prompt + text) if justify 
                           else (emotion_prompt + emotion_not_justify_prompt + text)) if evaluation_mode != "emotion_with_sentiment_analysis" 
                           else ((emotion_and_sentiment_justify_prompt + sentiment + "\n" + emotion_prompt + emotion_justify_prompt + text) if justify 
                           else (emotion_and_sentiment_justify_prompt + sentiment + "\n" + emotion_prompt + emotion_not_justify_prompt + text))
                        }
                    ]
                )
            output = response.choices[0].message.content.strip()
            break  

        except Exception as e:
            retryable_errors = ["429", "rate_limited", "500", "502", "503", "504", "timeout", "timed out", "ReadTimeout", "Service unavailable", "ConnectTimeout"]
            if any(error in str(e) for error in retryable_errors):
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s, 32s
                utils.log_message(f"Error de Mistral detectado {e}, reintentando en {wait}s (intento {attempt+1}/{max_retries})", "mistral", dataset, id)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    utils.log_message("Max retries alcanzado.", "mistral", dataset, id)
                    if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
                        return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
                    else:
                        return 'NA', 'NA', 'NA', 'NA', 'NA'
            else:
                utils.log_message(f"Error inesperado de Mistral: {e}", "mistral", dataset, id)
                raise

    if not output:
        utils.log_message("Null response received.", "mistral", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'
    try:
        utils.log_message(f"Mistral's raw response:\n{output}", "mistral", dataset, id)
        sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(output, "mistral", evaluation_mode, dataset, id)

        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            emotion_raw = sentiment
            emotion_mapped = normalize_emotion_label(emotion_raw)
            return emotion_raw, emotion_mapped, certainty, justification, processing_date, processing_hour
        else:
            return sentiment, certainty, justification, processing_date, processing_hour

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "mistral", dataset, id)
        if evaluation_mode == "emotion_analysis" or evaluation_mode == "emotion_with_sentiment_analysis":
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        else:
            return 'NA', 'NA', 'NA', 'NA', 'NA'

# ES: Registro de LLMs disponibles
# EN: Registry of available LLMs
# Para añadir un nuevo LLM:
# 1. Añadir la clave API en config.json
# 2. Implementar la función send_to_newllm(text, justify, evaluation_mode, dataset)
# 3. Añadir 'NEWLLM': send_to_newllm al diccionario LLM_FUNCTIONS
# 4. Si la clave API está presente, se añadirá automáticamente a AVAILABLE_LLMS
#
# To add a new LLM:
# 1. Add the API key in config.json
# 2. Implement the function send_to_newllm(text, justify, evaluation_mode, dataset)
# 3. Add 'NEWLLM': send_to_newllm to the LLM_FUNCTIONS dictionary
# 4. If the API key is present, it will be automatically added to AVAILABLE_LLMS
LLM_FUNCTIONS = {
    'CHATGPT': send_to_chatgpt,
    'GEMINI': send_to_gemini,
    'DEEPSEEK': send_to_deepseek,
    'MISTRAL': send_to_mistral  # Placeholder para Mistral, implementar send_to_mistral y asignar aquí
}

# ES: Lista de LLMs disponibles
# EN: List of available LLMs
AVAILABLE_LLMS = []
if config.get('OPEN_AI_KEY_SECRET'):
    AVAILABLE_LLMS.append('CHATGPT')
if config.get('genai_api_key'):
    AVAILABLE_LLMS.append('GEMINI')
if config.get('deepseek_api_key'):
    AVAILABLE_LLMS.append('DEEPSEEK')
if config.get('mistral_api_key'):
    AVAILABLE_LLMS.append('MISTRAL')

