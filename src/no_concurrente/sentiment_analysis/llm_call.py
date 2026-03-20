import json
import openai
import utils
import google.generativeai as genai
from openai import OpenAI

with open('config.json') as config_file:
    config = json.load(config_file)

# ES: Credenciales de la API de OpenAI
# EN: OpenAI API credentials
OPEN_AI_KEY_SECRET = config['OPEN_AI_KEY_SECRET']

# ES: Credenciales de la API de Gemini
# EN: Gemini API credentials
genai.configure(api_key= config['genai_api_key'])
generation_config = {"temperature": 0.9, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}
modelGemini = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)


# ES: Credenciales de la API de Deepseek
# EN: Deepseek API credentials
clientDeepseek = OpenAI(api_key= config['deepseek_api_key'], base_url="https://api.deepseek.com")

# ES: Pompts analisis de sentimiento
# EN: Sentiment analysis prompts
sentiment_prompt = "From the data provided, classify the sentiment of the text as positive, negative, or neutral. You must use the following values for the sentiment key: positive, negative, or neutral. The third key IS certainty, NOT certainly\n "
sentiment_justify_prompt = "Return the result as a JSON object with the following keys: sentiment, justification, and certainty. Format example: {\"sentiment\": \"negative\", \"justification\": \"The announcement ...\", \"certainty\": \"90%\"}\n "
sentiment_not_justify_prompt = "Return the result as a JSON object with the following keys: sentiment and certainty. Format example: {\"sentiment\": \"positive\", \"certainty\": \"90%\"}\n "

# ES: Pompts analisis de emociones
# EN: Emotion analysis prompts
emotion_prompt = "From the data provided, classify the emotion the text as for example Happiness, sadness, anger, etc or indeterminable. The third key IS certainty, NOT certainly\n "
emotion_justify_prompt = "Return the result as a JSON object with the following keys: emotion, justification, and certainty. Format example: {\"emotion\": \"anger\", \"justification\": \"The announcement ...\", \"certainty\": \"90%\"}\n "
emotion_not_justify_prompt = "Return the result as a JSON object with the following keys: emotion and certainty. Format example: {\"emotion\": \"anger\", \"certainty\": \"90%\"}\n "


# ES: Función para enviar los tweets a la API de OpenAI para generar una respuesta positiva o negativa
# EN: Function to send tweets to OpenAI API to generate a positive or negative response
def send_to_chatgpt(text, justify, evaluation_mode):
    # ES: Credenciales de la API de ChatGPT
    # EN: ChatGPT API credentials
    openai.api_key = OPEN_AI_KEY_SECRET
    
    # ES: Generar respuesta inicial de ChatGPT
    # EN: Generate initial ChatGPT response
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": ((text + sentiment_prompt + sentiment_justify_prompt) if justify else (text + sentiment_prompt + sentiment_not_justify_prompt)) 
                           if evaluation_mode == "sentiment_analysis" else ((text + emotion_prompt + emotion_justify_prompt) if justify 
                                                                      else (text + emotion_prompt + emotion_not_justify_prompt))
                    }
                ],
                stream=False
    )
    response_json = response.choices[0].message.content.strip()
    utils.log_message(f"ChatGPT's raw response:\n{response_json}", "chatgpt") 

    # ES: Verifico si la respuesta JSON no está vacía
    # EN: Check if the JSON response is not empty
    if not response_json:
        utils.log_message("Null response received.", "chatgpt")
        return 'NA'
        
    try:
        utils.log_message(f"ChatGPT's sanitized response:\n{response_json}", "chatgpt")
        if justify:
            sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(response_json, "chatgpt", justify, evaluation_mode)
            return sentiment, certainty, justification, processing_date, processing_hour
        else:
            sentiment, certainty, processing_date, processing_hour = utils.process_response(response_json, "chatgpt", justify, evaluation_mode)
            return sentiment, certainty, processing_date, processing_hour

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "chatgpt")
        return 'NA'
 
def send_to_deepseek(text, justify, evaluation_mode):

    # ES: Generar la respuesta inicial con Deepseek
    # EN: Generate the initial response with Deepseek
    response = clientDeepseek.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": ((text + sentiment_prompt + sentiment_justify_prompt) if justify else (text + sentiment_prompt + sentiment_not_justify_prompt)) 
                           if evaluation_mode == "sentiment_analysis" else ((text + emotion_prompt + emotion_justify_prompt) if justify 
                                                                      else (text + emotion_prompt + emotion_not_justify_prompt))
                    }
                ],
                stream=False
    )
    output = response.choices[0].message.content.strip()
    
    # ES: Verifico si la respuesta JSON no está vacía
    # EN: Check if the JSON response is not empty
    if not output:
        utils.log_message("Null response received.", "deepseek")
        return 'NA'
    try:
        # ES: Procesar la respuesta de Deepseek        
        # # EN: Process the Deepseek response
        utils.log_message(f"Deepseek's raw response:\n{output}", "deepseek")
        if justify:
            sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(output, "deepseek", justify, evaluation_mode)
            return sentiment, certainty, justification, processing_date, processing_hour
        else:
            sentiment, certainty, processing_date, processing_hour = utils.process_response(output, "deepseek", justify, evaluation_mode)
            return sentiment, certainty, processing_date, processing_hour
    
    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "deepseek")
        return 'NA'


def send_to_gemini(text, justify, evaluation_mode):
    # ES: Generar la respuesta inicial con Gemini
    # EN: Generate the initial response with Gemini
    try:
        if evaluation_mode == "sentiment_analysis":
            prompt = ( text + sentiment_prompt + sentiment_justify_prompt) if justify else (text + sentiment_prompt + sentiment_not_justify_prompt)
        else:
            prompt = ( text + emotion_prompt + emotion_justify_prompt) if justify else (text + emotion_prompt + emotion_not_justify_prompt)

        response = modelGemini.generate_content(prompt).text.strip()
        utils.log_message(f"Gemini's raw response:\n{response}", "gemini")
        if not response:
            utils.log_message("Null response received.", "gemini")
            return 'NA'
        
        if justify:
            sentiment, certainty, justification, processing_date, processing_hour = utils.process_response(response, "gemini", justify, evaluation_mode)
            return sentiment, certainty, justification, processing_date, processing_hour
        else:
             sentiment, certainty, processing_date, processing_hour = utils.process_response(response, "gemini", justify, evaluation_mode)
             return sentiment, certainty, processing_date, processing_hour

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "gemini")
        return 'NA'
    except Exception as e:
        utils.log_message(f"Error processing response: {e}", "gemini")
        return 'NA'

