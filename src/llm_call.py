import json
import openai
import utils
import anthropic
import google.generativeai as genai
import time
from llamaapi import LlamaAPI
from openai import OpenAI

# ES: Credenciales de la API de Twitter
# EN: Twitter API credentials
with open('config.json') as config_file:
    config = json.load(config_file)

API_KEY = config['API_KEY']
API_KEY_SECRET = config['API_KEY_SECRET']
ACCESS_TOKEN = config['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = config['ACCESS_TOKEN_SECRET']
BEARER_TOKEN = config['BEARER_TOKEN']

# ES: Credenciales de la API de OpenAI
# EN: OpenAI API credentials
OPEN_AI_KEY_SECRET = config['OPEN_AI_KEY_SECRET']

# ES: Credenciales de la API de Claude
# EN: Claude API credentials
claude_api_key = config["claude_api_key"]
client = anthropic.Anthropic(api_key=claude_api_key)

# ES: Credenciales de la API de Llama
# EN: Llama API credentials
llama = LlamaAPI(config['llama_api_key'])

# ES: Credenciales de la API de Gemini
# EN: Gemini API credentials
genai.configure(api_key= config['genai_api_key'])
generation_config = {"temperature": 0.9, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}
modelGemini = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)


# ES: Credenciales de la API de Deepseek
# EN: Deepseek API credentials
clientDeepseek = OpenAI(api_key= config['deepseek_api_key'], base_url="https://api.deepseek.com")

# ES: Pompts
# EN: Prompts
prompt = "From the data provided, classify the emotion the text as for example Happiness, sadness, anger, etc or indeterminable. The third key IS certainty, NOT certainly\n "
justify_prompt = "Return the result as a JSON object with the following keys: emotion, justification, and certainty. Format example: {\"emotion\": \"anger\", \"justification\": \"The announcement ...\", \"certainty\": \"90%\"}\n "
not_justify_prompt = "Return the result as a JSON object with the following keys: emotion and certainty. Format example: {\"emotion\": \"anger\", \"certainty\": \"90%\"}\n "
reevaluate_prompt= "Reevaluate the emotion of this content. The initial evaluation was neutral or indeterminable. Use the text and the provided url of the image to help make a better decision.\n"
    

# ES: Función para enviar los tweets a la API de OpenAI para generar una respuesta positiva o negativa
# EN: Function to send tweets to OpenAI API to generate a positive or negative response
def send_to_chatgpt(tweet, media, justification, n_media, n_items_used):
    # ES: Credenciales de la API de ChatGPT
    # EN: ChatGPT API credentials
    openai.api_key = OPEN_AI_KEY_SECRET
    text = tweet.text + '\n'
    
    # ES: Generar respuesta inicial de ChatGPT
    # EN: Generate initial ChatGPT response
    if justification:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text + prompt + justify_prompt}
            ],   
            max_tokens=200
        )
    else:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text + prompt+ not_justify_prompt}
            ],   
            max_tokens=200
        )

    response_json = response.choices[0].message.content.strip()
    utils.log_message(f"ChatGPT's raw response:\n{response_json}", "chatgpt") 

    # ES: Verifico si la respuesta JSON no está vacía
    # EN: Check if the JSON response is not empty
    if not response_json:
        utils.log_message("Null response received.", "chatgpt")
        return 'NA'
        
    try:
        sanitize_response = utils.sanitize_json(response_json)
        response_dict = json.loads(sanitize_response)
        sentiment = response_dict.get('sentiment', 'NA')
        certainty = response_dict.get('certainty', 'NA')
        certaintyInt = int(certainty[:-1])
        count = 0
        trustValue = 60  # ES: Valor de confianza para volver a enviar la imagen
                         # EN: Trust value to resend the image

        # ES: Si el sentimiento es "=", volver a enviar la imagen junto al texto para reevaluar la decisión
        # EN: If the sentiment is "=", resend the image along with the text to reevaluate
        while (sentiment == "=" or certaintyInt < trustValue) and (n_media>0):
            utils.log_message("Sentimiento neutral o indeterminable. Reenviando imagen para un análisis más detallado...", "chatgpt")
            if justification:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text + reevaluate_prompt + media[count] + justify_prompt}
                    ],
                    max_tokens=200
                )

            else:
                response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text + reevaluate_prompt + media[count] + not_justify_prompt}
                ],
                max_tokens=200
                )
                
            n_items_used += 1
            count += 1  
            n_media -= 1
            response_json = response.choices[0].message.content.strip()
            sanitize_response = utils.sanitize_json(response_json)
            utils.log_message(f"Reevaluated response:\n{sanitize_response}", "chatgpt")
            response_dict = json.loads(sanitize_response)
            sentiment = response_dict.get('sentiment', 'NA')
            certainty = response_dict.get('certainty', 'NA')
            certaintyInt = int(certainty[:-1])

        if justification:
            justification = response_dict.get('justification', 'NA')
            return {'sentiment': sentiment, 'certainty': certainty, 'justification': justification, 'n_items_used': n_items_used}
        else:
            return {'sentiment': sentiment, 'certainty': certainty, 'n_items_used': n_items_used}

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "chatgpt")
        return 'NA'
    
def related_hashtags_chatgpt(word):
    # ES: Credenciales de la API de ChatGPT
    # EN: ChatGPT API credentials
    openai.api_key = OPEN_AI_KEY_SECRET
    text = word + '\n'
    response = openai.chat.completions.create(
            model= "gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                
                {"role": "user", "content": text + 
                 "From the data provided, give me a total of 5 related hashtags to search in Twitter. "
                 "Return ONLY the list, avoid returning the input followed by ':'"
                 "The list must include the input word."
                 "Format example: ['#eficiency', '#energy', '#energetic', '#non-renewable', '#renewable'] \n"}
                ],   
                 
            max_tokens=200
        )
    response_text = response.choices[0].message.content
    utils.log_message(f"ChatGPT's response:\n{response_text}", "chatgpt")
    response_text = response.choices[0].message.content.strip() 
        # ES: Mostrar la respuesta de ChatGPT
        # EN: Display ChatGPT's response
    
    try:
            # ES: Evaluar la respuesta como una lista de Python
            # EN: Evaluate the response as a Python list
            synonims = eval(response_text)
            if isinstance(synonims, list):
                synonims = [synonim.strip() for synonim in synonims]
                return synonims
            else:
                utils.log_message("Response's fromat is not a list.", "chatgpt")
                return []
    except Exception as e:
            utils.log_message(f"Error processing answer: {e}", "chatgpt")
            return []
    
# ES: Función para obtener palabras relacionadas con ChatGPT (para Telegram)
# EN: Function to obtain related words using ChatGPT (for Telegram)
def related_words_chatgpt(word):
    openai.api_key = OPEN_AI_KEY_SECRET
    text = word + '\n'
    answer = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text + 
                 "From the data provided, give me a total of 5 related words. "
                 "Return ONLY the list, avoid returning the input followed by ':'"}
            ],
            max_tokens=200
        )
    answer_text = answer.choices[0].message.content.strip()
    try:
        synonyms = eval(answer_text)
        if isinstance(synonyms, list):
            return [synonym.strip() for synonym in synonyms]
        else:
            print("The format of the answer is not a list.")
            return []
    except Exception as e:
        print(f"Error while processing the response: {e}")
        return []
    
# ES: Función para enviar los tweets a la API de OpenAI para generar una respuesta positiva o negativa
# EN: Function to send tweets to OpenAI API to generate a positive or negative response
def send_to_claude(tweet, media, justification, n_media, n_items_used):
    text = tweet + '\n'

    
    if justification:
        prompt = ( text + prompt + justify_prompt)
    else:
        prompt = ( text + prompt + not_justify_prompt)

    try:
        # ES: Llamada a la API de Claude
        # EN: Call to Claude's API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": prompt}
            ]

        )
        response_text = message.content[0].text
        utils.log_message (response_text, "claude")    
        response_text = utils.sanitize_json(response_text)
        utils.log_message(f"Sanitized response:\n{response_text}", "claude")
        
        # Verifica si la respuesta no está vacía
        if not response_text:
            utils.log_message("Null or empty response received.", "claude")
            return 'NA'
        response_dict = json.loads(response_text)
        sentiment = response_dict.get('sentiment', 'NA')
        certainty = response_dict.get('certainty', 'NA')

        certaintyInt = int(certainty[:-1])

        trustValue = 70  # ES: Valor de confianza para volver a enviar la imagen
                         # EN: Trust value to resend the image

        # ES: Reenvío para reevaluación si es necesario
        # EN: Resend for reevaluation if necessary
        count = 0
        while (sentiment == "=" or certaintyInt < trustValue) and n_media>0:
            utils.log_message("Indeterminable sentiment. Resending image for further evaluation...", "claude")

            if justification:
                reevaluation_prompt = (text + reevaluate_prompt + media[count] + justify_prompt)
            else:
                reevaluation_prompt = (text + reevaluate_prompt + media[count] + not_justify_prompt)
            
            reevaluation_message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0,
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": reevaluation_prompt}]}
                ]
            )
            
            response_json = reevaluation_message.content[0].text
            utils.log_message(f"Reevaluated response:\n{response_json}", "claude")
            response_dict = json.loads(response_json)
            n_items_used += 1
            sentiment = response_dict.get('sentiment', 'NA')
            certainty = response_dict.get('certainty', 'NA')
            certaintyInt = int(certainty[:-1])
            n_media -= 1
            count += 1

        
        if justification:
            justification = response_dict.get('justification', 'NA')
            return {'sentiment': sentiment, 'certainty': certainty, 'justification': justification, 'n_items_used': n_items_used}
        else:
            return {'sentiment': sentiment, 'certainty': certainty, 'n_items_used': n_items_used}

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "claude")
        return 'NA'
    except Exception as e:
        utils.log_message(f"Error processing answer: {e}", "claude")
        return 'NA'

def send_to_gemini(tweet, media, justification, n_media, n_items_used):
    text = tweet + '\n'

    # ES: Generar la respuesta inicial con Gemini
    # EN: Generate the initial response with Gemini
    try:
        if justification:
            prompt = ( text + prompt + justify_prompt)
        else:
            prompt = ( text + prompt + not_justify_prompt)
        response = modelGemini.generate_content(prompt).text.strip()
        utils.log_message(f"Gemini's raw response:\n{response}", "gemini")
        response_json = utils.sanitize_json(response)
        utils.log_message(f"Gemini's sanitized response:\n{response_json}", "gemini")
        

        # ES: Verificar si la respuesta es válida
        # EN: Check if the response is valid
        if not response_json:
            utils.log_message("Null response received.", "gemini")
            return 'NA'

        response_dict = json.loads(response_json)
        sentiment = response_dict.get('sentiment', 'NA')
        certainty = response_dict.get('certainty', 'NA')
        certaintyInt = int(certainty[:-1]) if certainty.endswith('%') else 0

        trustValue = 70  # ES: Valor de confianza para volver a enviar la imagen
                         # EN: Trust value to resend the image

        # ES: Si el sentimiento es "=" o certeza baja, volver a enviar para reevaluar
        # EN: If the sentiment is "=" or low certainty, resend to reevaluate
        count = 0
        while (sentiment == "=" or certaintyInt < trustValue) and n_media > 0:
            if justification:
                reevaluation_prompt = (text + reevaluate_prompt + media[count] + justify_prompt)
            else:
                reevaluation_prompt = (text + reevaluate_prompt + media[count] + not_justify_prompt)

            response = modelGemini.generate_content(reevaluation_prompt).text.strip()
            utils.log_message(f"Raw reevaluated response:\n{response}", "gemini")
            response_json = utils.sanitize_json(response)
            utils.log_message(f"Reevaluated response:\n{response_json}", "gemini")
            response_dict = json.loads(response_json)
            n_items_used += 1
            n_media -= 1
            count += 1
            sentiment = response_dict.get('sentiment', 'NA')
            certainty = response_dict.get('certainty', 'NA')
            certaintyInt = int(certainty[:-1]) if certainty.endswith('%') else 0
            time.sleep(5) # ES: Esperar 5 segundos para evitar exceder el límite de solicitudes (version gratuita, adaptar a tus necesidades propias si lo deseas)
                          # EN: Wait 5 seconds to avoid exceeding the request limit (free version, adapt to your own needs if you wish)

        if justification:
            justification_text = response_dict.get('justification', 'NA')
            return {'sentiment': sentiment, 'certainty': certainty, 'justification': justification_text, 'n_items_used': n_items_used}
        else:
            return {'sentiment': sentiment, 'certainty': certainty, 'n_items_used': n_items_used}

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "gemini")
        return 'NA'
    except Exception as e:
        utils.log_message(f"Error processing response: {e}", "gemini")
        return 'NA'

def send_to_llama(tweet, media, justification, n_media, n_items_used):
    text = tweet + '\n'
    trustValue = 70
    

    initial_prompt = (
        text + prompt + justify_prompt + "The response must be encoded in UTF-8 and cannot contain any special symbols such as emojis.\n"
        "Return only the json response, dont add any additional text.\n"
    )

    if not justification:
        initial_prompt = ( 
         text + prompt + not_justify_prompt + "The response must be encoded in UTF-8 and cannot contain any special symbols such as emojis.\n"
         "Return only the json response, dont add any additional text.\n"
        )

    api_request = {
        "model": "llama3.1-70b",
        "messages": [{"role": "user", "content": initial_prompt}],
        "stream": False,
    }

    try:
        response = llama.run(api_request).json()
        utils.log_message(f"Raw response from Llama:\n {response}", "llama")
        choices = response.get("choices", [])
        if not choices or "message" not in choices[0]:
            utils.log_message("Unexpected API response format", "llama")
            return 'NA(1003)'

        response_json = choices[0]["message"]["content"].strip()
        utils.log_message(f"Raw response before sanitization: {response_json}", "llama")

        if not response_json.strip():
            utils.log_message("Received an empty response", "llama")
            return 'NA(1004)'

        sanitize_response = utils.sanitize_json(response_json)
        if not sanitize_response:
            return 'NA(1005)'

        utils.log_message(f"Sanitized response:\n{sanitize_response}", "llama")
        response_dict = json.loads(sanitize_response)
        sentiment = response_dict.get('sentiment', 'NA')
        certainty = response_dict.get('certainty', 'NA')
        certaintyInt = int(certainty[:-1])
        justification_text = response_dict.get('justification', 'NA')

        count = 0
        while (sentiment == "=" or certaintyInt < trustValue) and n_media > 0:
            utils.log_message("Sentimiento neutral o indeterminable. Reenviando imagen para análisis más detallado...", "llama")
            if justification:
                reevaluation_prompt = (text + reevaluate_prompt + media[count] + justify_prompt)
            else:
                reevaluation_prompt = (text + reevaluate_prompt + media[count] + not_justify_prompt)

            api_request = {
                "model": "llama3.1-70b",
                "messages": [{"role": "user", "content": reevaluation_prompt}],
                "stream": False,
            }
            response = llama.run(api_request).json()
            utils.log_message(f"Response from Llama:\n {response}", "llama")
            choices = response.get("choices", [])
            if not choices or "message" not in choices[0]:
                utils.log_message("Unexpected API response format", "llama")
                return 'NA(1003)'

            response_json = choices[0]["message"]["content"].strip()
            utils.log_message(f"Raw response before sanitization: {response_json}", "llama")

            if not response_json.strip():
                utils.log_message("Received an empty response", "llama")
                return 'NA(1004)'

            sanitize_response = utils.sanitize_json(response_json)
            if not sanitize_response:
                return 'NA(1005)'

            utils.log_message(f"Sanitized response:\n{sanitize_response}")
            response_dict = json.loads(sanitize_response)
            sentiment = response_dict.get('sentiment', 'NA')
            certainty = response_dict.get('certainty', 'NA')
            certaintyInt = int(certainty[:-1])
            justification_text = response_dict.get('justification', 'NA')
            n_media -= 1
            n_items_used+=1
            count += 1

        if justification:
            return {'sentiment': sentiment, 'certainty': certainty, 'justification': justification_text, 'nItemsUsed': n_items_used}
        else:
            return {'sentiment': sentiment, 'certainty': certainty, 'nItemsUsed': n_items_used}

    except json.JSONDecodeError as e:
        utils.log_message(f"Error decoding JSON: {e}", "llama")
        return 'NA(1001)'
    except Exception as e:
        utils.log_message(f"Error processing response: {e}", "llama")
        return 'NA(1002)'

def send_to_deepseek(tweet,media, justification, n_media, n_items_used):

    response = clientDeepseek.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": (tweet + prompt + justify_prompt) if justification else (tweet + prompt + not_justify_prompt)
                    }
                ],
                stream=False
    )
    
    output = response.choices[0].message.content.strip()
    utils.log_message(f"Deepseek's raw response:\n{output}\n")
    sentiment, certainty, justification = utils.parse_chatgpt_response(output)
    certaintyInt = int(certainty[:-1])

    trustValue = 60 
    # ES: Reevalúa si el sentimiento es neutral y n_media > 0
    # EN: Reevaluate if the sentiment is neutral and n_media > 0
    count = 0
    while sentiment == "=" and (n_media > 0 or certaintyInt < trustValue):
        sentiment, certainty, justification = reevaluate_neutral_sentiment_deepseek(tweet, justification, media, count)
        n_media -=1
        n_items_used += 1
        certaintyInt = int(certainty[:-1])
    
    return sentiment, certainty, justification

def reevaluate_neutral_sentiment_deepseek(tweet, justification, media, count):
    
    response = clientDeepseek.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": tweet + reevaluate_prompt + prompt + justify_prompt + + media[count] if justification else tweet + reevaluate_prompt +prompt + not_justify_prompt + + media[count]
                    }
                ],
                stream=False
    )
    
    output = response.choices[0].message.content.strip()
    utils.log_message(f"Deepseek's reevaluated response:\n{output}\n", "Deepseek")
    return utils.parse_chatgpt_response(output)