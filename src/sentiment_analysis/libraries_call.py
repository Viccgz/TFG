
import pandas as pd
from textblob import TextBlob
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
import urllib.request
import csv
from nrclex import NRCLex

def TextBlob_sentiment_analysis(df):
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # ES: Clasifica el sentimiento como positivo, negativo o neutral
        # EN: Classify sentiment as positive, negative or neutral
        if polarity > 0:
            sentiment = '+'
        elif polarity < 0:
            sentiment = '-'
        else:
            sentiment = '='
        
        return sentiment, polarity, subjectivity
    
    # ES: Aplica el análisis de sentimiento a la columna de texto
    # EN: Apply sentiment analysis to the text column
    df[['sentiment_TextBlob', 'polarity_TextBlob', 'subjectivity_TextBlob']] = df['text'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    return df

def vader_sentiment_analysis(df):
    nltk.download('vader_lexicon')

    # ES: Inicializa el analizador de intensidad de sentimiento VADER
    # EN: Initialize the VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(text):
        sentiment_scores = sia.polarity_scores(text)
        # ES: Usa la puntuación compuesta como medida de polaridad
        # EN: Use the compound score as a measure of polarity
        polarity = sentiment_scores['compound']
        pos = sentiment_scores['pos']
        neg = sentiment_scores['neg']
        neu = sentiment_scores['neu']
        
        # ES: Clasifica el sentimiento basado en la puntuación compuesta
        # EN: Classify sentiment based on compound score
        if polarity >= 0.05:
            sentiment = '+'
        elif polarity <= -0.05:
            sentiment = '-'
        else:
            sentiment = '='
        
        return sentiment, polarity, pos, neg, neu

    # ES: Aplica el análisis de sentimiento a la columna de texto
    # EN: Apply sentiment analysis to the text column
    df[['sentiment_VADER', 'compound_VADER', 'pos_VADER', 'neg_VADER', 'neu_VADER']] = df['text'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    return df

def BERT_sentiment_analysis(df):
    def preprocess(text):
        new_text = []
        for t in str(text).split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    # ES: Cargar modelo y tokenizer
    # EN: Load model and tokenizer
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # ES: Descargar etiquetas (labels)
    # EN: Download labels
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

    # ES: Clasificar cada mensaje
    # EN: Classify each message
    results = []

    for text in df["text"]:
        if pd.isna(text):
            results.append("undefined")
            continue

        preprocessed = preprocess(text)
        encoded_input = tokenizer(
    preprocessed,
    return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        label = labels[np.argmax(scores)]
        if label == "positive":
            label = "+"
            
        elif label == "negative":
            label = "-"

        else: 
            label = "="

        results.append(label)

    # ES: Añadir columna de sentimiento
    # EN: Add sentiment column
    df["sentiment_BERT"] = results

    return df

def NRCLex_emotion_analysis(df):
    
    def analyze_emotion(text):
        # ES: Guard clause: omitir filas que no sean string o NaN
        # EN: Guard clause: skip non-string/NaN rows
        if not isinstance(text, str):
            return 'neutral'
            
        emotion = NRCLex()
        emotion.load_raw_text(text)
        top_emotions = emotion.top_emotions
        
        if top_emotions:
            return top_emotions[0][0]  # ES: Emoción principal
                                       # EN: Primary emotion
        else:
            return 'neutral'
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')
    df['emotion_raw_NRCLex'] = df['text'].apply(analyze_emotion)
    return df

def GoEmotions_EmoRoBERTa_emotion_analysis(df, library):
    if library == "EmoRoBERTa":
        MODEL = "j-hartmann/emotion-english-distilroberta-base"
    elif library == "GoEmotions":
        MODEL = "monologg/bert-base-cased-goemotions-original"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    labels = model.config.id2label
    
    results = []
    for text in df["text"]:
        if pd.isna(text):
            results.append("neutral")
            continue
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        label = labels[np.argmax(scores)]
        results.append(label)
    if library == "EmoRoBERTa":
        df["emotion_raw_EmoRoBERTa"] = results
    elif library == "GoEmotions":        
        df["emotion_raw_GoEmotions"] = results
    return df
