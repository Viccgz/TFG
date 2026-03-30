
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
    # Function to analyze sentiment and return polarity and subjectivity
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Classify sentiment as positive, negative or neutral
        if polarity > 0:
            sentiment = '+'
        elif polarity < 0:
            sentiment = '-'
        else:
            sentiment = '='
        
        return sentiment, polarity, subjectivity

    # Apply sentiment analysis to the text column
    df[['sentiment_TextBlob', 'polarity_TextBlob', 'subjectivity_TextBlob']] = df['text'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    return df

def vader_sentiment_analysis(df):
    # Download VADER lexicon
    nltk.download('vader_lexicon')

    # Initialize the VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Function to analyze sentiment
    def analyze_sentiment(text):
        sentiment_scores = sia.polarity_scores(text)
        # Use the compound score as a measure of polarity
        polarity = sentiment_scores['compound']
        pos = sentiment_scores['pos']
        neg = sentiment_scores['neg']
        neu = sentiment_scores['neu']
        
        # Classify sentiment based on compound score
        if polarity >= 0.05:
            sentiment = '+'
        elif polarity <= -0.05:
            sentiment = '-'
        else:
            sentiment = '='
        
        return sentiment, polarity, pos, neg, neu

    # Apply sentiment analysis to the text column
    df[['sentiment_VADER', 'compound_VADER', 'pos_VADER', 'neg_VADER', 'neu_VADER']] = df['text'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    return df

def BERT_sentiment_analysis(df):
    # Preprocesamiento del texto
    def preprocess(text):
        new_text = []
        for t in str(text).split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    # Cargar modelo y tokenizer
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Descargar etiquetas (labels)
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

    # Clasificar cada mensaje
    results = []

    for text in df["text"]:
        if pd.isna(text):
            results.append("undefined")
            continue

        preprocessed = preprocess(text)
        encoded_input = tokenizer(preprocessed, return_tensors='pt')
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

    # Añadir columna de sentimiento
    df["sentiment_BERT"] = results

    return df

def NRCLex_emotion_analysis(df):
    
    def analyze_emotion(text):
        # Guard clause: skip non-string/NaN rows
        if not isinstance(text, str):
            return 'neutral'
            
        emotion = NRCLex()
        emotion.load_raw_text(text)
        top_emotions = emotion.top_emotions
        
        if top_emotions:
            return top_emotions[0][0]  # Primary emotion
        else:
            return 'neutral'
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')
    df['emotion_NRCLex'] = df['text'].apply(analyze_emotion)
    return df

def GoEmotions_EmoRoBERTa_emotion_analysis(df, library):
    if library == "EmoRoBERTa":
        MODEL = "j-hartmann/emotion-english-distilroberta-base"
    elif library == "GoEmotions":
        MODEL = "monologg/bert-base-cased-goemotions-original"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Get labels
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
        df["emotion_EmoRoBERTa"] = results
    elif library == "GoEmotions":        
        df["emotion_GoEmotions"] = results
    return df
