
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

def TextBlob_sentiment_analysis(input_file):
    # Read the Excel file
    #file_path = 'data/USElections2024_All.xlsx'
    reviews_df = pd.read_csv(input_file)

    # Check the structure of the DataFrame
    print(reviews_df.head())

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
    reviews_df[['sentiment', 'polarity', 'subjectivity']] = reviews_df['message'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    # Print the results
    print(reviews_df[['id', 'message', 'sentiment', 'polarity', 'subjectivity']])

    # Save the results to a new Excel file
    output_file_path = '/USElections2024_All_with_sentiment_TextBlob.xlsx'
    reviews_df.to_excel(output_file_path, index=False)

    print(f'Sentiment analysis results saved to {output_file_path}')
    return output_file_path

def vader_sentiment_analysis(input_file):
    # Download VADER lexicon
    nltk.download('vader_lexicon')

    # Read the Excel file
    # file_path = 'data/USElections2024_All2.xlsx'
    reviews_df = pd.read_excel(input_file)

    # Check the structure of the DataFrame
    print(reviews_df.head())

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
    reviews_df[['sentiment', 'compound', 'pos', 'neg', 'neu']] = reviews_df['message'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    # Print the results
    print(reviews_df[['id', 'message', 'sentiment', 'compound', 'pos', 'neg', 'neu']])

    # Save the results to a new Excel file
    output_file_path = 'data/USElections2024_All3.xlsx'
    reviews_df.to_excel(output_file_path, index=False)

    print(f'Sentiment analysis results saved to {output_file_path}')
    return output_file_path

def BERT_sentiment_analysis(input_file):
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

    # Leer archivo CSV
    df = pd.read_csv(input_file, encoding="latin1", sep=';')  # <-- cambia aquí el nombre de tu archivo

    # Clasificar cada mensaje
    results = []

    for text in df["message"]:
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

    # Guardar resultado a nuevo CSV
    df.to_csv("archivo_clasificado.csv", index=False)