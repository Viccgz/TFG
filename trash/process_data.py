import os
import pandas as pd

from src.concurrente.dataset_preprocessing.emotion_mapper import normalize_emotion_label
from src.concurrente.sentiment_analysis import libraries_call


df = pd.read_csv("output.csv", sep=",")
    
# Perform sentiment analysis with libraries
df = libraries_call.TextBlob_sentiment_analysis(df)
df = libraries_call.vader_sentiment_analysis(df)
df = libraries_call.BERT_sentiment_analysis(df)
    
# Save sentiment analysis results
sentiment_csv = "./data/results/concurrente/" + "deepseek" + "_"+ "isear" + "_sentiment_analysis_results.csv"
os.makedirs(os.path.dirname(sentiment_csv), exist_ok=True)
df.to_csv(sentiment_csv, index=False)
print(f"Sentiment analysis results saved to {sentiment_csv}")
    
# Perform emotion analysis with libraries
df = libraries_call.NRCLex_emotion_analysis(df)
df = libraries_call.GoEmotions_EmoRoBERTa_emotion_analysis(df, "EmoRoBERTa")
df = libraries_call.GoEmotions_EmoRoBERTa_emotion_analysis(df, "GoEmotions")
    
# Map raw emotions from each library to normalized emotions using normalize_emotion_label
df["emotion_mapped_NRCLex"] = df["emotion_raw_NRCLex"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
df["emotion_mapped_EmoRoBERTa"] = df["emotion_raw_EmoRoBERTa"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
df["emotion_mapped_GoEmotions"] = df["emotion_raw_GoEmotions"].apply(lambda x: normalize_emotion_label(x) if pd.notnull(x) else "neutral")
# Save emotion analysis results
emotion_csv = "./data/results/concurrente/" + "deepseek" + "_"+ "isear" + "_sentiment_emotion_analysis_results.csv"
os.makedirs(os.path.dirname(emotion_csv), exist_ok=True)
df.to_csv(emotion_csv, index=False)
print(f"Emotion analysis results saved to {emotion_csv}")

    
# Reordenar columnas en el orden especificado
order_base_columns = ["id", "text",
                 "sentiment_" + "deepseek", 
                 "certainty_sentiment_" + "deepseek", 
                 "justification_sentiment_" + "deepseek",
                 "processing_date_sentiment", "processing_hour_sentiment",
                 "emotion_gt", "emotion_gt_mapped",
                 "emotion_raw_" + "deepseek",
                 "emotion_mapped_" + "deepseek",
                 "certainty_emotion_" + "deepseek",
                 "justification_emotion_" + "deepseek",
                 "processing_date_emotion", "processing_hour_emotion"]
    
# Obtener columnas de librerias (BERT, TextBlob, VADER, NRCLex) excluyendo _id
order_remaining_cols = [col for col in df.columns if col not in order_base_columns and col != "total_process_time_seconds" and col != "_id"]
    
# Ordenar columnas finales: base + remaining + total_process_time_seconds
final_cols = order_base_columns + order_remaining_cols 
# Filtrar solo columnas que existan en el dataframe
final_cols = [col for col in final_cols if col in df.columns]
df = df[final_cols]
    
df.to_csv("deepseek_isear_sentiment_emotion_analysis_results.csv", index=False)

print("All processes completed successfully.")