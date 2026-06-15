# Script para ejecutar todas las evaluaciones de métricas
# Cambiar al directorio del proyecto
Set-Location "c:\Users\vicza\Desktop\TFG\TFG"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Iniciando evaluaciones de METRICS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# =============================================================================
# GOEMOTIONS
# =============================================================================
Write-Host "`n[GOEMOTIONS] Iniciando evaluaciones..." -ForegroundColor Yellow

Write-Host "  - ChatGPT Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_chatgpt --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/gpt_results_emotion --output-prefix gpt_emotion

Write-Host "  - ChatGPT Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_chatgpt --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/gpt_results_emotion --output-prefix gpt_emotion_mapped

Write-Host "  - ChatGPT Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_with_sentiment_chatgpt --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/gpt_results_emotion_with_sentiment --output-prefix gpt_emotion_with_sentiment

Write-Host "  - ChatGPT Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_with_sentiment_chatgpt --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/gpt_results_emotion_with_sentiment --output-prefix gpt_emotion_mapped_with_sentiment

Write-Host "  - Mistral Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_mistral --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/mistral_results_emotion --output-prefix mistral_emotion

Write-Host "  - Mistral Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_mistral --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/mistral_results_emotion --output-prefix mistral_emotion_mapped

Write-Host "  - Mistral Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_with_sentiment_mistral --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/mistral_results_emotion_with_sentiment --output-prefix mistral_emotion_with_sentiment

Write-Host "  - Mistral Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_with_sentiment_mistral --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/mistral_results_emotion_with_sentiment --output-prefix mistral_emotion_mapped_with_sentiment

Write-Host "  - DeepSeek Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_deepseek --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/deepseek_results_emotion --output-prefix deepseek_emotion

Write-Host "  - DeepSeek Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_deepseek --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/deepseek_results_emotion --output-prefix deepseek_emotion_mapped

Write-Host "  - DeepSeek Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_with_sentiment_raw_deepseek --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/deepseek_results_emotion_with_sentiment --output-prefix deepseek_emotion_with_sentiment

Write-Host "  - DeepSeek Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_with_sentiment_deepseek --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/deepseek_results_emotion_with_sentiment --output-prefix deepseek_emotion_mapped_with_sentiment

Write-Host "  - NRCLex Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_NRCLex --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/nrclex_results_emotion --output-prefix nrclex_emotion

Write-Host "  - NRCLex Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_NRCLex --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/nrclex_results_emotion --output-prefix nrclex_emotion_mapped

Write-Host "  - EmoRoBERTa Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_EmoRoBERTa --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/emoroberta_results_emotion --output-prefix emoroberta_emotion

Write-Host "  - EmoRoBERTa Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_EmoRoBERTa --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/emoroberta_results_emotion --output-prefix emoroberta_emotion_mapped

Write-Host "  - GoEmotions Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_raw_GoEmotions --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/goemotions/goemotions_results_emotion --output-prefix goemotions_emotion

Write-Host "  - GoEmotions Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/goemotions_final_results.csv --pred emotion_mapped_GoEmotions --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/goemotions/goemotions_results_emotion --output-prefix goemotions_emotion_mapped

Write-Host "[GOEMOTIONS] ¡Completado!" -ForegroundColor Green

# =============================================================================
# ISEAR
# =============================================================================
Write-Host "`n[ISEAR] Iniciando evaluaciones..." -ForegroundColor Yellow

Write-Host "  - ChatGPT Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_chatgpt --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/gpt_results_emotion --output-prefix gpt_emotion

Write-Host "  - ChatGPT Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_chatgpt --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/gpt_results_emotion --output-prefix gpt_emotion_mapped

Write-Host "  - ChatGPT Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_with_sentiment_chatgpt --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/gpt_results_emotion_with_sentiment --output-prefix gpt_emotion_with_sentiment

Write-Host "  - ChatGPT Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_with_sentiment_chatgpt --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/gpt_results_emotion_with_sentiment --output-prefix gpt_emotion_mapped_with_sentiment

Write-Host "  - DeepSeek Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_deepseek --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/deepseek_results_emotion_new_prompt --output-prefix deepseek_emotion

Write-Host "  - DeepSeek Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_deepseek --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/deepseek_results_emotion_new_prompt --output-prefix deepseek_emotion_mapped

Write-Host "  - DeepSeek Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_with_sentiment_raw_deepseek --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/deepseek_results_emotion_new_prompt_with_sentiment --output-prefix deepseek_emotion_with_sentiment

Write-Host "  - DeepSeek Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_with_sentiment_deepseek --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/deepseek_results_emotion_new_prompt_with_sentiment --output-prefix deepseek_emotion_mapped_with_sentiment

Write-Host "  - Gemini Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_gemini --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/gemini_results_emotion --output-prefix gemini_emotion

Write-Host "  - Gemini Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_gemini --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/gemini_results_emotion --output-prefix gemini_emotion_mapped

Write-Host "  - Mistral Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_mistral --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/mistral_results_emotion --output-prefix mistral_emotion

Write-Host "  - Mistral Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_mistral --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/mistral_results_emotion --output-prefix mistral_emotion_mapped

Write-Host "  - Mistral Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_with_sentiment_mistral --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/mistral_results_emotion_with_sentiment --output-prefix mistral_emotion_with_sentiment

Write-Host "  - Mistral Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_with_sentiment_mistral --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/mistral_results_emotion_with_sentiment --output-prefix mistral_emotion_mapped_with_sentiment

Write-Host "  - NRCLex Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_NRCLex --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/nrclex_results_emotion --output-prefix nrclex_emotion

Write-Host "  - NRCLex Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_NRCLex --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/nrclex_results_emotion --output-prefix nrclex_emotion_mapped

Write-Host "  - EmoRoBERTa Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_EmoRoBERTa --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/emoroberta_results_emotion --output-prefix emoroberta_emotion

Write-Host "  - EmoRoBERTa Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_EmoRoBERTa --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/emoroberta_results_emotion --output-prefix emoroberta_emotion_mapped

Write-Host "  - GoEmotions Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_raw_GoEmotions --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/isear/goemotions_results_emotion --output-prefix goemotions_emotion

Write-Host "  - GoEmotions Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/isear_final_results.csv --pred emotion_mapped_GoEmotions --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/isear/goemotions_results_emotion --output-prefix goemotions_emotion_mapped

Write-Host "[ISEAR] ¡Completado!" -ForegroundColor Green

# =============================================================================
# KAGGLE
# =============================================================================
Write-Host "`n[KAGGLE] Iniciando evaluaciones..." -ForegroundColor Yellow

Write-Host "  - ChatGPT Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_chatgpt --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/gpt_results_emotion --output-prefix gpt_emotion

Write-Host "  - ChatGPT Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_chatgpt --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/gpt_results_emotion --output-prefix gpt_emotion_mapped

Write-Host "  - NRCLex Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_NRCLex --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/nrclex_results_emotion --output-prefix nrclex_emotion

Write-Host "  - NRCLex Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_NRCLex --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/nrclex_results_emotion --output-prefix nrclex_emotion_mapped

Write-Host "  - EmoRoBERTa Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_EmoRoBERTa --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/emoroberta_results_emotion --output-prefix emoroberta_emotion

Write-Host "  - EmoRoBERTa Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_EmoRoBERTa --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/emoroberta_results_emotion --output-prefix emoroberta_emotion_mapped

Write-Host "  - GoEmotions Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_GoEmotions --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/goemotions_results_emotion --output-prefix goemotions_emotion

Write-Host "  - GoEmotions Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_GoEmotions --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/goemotions_results_emotion --output-prefix goemotions_emotion_mapped

Write-Host "  - ChatGPT Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_with_sentiment_chatgpt --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/gpt_results_emotion_with_sentiment --output-prefix gpt_emotion_with_sentiment

Write-Host "  - ChatGPT Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_with_sentiment_chatgpt --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/gpt_results_emotion_with_sentiment --output-prefix gpt_emotion_mapped_with_sentiment

Write-Host "  - DeepSeek Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_deepseek --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/deepseek_results_emotion --output-prefix deepseek_emotion

Write-Host "  - DeepSeek Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped__deepseek --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/deepseek_results_emotion --output-prefix deepseek_emotion_mapped

Write-Host "  - DeepSeek Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_with_sentiment_raw_deepseek --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/deepseek_results_emotion_with_sentiment --output-prefix deepseek_emotion_with_sentiment

Write-Host "  - DeepSeek Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_with_sentiment_deepseek --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/deepseek_results_emotion_with_sentiment --output-prefix deepseek_emotion_mapped_with_sentiment

Write-Host "  - Mistral Fine-grained emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_mistral --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/mistral_results_emotion --output-prefix mistral_emotion

Write-Host "  - Mistral Mapped emotions" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_mistral --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/mistral_results_emotion --output-prefix mistral_emotion_mapped

Write-Host "  - Mistral Fine-grained emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_raw_with_sentiment_mistral --gt emotion_raw_gt --task fine_grained_emotions --output-dir data/results/metrics/kaggle/mistral_results_emotion_with_sentiment --output-prefix mistral_emotion_with_sentiment

Write-Host "  - Mistral Mapped emotions with sentiment" -ForegroundColor Gray
python src/concurrente/metrics.py --csv data/results/concurrente/kaggle_final_results.csv --pred emotion_mapped_with_sentiment_mistral --gt emotion_mapped_gt --task mapped_emotions --output-dir data/results/metrics/kaggle/mistral_results_emotion_with_sentiment --output-prefix mistral_emotion_mapped_with_sentiment

Write-Host "[KAGGLE] ¡Completado!" -ForegroundColor Green

# =============================================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "¡TODAS LAS EVALUACIONES COMPLETADAS!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
