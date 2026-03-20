import pandas as pd
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
import krippendorff
from itertools import combinations
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from wordcloud import WordCloud

def calculate_majority(file_path):
        # Load the Excel file
    #file_path = 'data/USElections2024_All3.xlsx'
    df = pd.read_excel(file_path)

    # List of LLM columns and Python library columns
    llm_columns = ['sentiment_gpt', 'sentiment_deepseek', 'sentiment_claude', 
                'sentiment_gemini', 'sentiment_llama']
    lib_columns = ['sentimentTextBlob', 'sentimentVader']

    # Function to calculate counts and majority decision
    def calculate_stats(row, columns):
        # Get the values for the specified columns
        values = [row[col] for col in columns]
        
        # Count each sentiment
        counts = Counter(values)
        pos = counts.get('+', 0)
        neg = counts.get('-', 0)
        neu = counts.get('=', 0)
        
        # Determine majority decision
        max_count = max(pos, neg, neu)
        # Check if there's a tie for the max count
        if (pos == max_count and neg == max_count) or \
        (pos == max_count and neu == max_count) or \
        (neg == max_count and neu == max_count):
            majority = 'tie'
        else:
            if max_count == pos:
                majority = '+'
            elif max_count == neg:
                majority = '-'
            else:
                majority = '='
        
        return pos, neg, neu, majority

    # Process each row to calculate the new columns
    for index, row in df.iterrows():
        # Calculate for LLMs
        llm_pos, llm_neg, llm_neu, llm_majority = calculate_stats(row, llm_columns)
        df.at[index, 'LLM_positives'] = llm_pos
        df.at[index, 'LLM_negatives'] = llm_neg
        df.at[index, 'LLM_neutrals'] = llm_neu
        df.at[index, 'LLM_majority'] = llm_majority
        
        # Calculate for Python libraries
        lib_pos, lib_neg, lib_neu, lib_majority = calculate_stats(row, lib_columns)
        df.at[index, 'Lib_positives'] = lib_pos
        df.at[index, 'Lib_negatives'] = lib_neg
        df.at[index, 'Lib_neutrals'] = lib_neu
        df.at[index, 'Lib_majority'] = lib_majority

    # Save the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False)

    print("Processing complete. The file has been updated with the new columns.")

def calculate_accuracy(file_path):
        # Load the original Excel file
    #input_file = 'data/KamalaHarris_X_All4.xlsx'
    output_file = 'r1_MetricsAccuracy.xlsx'
    df = pd.read_excel(file_path)

    # List of all coders (including LLM_majority and Lib_majority)
    coders = ['sentiment_gpt', 'sentiment_deepseek', 'sentiment_claude', 
            'sentiment_gemini', 'sentiment_llama', 'sentimentTextBlob', 
            'sentimentVader', 'LLM_majority', 'Lib_majority']

    # Initialize a dictionary to store metrics
    metrics = {
        'Coder': [],
        'Positives': [],
        'Negatives': [],
        'Neutrals': [],
        'Ties': [],
        'Accuracy': [],
        'F1_Score_Weighted': [],
        'F1_Score_Positive': [],
        'F1_Score_Negative': [],
        'F1_Score_Neutral': [],
        'Precision_Weighted': [],
        'Recall_Weighted': [],
        'N_Valid_Comparisons': []
    }

    # Calculate metrics for each coder
    for coder in coders:
        # Skip if coder column doesn't exist
        if coder not in df.columns:
            continue
        
        # Count sentiment distribution
        counts = df[coder].value_counts()
        pos = counts.get('+', 0)
        neg = counts.get('-', 0)
        neu = counts.get('=', 0)
        ties = counts.get('tie', 0)
        
        # Create mask for valid comparisons (excluding ties and NaN)
        valid_mask = (~df[coder].isin(['tie', np.nan])) & (~df['GTsentiment'].isin(['tie', np.nan]))
        
        # Calculate accuracy only on valid comparisons
        if sum(valid_mask) > 0:
            correct = (df.loc[valid_mask, coder] == df.loc[valid_mask, 'GTsentiment']).sum()
            accuracy = correct / sum(valid_mask)
        else:
            accuracy = np.nan
        
        # Convert to numerical for sklearn (1: positive, -1: negative, 0: neutral)
        y_true = df.loc[valid_mask, 'GTsentiment'].map({'+': 1, '-': -1, '=': 0})
        y_pred = df.loc[valid_mask, coder].map({'+': 1, '-': -1, '=': 0})
        
        # Calculate metrics only if we have valid comparisons
        if len(y_true) > 0:
            # Calculate weighted scores
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Calculate per-class F1 scores using weighted approach
            f1_pos = f1_score(y_true, y_pred, labels=[1], average='weighted', zero_division=0)
            f1_neg = f1_score(y_true, y_pred, labels=[-1], average='weighted', zero_division=0)
            f1_neu = f1_score(y_true, y_pred, labels=[0], average='weighted', zero_division=0)
        else:
            f1_weighted = np.nan
            precision_weighted = np.nan
            recall_weighted = np.nan
            f1_pos = np.nan
            f1_neg = np.nan
            f1_neu = np.nan
        
        # Store metrics
        metrics['Coder'].append(coder)
        metrics['Positives'].append(pos)
        metrics['Negatives'].append(neg)
        metrics['Neutrals'].append(neu)
        metrics['Ties'].append(ties)
        metrics['Accuracy'].append(accuracy)
        metrics['F1_Score_Weighted'].append(f1_weighted)
        metrics['F1_Score_Positive'].append(f1_pos)
        metrics['F1_Score_Negative'].append(f1_neg)
        metrics['F1_Score_Neutral'].append(f1_neu)
        metrics['Precision_Weighted'].append(precision_weighted)
        metrics['Recall_Weighted'].append(recall_weighted)
        metrics['N_Valid_Comparisons'].append(sum(valid_mask))

    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(metrics)

    # Save to new Excel file
    metrics_df.to_excel(output_file, index=False)

    print(f"Metrics calculated and saved to {output_file}")

def interrated(file_path):
        def prepare_data(df):
            """Convert raw data to numerical matrix with consistent shape"""
            sent_map = {'+': 2, '-': 0, '=': 1}
            coders = ['GTsentiment', 'sentiment_gpt', 'sentiment_deepseek', 
                    'sentiment_claude', 'sentiment_gemini', 'sentiment_llama',
                    'sentimentTextBlob', 'sentimentVader', 'LLM_majority', 'Lib_majority']
            
            # Initialize matrix with NaNs
            data = np.full((len(df), len(coders)), np.nan)
            
            # Fill with valid ratings
            for msg_idx, row in df.iterrows():
                for coder_idx, coder in enumerate(coders):
                    val = row.get(coder, np.nan)
                    if val in sent_map:
                        data[msg_idx, coder_idx] = sent_map[val]
            
            return data, coders

        def calculate_pairwise_metrics(data, coders):
            """Calculate Cohen's Kappa and Krippendorff's Alpha for each coder vs GT"""
            gt = data[:, 0]
            results = []
            
            for coder_idx, coder in enumerate(coders[1:], 1):  # Skip GT
                valid_mask = ~np.isnan(data[:, coder_idx]) & ~np.isnan(gt)
                n_valid = sum(valid_mask)
                
                # Cohen's Kappa
                kappa = np.nan
                if n_valid > 0:
                    kappa = cohen_kappa_score(gt[valid_mask], data[valid_mask, coder_idx])
                
                # Krippendorff's Alpha (pairwise)
                alpha = np.nan
                if n_valid > 0:
                    pairwise_data = np.vstack([gt[valid_mask], data[valid_mask, coder_idx]])
                    try:
                        alpha = krippendorff.alpha(
                            reliability_data=pairwise_data,
                            value_domain=[0, 1, 2],
                            level_of_measurement='ordinal'
                        )
                    except:
                        pass
                
                results.append({
                    'Coder': coder,
                    'Kappa_vs_GT': round(kappa, 3) if not np.isnan(kappa) else np.nan,
                    'Alpha_vs_GT': round(alpha, 3) if not np.isnan(alpha) else np.nan,
                    'N_Valid': n_valid
                })
            
            return pd.DataFrame(results)

        def calculate_group_metrics(data, coders):
            """Calculate Kappa and Alpha for different coder groups"""
            groups = {
                'All_LLMs': [1,2,3,4,5],                   # GPT, DeepSeek, Claude, Gemini, Llama
                'All_LLMs_plus_majority': [1,2,3,4,5,8],    # + LLM_majority
                'Python_Libs': [6,7],                       # TextBlob, Vader
                'All_Automated': [1,2,3,4,5,6,7],           # All LLMs + Python Libs
                'Full_System': [0,1,2,3,4,5,6,7],           # GT + All LLMs + Python Libs
                'LLMs_plus_GT': [0,1,2,3,4,5],              # GT + All LLMs
                'Libs_plus_GT': [0,6,7]                     # GT + Python Libs
            }
            
            results = []
            
            for group_name, indices in groups.items():
                group_data = data[:, indices]
                
                # Filter messages with at least 2 valid ratings
                valid_msgs = []
                for msg_ratings in group_data:
                    valid_ratings = [x for x in msg_ratings if not np.isnan(x)]
                    if len(valid_ratings) >= 2:
                        valid_msgs.append(valid_ratings)
                
                # Skip if not enough data
                if len(valid_msgs) < 2:
                    results.append({
                        'Group': group_name,
                        'Krippendorff_Alpha': np.nan,
                        'Avg_Pairwise_Kappa': np.nan,
                        'N_Valid_Messages': len(valid_msgs)
                    })
                    continue
                
                # Calculate Krippendorff's Alpha
                max_len = max(len(x) for x in valid_msgs)
                padded_data = [x + [np.nan]*(max_len - len(x)) for x in valid_msgs]
                
                try:
                    alpha = krippendorff.alpha(
                        reliability_data=np.array(padded_data).T,
                        value_domain=[0, 1, 2],
                        level_of_measurement='ordinal'
                    )
                except:
                    alpha = np.nan
                
                # Calculate average pairwise Cohen's Kappa
                kappas = []
                n_coders = len(indices)
                for i in range(n_coders):
                    for j in range(i+1, n_coders):
                        valid_mask = ~np.isnan(group_data[:, i]) & ~np.isnan(group_data[:, j])
                        if sum(valid_mask) > 0:
                            kappa = cohen_kappa_score(
                                group_data[valid_mask, i], 
                                group_data[valid_mask, j]
                            )
                            kappas.append(kappa)
                
                avg_kappa = np.mean(kappas) if kappas else np.nan
                
                results.append({
                    'Group': group_name,
                    'Krippendorff_Alpha': round(alpha, 3) if not np.isnan(alpha) else np.nan,
                    'Avg_Pairwise_Kappa': round(avg_kappa, 3) if not np.isnan(avg_kappa) else np.nan,
                    'N_Valid_Messages': len(valid_msgs)
                })
            
            return pd.DataFrame(results)

        # Main execution
        df = pd.read_excel(file_path)
        data, coders = prepare_data(df)

        # Calculate metrics
        pairwise_df = calculate_pairwise_metrics(data, coders)
        group_df = calculate_group_metrics(data, coders)

        # Save results
        with pd.ExcelWriter('r2_interrater_All_individual_groups.xlsx') as writer:
            pairwise_df.to_excel(writer, sheet_name='Pairwise Metrics', index=False)
            group_df.to_excel(writer, sheet_name='Group Metrics', index=False)

        print("Results saved to 'r2_interrater_All_individual_groups.xlsx'")

def calculateStatisticalDiff(file_path):
        # Load the Excel file
    df = pd.read_excel(file_path)

    # List of LLM sentiment columns
    llms = ['sentiment_gpt', 'sentiment_claude', 'sentiment_gemini', 'sentiment_llama', 'sentiment_deepseek']

    # Initialize Excel writer for output
    writer = pd.ExcelWriter('r3_llm_sentiment_comparison.xlsx', engine='openpyxl')

    # 1. Create a contingency table for all LLMs
    # Prepare data: Create a DataFrame where each row is a message, and columns are sentiment labels for each LLM
    sentiment_data = []
    for llm in llms:
        counts = df[llm].value_counts()
        sentiment_data.append({
            'LLM': llm,
            '+': counts.get('+', 0),
            '-': counts.get('-', 0),
            '=': counts.get('=', 0)
        })
    contingency_table = pd.DataFrame(sentiment_data).set_index('LLM')[['+', '-', '=']]
    print("\n1. Contingency Table (LLM vs. Sentiment):")
    print(contingency_table)

    # Perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("Interpretation: ", "Significant difference" if p < 0.05 else "No significant difference")

    # Save contingency table and results
    contingency_table.to_excel(writer, sheet_name='LLM_Sentiment_Contingency')
    results_df = pd.DataFrame({
        'Chi2_Statistic': [chi2],
        'P-value': [p],
        'Degrees_of_Freedom': [dof],
        'Interpretation': ["Significant difference" if p < 0.05 else "No significant difference"]
    })
    results_df.to_excel(writer, sheet_name='LLM_Sentiment_Chi2')

    # 2. Post-hoc pairwise chi-square tests
    pairwise_results = []
    for llm1, llm2 in combinations(llms, 2):
        # Subset contingency table for the pair
        pair_table = contingency_table.loc[[llm1, llm2]]
        chi2_pair, p_pair, dof_pair, _ = stats.chi2_contingency(pair_table)
        pairwise_results.append({
            'Comparison': f"{llm1} vs {llm2}",
            'Chi2_Statistic': chi2_pair,
            'P-value': p_pair,
            'Degrees_of_Freedom': dof_pair,
            'Interpretation': "Significant difference" if p_pair < 0.05 else "No significant difference"
        })
    pairwise_df = pd.DataFrame(pairwise_results)
    print("\n2. Post-hoc Pairwise Chi-square Tests:")
    print(pairwise_df)
    pairwise_df.to_excel(writer, sheet_name='LLM_Sentiment_Pairwise')

    # 3. Visualize sentiment distributions
    contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#36A2EB', '#FFCE56', '#FF6384'])
    plt.title('Sentiment Distribution Across LLMs')
    plt.xlabel('LLM')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig('sentiment_by_llm.png')
    plt.close()

    # Save the Excel file
    writer.close()

def calculateSummarySentimentLLMs(file_path):
        # Load the Excel file
    df = pd.read_excel(file_path)

    # Initialize Excel writer
    writer = pd.ExcelWriter('r3_sentimentAnalysis.xlsx', engine='openpyxl')

    # 1. Count messages by source
    source_counts = df['source'].value_counts()
    print("\n1. Number of Messages by Source:")
    print(source_counts)
    source_counts.to_excel(writer, sheet_name='Source_Counts')

    # 2. Count messages by language
    lang_counts = df['lang'].value_counts()
    print("\n2. Number of Messages by Language:")
    print(lang_counts)
    lang_counts.to_excel(writer, sheet_name='Language_Counts')

    # 3. Count messages by language for each source (top 5 languages per source + Others)
    lang_source_counts = df.groupby(['source', 'lang']).size().unstack(fill_value=0)
    # For each source, get top 5 languages and sum others
    top_langs_by_source = {}
    for source in lang_source_counts.index:
        top_langs = lang_source_counts.loc[source].sort_values(ascending=False).head(10)
        others_count = lang_source_counts.loc[source].sum() - top_langs.sum()
        top_langs['Others'] = others_count
        top_langs_by_source[source] = top_langs
    top_langs_df = pd.DataFrame(top_langs_by_source).fillna(0).astype(int)
    print("\n3. Top 10 Languages by Source (with Others):")
    print(top_langs_df)
    top_langs_df.to_excel(writer, sheet_name='Top_Languages_by_Source')

    # 4. Sentiment analysis for each coder and LLM_majority
    coders = ['sentiment_gpt', 'sentiment_deepseek', 'sentiment_claude', 'sentiment_gemini', 
            'sentiment_llama', 'sentimentTextBlob', 'sentimentVader', 'LLM_majority']

    for coder in coders:
        # Initialize DataFrame for combined sentiment table
        sentiment_table = pd.DataFrame()
        
        # Overall sentiment
        overall_counts = df[coder].value_counts()
        overall_perc = df[coder].value_counts(normalize=True) * 100
        
        # Sentiment by source
        source_counts = df.groupby('source')[coder].value_counts().unstack(fill_value=0)
        source_perc = df.groupby('source')[coder].value_counts(normalize=True).unstack(fill_value=0) * 100
        
        # Sentiment for English messages (lang='en')
        english_counts = df[df['lang'] == 'en'][coder].value_counts()
        english_perc = df[df['lang'] == 'en'][coder].value_counts(normalize=True) * 100
        
        # Sentiment for non-English messages (lang!='en')
        non_english_counts = df[df['lang'] != 'en'][coder].value_counts()
        non_english_perc = df[df['lang'] != 'en'][coder].value_counts(normalize=True) * 100
        
        # Combine into a single table
        sentiment_table['OverallCount'] = overall_counts
        sentiment_table['Overall_Perc'] = overall_perc
        sentiment_table['TelegramCount'] = source_counts.loc['Telegram'] if 'Telegram' in source_counts.index else 0
        sentiment_table['Telegram_Perc'] = source_perc.loc['Telegram'] if 'Telegram' in source_perc.index else 0
        sentiment_table['XCount'] = source_counts.loc['X'] if 'X' in source_counts.index else 0
        sentiment_table['X_Perc'] = source_perc.loc['X'] if 'X' in source_perc.index else 0
        sentiment_table['EnglishCount'] = english_counts
        sentiment_table['English_Perc'] = english_perc
        sentiment_table['non-EnglishCount'] = non_english_counts
        sentiment_table['nonEnglish_Perc'] = non_english_perc
        
        # Fill NaN with 0 for counts and percentages
        sentiment_table = sentiment_table.fillna(0)
        
        # Ensure all expected sentiment labels are present
        for label in ['+', '-', '=', 'tie'] if coder == 'LLM_majority' else ['+', '-', '=']:
            if label not in sentiment_table.index:
                sentiment_table.loc[label] = [0] * len(sentiment_table.columns)
        
        # Sort by sentiment labels for consistency
        sentiment_table = sentiment_table.sort_index()
        
        print(f"\n4. {coder} - Sentiment Analysis:")
        print(sentiment_table)
        sentiment_table.to_excel(writer, sheet_name=f'{coder}')
        
        # Create and save bar chart for overall sentiment distribution
        plt.figure(figsize=(8, 6))
        overall_perc.plot(kind='bar', color=['#36A2EB', '#FFCE56', '#FF6384', '#D3D3D3'])
        plt.title(f'Overall Sentiment Distribution ({coder})')
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'overall_sentiment_{coder}.png')
        plt.close()

    # Save the Excel file
    writer.close()

def calculateStatisticsSentiment(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Remove rows where LLM_majority is 'tie'
    df = df[df['LLM_majority'] != 'tie']

    # Initialize Excel writer for output
    writer = pd.ExcelWriter('r3_statisticalTesting_with_pairwise.xlsx', engine='openpyxl')

    # 1. Chi-square test for sentiment differences between sources
    print("\n1. Chi-square Test for Sentiment Differences Between Sources (Excluding 'tie'):")
    # Create contingency table for source vs. LLM_majority sentiment
    source_sentiment = pd.crosstab(df['source'], df['LLM_majority'])
    print("Contingency Table (Source vs. Sentiment):")
    print(source_sentiment)

    # Perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(source_sentiment)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("Interpretation: ", "Significant difference" if p < 0.05 else "No significant difference")

    # Save contingency table and results
    source_sentiment.to_excel(writer, sheet_name='Source_Sentiment_Contingency')
    results_df = pd.DataFrame({
        'Chi2_Statistic': [chi2],
        'P-value': [p],
        'Degrees_of_Freedom': [dof],
        'Interpretation': ["Significant difference" if p < 0.05 else "No significant difference"]
    })
    results_df.to_excel(writer, sheet_name='Source_Sentiment_Chi2')

    # Post-hoc analysis: Pairwise chi-square tests for sources
    sources = source_sentiment.index
    pairwise_results = []
    for source1, source2 in combinations(sources, 2):
        # Subset contingency table for the pair
        pair_table = source_sentiment.loc[[source1, source2]]
        chi2_pair, p_pair, dof_pair, _ = stats.chi2_contingency(pair_table)
        pairwise_results.append({
            'Comparison': f"{source1} vs {source2}",
            'Chi2_Statistic': chi2_pair,
            'P-value': p_pair,
            'Degrees_of_Freedom': dof_pair,
            'Interpretation': "Significant difference" if p_pair < 0.05 else "No significant difference"
        })
    pairwise_df = pd.DataFrame(pairwise_results)
    print("\nPost-hoc Pairwise Chi-square Tests for Sources:")
    print(pairwise_df)
    pairwise_df.to_excel(writer, sheet_name='Source_Sentiment_Pairwise')

    # 2. Chi-square test for sentiment by hashtag/channel within each source
    print("\n2. Chi-square Tests for Sentiment by Hashtag/Channel within Each Source (Excluding 'tie'):")
    hashtag_results = []
    for source in df['source'].unique():
        # Filter data for the source
        source_df = df[df['source'] == source]
        
        # Get hashtag/channel counts and filter for >= 200 instances
        hashtag_counts = source_df['hashtag_chat'].value_counts()
        valid_hashtags = hashtag_counts[hashtag_counts >= 200].index
        
        if len(valid_hashtags) < 2:
            print(f"\nSource: {source} - Insufficient hashtags/channels with >= 200 instances")
            continue
        
        # Filter data for valid hashtags
        source_df = source_df[source_df['hashtag_chat'].isin(valid_hashtags)]
        
        # Create contingency table for hashtag vs. sentiment
        hashtag_sentiment = pd.crosstab(source_df['hashtag_chat'], source_df['LLM_majority'])
        print(f"\nContingency Table for {source} (Hashtag/Channel vs. Sentiment):")
        print(hashtag_sentiment)
        
        # Perform chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(hashtag_sentiment)
        print(f"\nSource: {source}")
        print(f"Chi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        print(f"Degrees of Freedom: {dof}")
        print("Interpretation: ", "Significant difference" if p < 0.05 else "No significant difference")
        
        # Save contingency table and results
        hashtag_sentiment.to_excel(writer, sheet_name=f'{source}_Hashtag_Sentiment')
        hashtag_results.append({
            'Source': source,
            'Chi2_Statistic': chi2,
            'P-value': p,
            'Degrees_of_Freedom': dof,
            'Interpretation': "Significant difference" if p < 0.05 else "No significant difference"
        })
        
        # Post-hoc pairwise chi-square tests for hashtags/channels
        hashtags = hashtag_sentiment.index
        pairwise_hashtag_results = []
        for hashtag1, hashtag2 in combinations(hashtags, 2):
            # Subset contingency table for the pair
            pair_table = hashtag_sentiment.loc[[hashtag1, hashtag2]]
            chi2_pair, p_pair, dof_pair, _ = stats.chi2_contingency(pair_table)
            pairwise_hashtag_results.append({
                'Comparison': f"{hashtag1} vs {hashtag2}",
                'Chi2_Statistic': chi2_pair,
                'P-value': p_pair,
                'Degrees_of_Freedom': dof_pair,
                'Interpretation': "Significant difference" if p_pair < 0.05 else "No significant difference"
            })
        pairwise_hashtag_df = pd.DataFrame(pairwise_hashtag_results)
        print(f"\nPost-hoc Pairwise Chi-square Tests for Hashtags/Channels in {source}:")
        print(pairwise_hashtag_df)
        pairwise_hashtag_df.to_excel(writer, sheet_name=f'{source}_Hashtag_Pairwise')
        
        # Visualize contingency table as a stacked bar chart
        hashtag_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#36A2EB', '#FFCE56', '#FF6384'])
        plt.title(f'Sentiment Distribution by Hashtag/Channel for {source} (Excluding tie)')
        plt.xlabel('Hashtag/Channel')
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(f'sentiment_by_hashtag_{source}.png')
        plt.close()

    # Save hashtag results
    hashtag_results_df = pd.DataFrame(hashtag_results)
    print("\nSummary of Hashtag/Channel Sentiment Tests:")
    print(hashtag_results_df)
    hashtag_results_df.to_excel(writer, sheet_name='Hashtag_Sentiment_Results')

    # Save the Excel file
    writer.close()

def carryOutTextAnalysis(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Print column names to help identify the text column
    print("Available columns in the dataset:")
    print(df.columns.tolist())

    # Specify the correct text column name (replace 'text' with the actual column name)
    text_column = 'message'  # CHANGE THIS TO THE CORRECT COLUMN NAME (e.g., 'message', 'content')

    # Check if the text column exists
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in the dataset. Please update 'text_column' to the correct column name.")

    # Remove rows where LLM_majority is 'tie' or '='
    df = df[~df['LLM_majority'].isin(['tie', '='])]

    # Initialize Excel writer for output
    writer = pd.ExcelWriter('r3_sentiment_text_analysis.xlsx', engine='openpyxl')

    # Function to preprocess text
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        # Convert to lowercase, remove special characters, and normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Apply preprocessing to the text column
    df['processed_text'] = df[text_column].apply(preprocess_text)

    # Function to generate TF-IDF and word clouds
    def analyze_text(data, group_name, sheet_prefix, writer):
        # Separate positive and negative sentiment texts
        pos_texts = data[data['LLM_majority'] == '+']['processed_text'].tolist()
        neg_texts = data[data['LLM_majority'] == '-']['processed_text'].tolist()
        
        if not pos_texts or not neg_texts:
            print(f"No data for {group_name} (Positive or Negative sentiments missing)")
            return
        
        # Combine texts for TF-IDF
        all_texts = pos_texts + neg_texts
        labels = ['Positive'] * len(pos_texts) + ['Negative'] * len(neg_texts)
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores for each term by sentiment
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df['Sentiment'] = labels
        
        pos_tfidf = tfidf_df[tfidf_df['Sentiment'] == 'Positive'][feature_names].mean().sort_values(ascending=False)
        neg_tfidf = tfidf_df[tfidf_df['Sentiment'] == 'Negative'][feature_names].mean().sort_values(ascending=False)
        
        # Save top 20 terms for each sentiment
        pos_tfidf_df = pd.DataFrame(pos_tfidf.head(20), columns=['TF-IDF Score'])
        neg_tfidf_df = pd.DataFrame(neg_tfidf.head(20), columns=['TF-IDF Score'])
        pos_tfidf_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Positive_Terms')
        neg_tfidf_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Negative_Terms')
        
        print(f"\nTop Terms for {group_name} - Positive Sentiment:")
        print(pos_tfidf_df)
        print(f"\nTop Terms for {group_name} - Negative Sentiment:")
        print(neg_tfidf_df)
        
        # Generate word clouds
        pos_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pos_tfidf)
        neg_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(neg_tfidf)
        
        # Save word clouds
        plt.figure(figsize=(10, 5))
        plt.imshow(pos_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Positive Sentiment Word Cloud - {group_name}')
        plt.savefig(f'wordcloud_positive_{group_name.replace("/", "_")}.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.imshow(neg_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Negative Sentiment Word Cloud - {group_name}')
        plt.savefig(f'wordcloud_negative_{group_name.replace("/", "_")}.png')
        plt.close()

    # 1. Global analysis
    print("\n1. Global Sentiment Text Analysis:")
    analyze_text(df, "Global", "Global", writer)

    # 2. Analysis by source
    print("\n2. Sentiment Text Analysis by Source:")
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        analyze_text(source_df, source, source, writer)

    # 3. Analysis by hashtag/channel within each source
    print("\n3. Sentiment Text Analysis by Hashtag/Channel within Each Source:")
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        hashtag_counts = source_df['hashtag_chat'].value_counts()
        valid_hashtags = hashtag_counts[hashtag_counts >= 200].index
        
        for hashtag in valid_hashtags:
            hashtag_df = source_df[source_df['hashtag_chat'] == hashtag]
            hashtag_name = hashtag.replace("/", "_")  # Replace invalid characters for sheet names
            analyze_text(hashtag_df, f"{source}/{hashtag_name}", f"{source}_{hashtag_name}", writer)

    # Save the Excel file
    writer.close()