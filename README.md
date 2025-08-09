# 11531666_SAN_ERP
This script contains the complete, reproducible Python code used for the analysis of YouTube comments. It is designed to be self-contained and heavily annotated to meet the reproducibility and clarity requirements of the project's additional materials.


# ==============================================================================
# MSc Data Science - Extended Research Project
# Technical Appendix: Python Code for Data Analysis
#
# This script contains the complete, reproducible Python code used for the
# analysis of YouTube comments. It is designed to be self-contained and
# heavily annotated to meet the reproducibility and clarity requirements
# of the project's additional materials.
#
# The code performs the following key tasks:
# 1. Data Collection: Fetches top-level comments from YouTube videos related to
#    a specific policy topic using the YouTube Data API v3.
# 2. Text Preprocessing: Cleans, tokenizes, and stems the collected comments.
# 3. Exploratory Analysis: Visualizes word frequencies, bigram networks, and
#    performs Topic Modeling using Non-negative Matrix Factorization (NMF).
# 4. Sentiment Analysis: Compares multiple sentiment models (VADER, TextBlob,
#    BERT) and trains a custom Logistic Regression classifier.
# 5. Model Comparison: Evaluates the performance of all models to identify the
#    best-performing one for a final deep-dive analysis.
#
# To run this script:
# - First, set up the environment and dependencies. Please run the `requirements.txt` file by running this command in your terminal: 'pip install -r requirements.txt'
#   This file is included in the same folder as this script.
# - Ensure all required libraries are installed by running:
#   pip install -r requirements.txt
# - Securely provide your YouTube Data API key. This script expects the key to
#   be set as an environment variable named 'YOUTUBE_API_KEY'. This prevents
#   your key from being exposed in a public repository.
# - For the Logistic Regression section, this script loads a
#   'manual_checked_sample.csv' file for training data. When replicating this code, please generate a manually checked sample dataset. If this file is not
#   found, the script will gracefully skip that part of the analysis.
#
# ==============================================================================

# --- 0. Import Required Libraries ---------------------------------------------
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from matplotlib import cm
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# --- 1. Data Collection -------------------------------------------------------

def get_youtube_api_client():
    """
    Initializes and returns a YouTube Data API client, fetching the API key
    from an environment variable for security and reproducibility.
    """
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable not set.")
        print("Please set this variable with your YouTube Data API key.")
        return None

    try:
        youtube_client = build('youtube', 'v3', developerKey=api_key)
        return youtube_client
    except Exception as e:
        print(f"Error initializing YouTube API client: {e}")
        return None

# Helper Function: Search for YouTube Videos
def search_videos(youtube_client, query, max_results=5):
    """
    Searches YouTube for videos matching a given query, focusing on UK region and English language.
    Returns a list of dictionaries with video details.
    """
    if not youtube_client:
        return []

    video_details = []
    try:
        search_response = youtube_client.search().list(
            q=query,
            type="video",
            part="id, snippet",
            maxResults=max_results,
            relevanceLanguage="en",
            regionCode="GB"
        ).execute()

        for item in search_response.get("items", []):
            if 'id' in item and 'videoId' in item['id']:
                video_details.append({
                    "video_id": item['id']['videoId'],
                    "title": item['snippet']['title'],
                    "description": item['snippet']['description'],
                    "published_at": item['snippet']['publishedAt']
                })
            else:
                print(f"Warning: Skipping invalid item (no video ID) for query '{query}'.")
    except HttpError as e:
        print(f"API Error during search for '{query}': {str(e)}")
    except Exception as e:
        print(f"Unexpected error during search for '{query}': {str(e)}")
    return video_details

# Helper Function: Retrieve Video Comments
def get_video_comments(youtube_client, video_id, max_comments=2000):
    """
    Fetches top-level comments from a YouTube video, handling pagination.
    Returns a list of dictionaries with comment details.
    """
    if not youtube_client:
        return []

    comments = []
    next_page_token = None
    try:
        while len(comments) < max_comments:
            response = youtube_client.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_id": item["id"],
                    "text": top_comment["textDisplay"],
                    "author": top_comment["authorDisplayName"],
                    "published_at": top_comment["publishedAt"],
                    "like_count": top_comment["likeCount"]
                })

            if len(comments) >= max_comments:
                break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as e:
        if "commentsDisabled" in str(e):
            print(f"Warning: Comments are disabled for video {video_id}. Skipping.")
        else:
            print(f"API Error for video {video_id}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error for video {video_id}: {str(e)}")
    return comments

# Main workflow to collect all comments
def collect_data(keywords):
    """
    Main function to orchestrate the data collection process.
    """
    print("--- Starting Data Collection ---")
    youtube_client = get_youtube_api_client()
    if not youtube_client:
        return pd.DataFrame(), pd.DataFrame()

    all_comments = []
    all_video_details = []

    for keyword in keywords:
        print(f"\nSearching for videos with keyword: '{keyword}'")
        video_details = search_videos(youtube_client, query=keyword, max_results=5)
        all_video_details.extend(video_details)
        
        if not video_details:
            print("No videos found for this keyword.")
        else:
            print(f"\n--- Displaying Video Metadata for keyword: '{keyword}' ---")
            for detail in video_details:
                print(f"Video ID: {detail['video_id']}")
                print(f"Title: {detail['title']}")
                print(f"Published: {detail['published_at']}")
                print(f"Description: {detail['description'][:100]}...")
                print("-" * 20)
        
        for detail in video_details:
            video_id = detail['video_id']
            print(f"Collecting comments from video: {video_id}")
            video_comments = get_video_comments(youtube_client, video_id=video_id, max_comments=2000)
            all_comments.extend(video_comments)

    df_videos = pd.DataFrame(all_video_details)
    df_raw_comments = pd.DataFrame(all_comments)
    return df_videos, df_raw_comments

# --- Main execution block -----------------------------------------------------

if __name__ == "__main__":
    # Define Search Keywords
    KEYWORDS = [
        'maths to 18', 'Sunak education policy', 'post-16 mathematics reform',
        "sunak math policy", "math until 18", "math to 18",
        "extended math education", "compulsory math 18",
        "sunak education reform", "uk math extension",
        "british math curriculum to 18", "England math requirement sunak",
        "Sunak school reform"
    ]
    
    # Collect and save data
    df_videos, df_raw_comments = collect_data(KEYWORDS)

    if not df_raw_comments.empty:
        # Convert raw comments to a DataFrame
        df_raw_comments = pd.DataFrame(df_raw_comments)
        
        # Save raw data to a platform-agnostic path
        output_path_raw_comments = os.path.join("data", "youtube_comments_metadata_raw.csv")
        os.makedirs("data", exist_ok=True)
        df_raw_comments.to_csv(output_path_raw_comments, index=False)
        print(f"\nRaw collected comments (before filtering/cleaning) saved to: {output_path_raw_comments}")

        # Remove duplicates and prepare the main DataFrame
        df_main = df_raw_comments.drop_duplicates(subset=["text"]).copy()
        print(f"Removed duplicates. Retained {len(df_main)} unique comments for analysis.")

        # Filter comments by date range
        print("\n--- Filtering comments by date range: 2023.01.01 to 2023.12.31 ---")
        df_main['published_at'] = pd.to_datetime(df_main['published_at']).dt.tz_convert('UTC')
        start_date = pd.to_datetime('2023-01-01', utc=True)
        end_date = pd.to_datetime('2023-12-31', utc=True)
        df_main = df_main[(df_main['published_at'] >= start_date) & (df_main['published_at'] <= end_date)].reset_index(drop=True)
        print(f"Retained {len(df_main)} comments after date filtering.")
        
        # Save filtered data
        output_path_filtered_comments = os.path.join("data", "youtube_comments_2023.csv")
        df_main.to_csv(output_path_filtered_comments, index=False)
        print(f"Filtered comments for 2023 saved to: {output_path_filtered_comments}")
        
        # --- 2. Text Preprocessing and Configuration ----------------------------------

        # Configuration dictionary for all analysis steps
        config = {
            'target_terms': ["sunak", "rishi", "math", "maths", "mathematics", "A-level", "post-16"],
            'stopwords': set(stopwords.words('english')) | {'video', 'comment', 'like', 'subscribe', 'youtube', 'http', 'www', 't', 's'},
            'bigram_min_count': 10,
            'top_n_bigrams': 30,
            'top_n_words': 20,
            'nmf_topic_range': range(3, 8),
            'sentiment_cutoffs': {'positive': 0.05, 'negative': -0.05},
            'stems': {},  # Placeholder for custom stemmed lexicon
            'bert_batch_size': 32
        }

        # Text Preprocessing Functions
        def preprocess_text(text):
            """
            Cleans and tokenizes text by removing special characters, URLs, and stopwords, then performs stemming.
            """
            stemmer = SnowballStemmer('english')
            if not isinstance(text, str):
                return []
            text = re.sub(r'http\S+|@\w+|#\w+', '', text)
            text = re.sub(r'[^A-Za-z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            tokens = [stemmer.stem(t.lower()) for t in word_tokenize(text) if t.isalpha()]
            tokens = [t for t in tokens if t not in config['stopwords'] and len(t) > 2]
            return tokens

        def clean_text_base(text):
            """Basic cleaning for non-stemmed text, used for models like BERT."""
            if not isinstance(text, str):
                return ""
            text = re.sub(r'http\S+|@\w+|#\w+', '', text)
            text = re.sub(r'[^A-Za-z0-9\s.,?!]', '', text)
            return text.strip()

        # Apply preprocessing to the main dataset
        if not df_main.empty:
            df_main['raw_comment'] = df_main['text'].astype(str).str.strip()
            df_main = df_main[df_main['raw_comment'].str.contains(
                r"\b(?:" + "|".join(map(re.escape, config['target_terms'])) + r")\b", case=False, na=False
            )].reset_index(drop=True)

            df_main['tokens'] = df_main['raw_comment'].apply(preprocess_text)
            df_main['clean_text'] = df_main['tokens'].apply(lambda tk: " ".join(tk))
            df_main['clean_text_base'] = df_main['raw_comment'].apply(clean_text_base)
            df_main = df_main[df_main['clean_text'].str.len() > 0].reset_index(drop=True)
            print(f"\n{len(df_main)} comments after preprocessing and filtering.")
        else:
            print("No data to preprocess.")
            # Create empty dataframe to avoid errors in subsequent steps
            df_main = pd.DataFrame(columns=['raw_comment', 'tokens', 'clean_text', 'clean_text_base'])

        # --- 3. Word Frequency & Bigram Network ---------------------------------------

        # Function to analyze and plot top N most frequent words
        def analyze_word_frequency(df, n_top_words):
            """Generates and plots top N most frequent words from preprocessed tokens."""
            print("\n--- Word Frequency Analysis ---")
            if df.empty:
                print("No data for word frequency analysis.")
                return None
            all_tokens = [token for sublist in df['tokens'] for token in sublist]
            word_counts = Counter(all_tokens)
            top_words_df = pd.DataFrame(word_counts.items(), columns=['word', 'count']).sort_values('count', ascending=False).head(n_top_words)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='count', y='word', data=top_words_df, palette='viridis', hue='word', legend=False)
            for i, (cnt, wd) in enumerate(zip(top_words_df['count'], top_words_df['word'])):
                plt.text(cnt + 1, i, str(cnt), va='center')
            plt.title(f"Top {n_top_words} Most Frequent Words", pad=15)
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            plt.tight_layout()
            plt.show()
            return top_words_df

        # Function to generate and plot bigram network
        def generate_bigrams_detailed(df, top_n_bigrams):
            """Generates a network graph of the most frequent bigrams."""
            print("\n--- Bigram Network Analysis ---")
            if df.empty:
                print("No data for bigram analysis.")
                return None
            bigram_counts = defaultdict(int)
            for tokens in df['tokens']:
                for i in range(len(tokens) - 1):
                    bigram_counts[(tokens[i], tokens[i+1])] += 1

            top_bigrams = dict(sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_bigrams])

            G = nx.Graph()
            for (w1, w2), cnt in top_bigrams.items():
                G.add_edge(w1, w2, weight=cnt)

            pos = nx.spring_layout(G, k=0.8, seed=42, iterations=50)
            max_weight = max(top_bigrams.values()) if top_bigrams else 1
            node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]
            edge_widths = [5 * G[u][v]['weight'] / max_weight for u, v in G.edges()]

            plt.figure(figsize=(14, 10))
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=cm.Paired(np.linspace(0, 1, len(G))))
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.7)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.5)

            plt.title(f"Top {top_n_bigrams} Frequent Bigrams", pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            return top_bigrams

        if not df_main.empty:
            word_freq = analyze_word_frequency(df_main, config['top_n_words'])
            bigrams = generate_bigrams_detailed(df_main, config['top_n_bigrams'])

        # --- 4. Topic Modeling & Coherence Sweep --------------------------------------

        # Function to perform NMF and coherence score sweep
        def perform_nmf_coherence_sweep(df, topic_range):
            """
            Uses TF-IDF to prepare data for NMF and calculates coherence scores for different
            numbers of topics (k) to find the optimal number of topics.
            """
            print("\n--- Topic Modeling & Coherence Sweep ---")
            if df.empty:
                print("No data for topic modeling.")
                return None, None, None

            tfidf_vec = TfidfVectorizer(min_df=5, max_df=0.3)
            tfidf_matrix = tfidf_vec.fit_transform(df['clean_text'])
            vocab = tfidf_vec.get_feature_names_out()

            # Create a frequency-based vocabulary for coherence scoring
            count_vec = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
            count_matrix = count_vec.fit_transform(df['tokens'])
            freq_vocab = count_vec.get_feature_names_out()
            word_counts = count_matrix.sum(axis=0).A1
            freq_df = pd.DataFrame({'word': freq_vocab, 'count': word_counts}).sort_values('count', ascending=False).reset_index(drop=True)

            ks = list(topic_range)
            coherence_scores = []

            for k in ks:
                nmf = NMF(n_components=k, random_state=42, max_iter=500)
                nmf.fit(tfidf_matrix)
                topics = [[vocab[i] for i in comp.argsort()[:-11:-1]] for comp in nmf.components_]
                
                # Calculate a simple coherence score based on top words
                coh = np.mean([len(set(t) & set(freq_df['word'][:100])) for t in topics])
                coherence_scores.append(coh)
                print(f"k={k}, coherence: {coh:.3f}")

            # Plot coherence scores
            plt.figure(figsize=(8, 5))
            plt.plot(ks, coherence_scores, 'o-', linewidth=2, markersize=6)
            for k, sc in zip(ks, coherence_scores):
                plt.text(k, sc + 0.005, f"{sc:.2f}", ha='center')
            plt.title('Coherence Score vs Number of Topics (k)', pad=15)
            plt.xlabel('Number of Topics (k)')
            plt.ylabel('Coherence Score')
            plt.tight_layout()
            plt.show()

            # Logic to select k based on user's preference
            coherence_df = pd.DataFrame({'k': ks, 'score': coherence_scores}).sort_values(by='score', ascending=False)
            
            best_k = None
            # Check if the highest scoring k is > 5
            if coherence_df.iloc[0]['k'] > 5:
                best_k = int(coherence_df.iloc[0]['k'])
                print(f"Selecting the highest scoring k which is > 5: {best_k}")
            else:
                # If not, find the next highest scoring k > 5
                k_options = coherence_df[coherence_df['k'] > 5]
                if not k_options.empty:
                    best_k = int(k_options.iloc[0]['k'])
                    print(f"The best k <= 5. Selecting the next highest scoring k > 5: {best_k}")
                else:
                    # Fallback to the absolute highest scoring k if no k > 5 is available
                    best_k = int(coherence_df.iloc[0]['k'])
                    print(f"No k > 5 found in the range. Selecting best overall k: {best_k}")

            return best_k, tfidf_matrix, vocab

        # Execute the topic modeling analysis
        if not df_main.empty:
            best_k, tfidf_matrix, vocab = perform_nmf_coherence_sweep(df_main, config['nmf_topic_range'])
        else:
            best_k = None
            tfidf_matrix = None
            vocab = None

        # --- 5. Final NMF & Topic Visualizations --------------------------------------

        # Function to get human-readable topic names
        def get_topic_names(nmf_model, vocab, n_top_words):
            """Assigns human-readable names to topics based on their top terms."""
            topic_names = {}
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [vocab[i] for i in top_words_idx]
                topic_names[topic_idx] = f"Topic {topic_idx}: {', '.join(top_words[:3])}"
            return topic_names

        # Main workflow for final NMF and visualizations
        if best_k and not df_main.empty:
            print("\n--- Final NMF & Topic Visualizations ---")
            nmf_final = NMF(n_components=best_k, random_state=42, max_iter=500)
            topic_matrix = nmf_final.fit_transform(tfidf_matrix)
            df_main['dominant_topic'] = topic_matrix.argmax(axis=1)

            topic_names = get_topic_names(nmf_final, vocab, 10)
            df_main['dominant_topic_name'] = df_main['dominant_topic'].map(topic_names)

            # 5a) Top-terms bar chart per topic
            for t in range(best_k):
                comp = nmf_final.components_[t]
                top_idx = comp.argsort()[-10:]
                terms = vocab[top_idx][::-1]
                weights = comp[top_idx][::-1]
                plt.figure(figsize=(6, 4))
                sns.barplot(x=weights, y=terms, hue=terms, legend=False, palette='viridis')
                for i, w in enumerate(weights):
                    plt.text(w + 0.001, i, f"{w:.2f}", va='center')
                plt.title(f"{topic_names[t]} : Top 10 Terms", pad=10)
                plt.xlabel("Weight")
                plt.tight_layout()
                plt.show()
                
            # Print all top terms for each topic
            print("\n--- Top 10 Terms for Each Topic ---")
            for t in range(best_k):
                comp = nmf_final.components_[t]
                top_idx = comp.argsort()[-10:]
                terms = [vocab[i] for i in top_idx][::-1]
                print(f"{topic_names[t]}: {', '.join(terms)}")


            # 5b) Comments per topic bar chart
            counts = df_main['dominant_topic_name'].value_counts().sort_index()
            plt.figure(figsize=(8, 6))
            sns.barplot(x=counts.index, y=counts.values, palette='tab10', hue=counts.index, legend=False)
            for i, c in enumerate(counts):
                plt.text(i, c + 5, str(c), ha='center')
            plt.title("Number of Comments per Topic", pad=10)
            plt.xlabel("Topic")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            # 5c) WordCloud for each topic
            for t in range(best_k):
                comp = nmf_final.components_[t]
                top_idx = comp.argsort()[-50:]
                freqs = {vocab[i]: float(comp[i]) for i in top_idx}
                wc = WordCloud(width=800, height=400, background_color='white', prefer_horizontal=0.9).generate_from_frequencies(freqs)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word-Cloud for {topic_names[t]}', pad=15)
                plt.tight_layout()
                plt.show()

        # --- 6. Sentiment Analysis with Multiple Models -------------------------------

        class EnhancedSentimentAnalyzer:
            """
            A class to encapsulate and compare multiple sentiment analysis models.
            Includes VADER, TextBlob, and a BERT-based pipeline.
            """
            def __init__(self):
                self.vader = SentimentIntensityAnalyzer()
                self.vader.lexicon.update(config['stems'])
                self.bert_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.bert_name)
                self.bert_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.max_length = self.tokenizer.model_max_length

            def truncate_text(self, text):
                """Truncates text to fit model's maximum sequence length."""
                tokens = self.tokenizer.tokenize(str(text))
                if len(tokens) > self.max_length - 2:
                    tokens = tokens[:self.max_length - 2]
                return self.tokenizer.convert_tokens_to_string(tokens)

            def vader_score(self, text):
                """Generates sentiment score using VADER."""
                score = self.vader.polarity_scores(text)['compound']
                if score > config['sentiment_cutoffs']['positive']:
                    return {'sentiment': 'positive', 'score': score}
                elif score < config['sentiment_cutoffs']['negative']:
                    return {'sentiment': 'negative', 'score': score}
                return {'sentiment': 'neutral', 'score': 0.0}

            def textblob_score(self, text):
                """Generates sentiment score using TextBlob."""
                polarity = TextBlob(str(text)).sentiment.polarity
                if polarity > 0.1:
                    return {'sentiment': 'positive', 'score': polarity}
                elif polarity < -0.1:
                    return {'sentiment': 'negative', 'score': polarity}
                return {'sentiment': 'neutral', 'score': 0.0}

            def bert_score_batch(self, texts):
                """Batch processing for BERT sentiment analysis to improve efficiency."""
                results = []
                for i in range(0, len(texts), config['bert_batch_size']):
                    batch = texts[i:i + config['bert_batch_size']]
                    processed = [self.truncate_text(t) for t in batch]
                    batch_results = self.bert_pipeline(processed, truncation=True, max_length=self.max_length)
                    
                    for res in batch_results:
                        sentiment = 'positive' if res['label'] == 'POSITIVE' else 'negative'
                        score = res['score'] if sentiment == 'positive' else -res['score']
                        if abs(score) < 0.3:
                            sentiment = 'neutral'
                        results.append({'sentiment': sentiment, 'score': score})
                return results

        # Main workflow for sentiment analysis and model comparison
        if not df_main.empty:
            print("\n" + "="*50)
            print("PART 3: SENTIMENT ANALYSIS WITH MODEL COMPARISON")
            print("="*50)

            analyzer = EnhancedSentimentAnalyzer()

            # Process main dataset with all models
            print("Running VADER sentiment analysis...")
            df_main['vader'] = df_main.clean_text_base.apply(analyzer.vader_score)
            print("Running TextBlob sentiment analysis...")
            df_main['textblob'] = df_main.clean_text_base.apply(analyzer.textblob_score)
            print("Running BERT sentiment analysis...")
            bert_results = analyzer.bert_score_batch(df_main.raw_comment.tolist())
            df_main['bert'] = bert_results

            # --- Train Logistic Regression model using the provided manual data ---
            try:
                # Load the manually checked sample dataset for training
                print("\nLoading and training on your manual_checked_sample.csv file...")
                df_manual = pd.read_csv('manual_checked_sample.csv')
                
                # Ensure the 'clean_comment' and 'clean_comment_base' columns are treated as string types before any operations
                for col in ['clean_comment', 'clean_comment_base']:
                    if col in df_manual.columns:
                        df_manual[col] = df_manual[col].astype(str)

                df_manual['manual_label'] = df_manual['manual_label'].str.lower()
                df_manual = df_manual[df_manual['manual_label'].isin(['positive', 'negative', 'neutral'])]
                print(f"Loaded {len(df_manual)} comments from manual_checked_sample.csv for training.")
                
                if not df_manual.empty:
                    # Apply the same text cleaning as the main dataset
                    if 'clean_comment' in df_manual.columns:
                        X_train_data = df_manual['clean_comment'].fillna('').apply(clean_text_base)
                    else:
                        raise KeyError("Column 'clean_comment' not found in manual_check_sample.csv")
                    
                    label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
                    y_train_labels = df_manual['manual_label'].map(label_mapping)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_train_data,
                        y_train_labels,
                        test_size=0.2, random_state=42
                    )
                    
                    # --- Hyperparameter Tuning for Logistic Regression ---
                    print("\nPerforming hyperparameter tuning with GridSearchCV...")
                    
                    # Define the pipeline with a vectorizer and a classifier
                    pipeline_logreg = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('clf', LogisticRegression(max_iter=1000, solver='liblinear')) # Using liblinear for L1/L2 regularization
                    ])
                    
                    # Define the parameter grid to search over
                    param_grid = {
                        'tfidf__ngram_range': [(1, 1), (1, 2)], # Uni-grams and bigrams
                        'clf__C': [0.1, 1, 10, 100] # Inverse of regularization strength
                    }
                    
                    # Instantiate GridSearchCV with the pipeline and parameter grid
                    grid_search = GridSearchCV(
                        pipeline_logreg,
                        param_grid,
                        cv=5, # 5-fold cross-validation
                        verbose=1,
                        n_jobs=-1 # Use all available cores
                    )
                    
                    # Fit the grid search to the training data
                    grid_search.fit(X_train, y_train)

                    print("\nBest parameters found: ", grid_search.best_params_)
                    best_logreg_model = grid_search.best_estimator_

                    print("\nOptimized Logistic Regression Performance on Test Set:")
                    y_pred = best_logreg_model.predict(X_test)
                    print(classification_report(y_test, y_pred))

                    # Apply the trained Logistic Regression model to the full dataset
                    df_main['logreg_pred'] = best_logreg_model.predict(df_main['clean_text_base'].fillna(''))
                    df_main['logreg_prob'] = best_logreg_model.predict_proba(df_main['clean_text_base'].fillna('')).max(axis=1)
                else:
                    print("Not enough manually checked data to train a Logistic Regression model.")
                    df_main['logreg_pred'] = np.nan
                    df_main['logreg_prob'] = np.nan
                
            except FileNotFoundError:
                print("\n'manual_check_sample.csv' not found. Skipping Logistic Regression training and evaluation.")
                df_main['logreg_pred'] = np.nan
                df_main['logreg_prob'] = np.nan
            except Exception as e:
                print(f"\nError training Logistic Regression: {e}. Skipping.")
                df_main['logreg_pred'] = np.nan
                df_main['logreg_prob'] = np.nan

            # Define models and unpack results
            models = ['vader', 'textblob', 'bert']
            if 'logreg_pred' in df_main.columns:
                models.append('logreg')

            for model_name in models:
                if model_name == 'logreg':
                    logreg_mapping = {1: 'positive', 0: 'neutral', -1: 'negative'}
                    df_main[f'{model_name}_sentiment'] = df_main['logreg_pred'].map(logreg_mapping)
                    df_main[f'{model_name}_score'] = df_main['logreg_prob']
                else:
                    df_main[f'{model_name}_sentiment'] = df_main[model_name].apply(lambda x: x['sentiment'])
                    df_main[f'{model_name}_score'] = df_main[model_name].apply(lambda x: x['score'])

            # --- 7. Model Comparison Framework ----------------------------------------
            print("\n" + "="*50)
            print("MODEL COMPARISON ANALYSIS")
            print("="*50)

            # Create a consensus sentiment for comparison
            df_main['consensus_sentiment'] = df_main.apply(
                lambda row: max(
                    [row.get(f'{m}_sentiment') for m in models if pd.notna(row.get(f'{m}_sentiment'))],
                    key=Counter([row.get(f'{m}_sentiment') for m in models if pd.notna(row.get(f'{m}_sentiment'))]).get
                ) if any(pd.notna(row.get(f'{m}_sentiment')) for m in models) else 'neutral',
                axis=1
            )

            model_metrics = {}
            for model in models:
                df_temp = df_main.dropna(subset=[f'{model}_sentiment'])
                if df_temp.empty:
                    continue
                consensus_agreement = (df_temp[f'{model}_sentiment'] == df_temp['consensus_sentiment']).mean()
                avg_confidence = df_temp[f'{model}_score'].abs().mean()
                pred_variance = df_temp.groupby('dominant_topic_name')[f'{model}_score'].var().mean() if 'dominant_topic_name' in df_temp.columns else np.nan
                
                pairwise_agreements = [
                    (df_temp[f'{model}_sentiment'] == df_temp[f'{other_model}_sentiment']).mean()
                    for other_model in models if other_model != model
                ]
                avg_pairwise_agreement = np.mean(pairwise_agreements) if pairwise_agreements else 0.0

                model_metrics[model] = {
                    'consensus_agreement': consensus_agreement,
                    'avg_confidence': avg_confidence,
                    'prediction_variance': pred_variance,
                    'avg_pairwise_agreement': avg_pairwise_agreement,
                }

            metrics_df = pd.DataFrame(model_metrics).T

            # Normalize metrics for comparison
            normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
            normalized['prediction_variance'] = 1 - normalized['prediction_variance']
            normalized['overall_score'] = normalized.mean(axis=1)
            normalized = normalized.sort_values('overall_score', ascending=False)

            print("\nModel Performance Metrics:")
            print(metrics_df.round(4))
            print("\nNormalized Metrics (0-1 scale) with Overall Score:")
            print(normalized.round(4))
            
            # Visualize overall performance
            plt.figure(figsize=(10, 6))
            sns.barplot(x=normalized.index, y='overall_score', data=normalized, palette='viridis', hue=normalized.index, legend=False)
            for i, score in enumerate(normalized['overall_score']):
                plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontweight='bold')
            plt.title("Model Overall Performance Comparison", pad=15)
            plt.xlabel('Model')
            plt.ylabel('Overall Score (0-1)')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()

            best_model = normalized.index[0]
            best_score = normalized['overall_score'].iloc[0]
            print(f"\nBest Performing Model: {best_model.capitalize()} (Overall Score: {best_score:.4f})")
            
            # --- 8. Deep Dive with Best Model -----------------------------------------
            print("\n" + "="*50)
            print(f"DEEP DIVE WITH BEST MODEL: {best_model.upper()}")
            print("="*50)

            # Overall Sentiment Distribution
            plt.figure(figsize=(8, 6))
            sent_counts = df_main[f'{best_model}_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'])
            sns.barplot(x=sent_counts.index, y=sent_counts.values, palette=['green', 'gray', 'red'], hue=sent_counts.index, legend=False)
            for i, v in enumerate(sent_counts):
                if not pd.isna(v):
                    pct = f"{v/len(df_main):.1%}"
                    plt.text(i, v + 5, f"{int(v)} ({pct})", ha='center')
            plt.title(f"Overall Sentiment Distribution ({best_model.capitalize()})", pad=15)
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

            # Sentiment by Topic
            plt.figure(figsize=(12, 8))
            sent_topic = pd.crosstab(df_main['dominant_topic_name'], df_main[f'{best_model}_sentiment'], normalize='index') * 100
            sent_topic.plot(kind='bar', stacked=True, width=0.8, color={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}, ax=plt.gca())
            plt.title(f"Sentiment Distribution by Topic ({best_model.capitalize()})", pad=15)
            plt.xlabel('Topic')
            plt.ylabel('Percentage')
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            # Top Words by Sentiment
            def get_top_words_by_sentiment(sentiment, model, n=15):
                """Gets top words for a specific sentiment and model from the dataset."""
                sent_comments = df_main[df_main[f'{model}_sentiment'] == sentiment]['tokens']
                if sent_comments.empty or all(not comment for comment in sent_comments):
                    return pd.DataFrame({'word': [], 'count': []})
                
                vec = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
                word_counts = vec.fit_transform(sent_comments).sum(axis=0).A1
                vocab = vec.get_feature_names_out()
                top_words = pd.DataFrame({'word': vocab, 'count': word_counts}).sort_values('count', ascending=False).head(n)
                return top_words

            sentiments = ['positive', 'negative', 'neutral']
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            for i, sent in enumerate(sentiments):
                top_words = get_top_words_by_sentiment(sent, best_model)
                palette = 'viridis' if sent == 'positive' else ('Reds_r' if sent == 'negative' else 'coolwarm')
                sns.barplot(x='count', y='word', data=top_words, ax=axes[i], palette=palette, hue='word', legend=False)
                axes[i].set_title(f"Top Words in {sent.capitalize()} Comments ({best_model.capitalize()})", pad=10)
                axes[i].set_xlabel('Frequency')
                axes[i].set_ylabel('Word')
            plt.tight_layout()
            plt.show()

            # --- 9. Export Results ----------------------------------------------------
            # Export final dataframes to CSV files for inclusion in the project appendix.
            print("\n--- Exporting Final Results ---")
            df_main.to_csv('sentiment_analysis_with_model_comparison.csv', index=False)
            metrics_df.to_csv('model_performance_metrics.csv')
            normalized.to_csv('normalized_model_scores.csv')
            print("Analysis complete. Exported files:")
            print("- youtube_comments_metadata_raw.csv")
            print("- youtube_video_metadata.csv")
            print("- youtube_comments_2023.csv")
            print("- sentiment_analysis_with_model_comparison.csv")
            print("- model_performance_metrics.csv")
            print("- normalized_model_scores.csv")
        else:
            print("Script finished without performing full analysis due to a lack of data.")
    else:
        print("No unique comments collected. Cannot proceed with analysis.")
