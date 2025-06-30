import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
from joblib import Parallel, delayed
import multiprocessing
import gc

warnings.filterwarnings("ignore")

# === CONFIG ===
# Switch dataset here:
DB_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\openalex_ai_works_sample_100k.db"

OUTPUT_DIR = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\sklearn_feature_extraction\silhouette_score"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def decompress_abstract(json_str):
    inverted_index = json.loads(json_str)
    position_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word.append((pos, word))
    position_word.sort()
    words = [w for _, w in position_word]
    return ' '.join(words)


def load_documents(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT abstract_inverted_index 
    FROM works 
    WHERE abstract_inverted_index IS NOT NULL AND language = 'en'
    """
    rows = conn.execute(query).fetchall()
    documents = [decompress_abstract(row[0]) for row in rows]
    conn.close()
    return documents


def preprocess_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    phrase_patterns = {
        'machine learning': 'machinelearning',
        'deep learning': 'deeplearning',
        'artificial intelligence': 'artificialintelligence',
        'computer vision': 'computervision',
        'natural language': 'naturallanguage',
        'neural network': 'neuralnetwork',
        'reinforcement learning': 'reinforcementlearning'
    }
    for phrase, replacement in phrase_patterns.items():
        text = text.replace(phrase, replacement)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([w for w in text.split() if len(w) > 2])
    return text


def parallel_preprocess(texts):
    num_cores = multiprocessing.cpu_count()
    processed = Parallel(n_jobs=num_cores)(
        delayed(preprocess_text)(text) for text in texts
    )
    return processed


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=50,
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return X


def compute_k_silhouettes(X, k, n_trials, sil_sample_size):
    scores = []
    # Create tqdm progress bar for trials inside each cluster count k
    for _ in tqdm(range(n_trials), desc=f"Trials for k={k}", leave=False, position=k):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=None,
            batch_size=1000,
            n_init=1
        )
        labels = kmeans.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels, sample_size=sil_sample_size, random_state=None)
            scores.append(score)
    return k, scores


def compute_silhouette_scores_parallel(X, k_values, n_trials=1, sil_sample_size=10000):
    num_cores = min(len(k_values), multiprocessing.cpu_count())
    results = Parallel(n_jobs=num_cores, backend="threading")(
        delayed(compute_k_silhouettes)(X, k, n_trials, sil_sample_size)
        for k in tqdm(k_values, desc="Parallel cluster sizes")
    )
    return dict(results)


def plot_silhouette_scores(results, output_svg):
    k_values = sorted(results.keys())
    means = []
    stds = []
    for k in k_values:
        scores = np.array(results[k])
        means.append(np.nanmean(scores))
        stds.append(np.nanstd(scores))

    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(12, 7))
    plt.plot(k_values, means, marker='o', label='Mean Silhouette Score')
    plt.fill_between(k_values, means - stds, means + stds, color='skyblue', alpha=0.4, label='±1 std. dev.')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_svg, format='svg')
    print(f"Silhouette score plot saved to: {output_svg}")
    plt.show()


def main():
    print(f"Loading documents from {DB_PATH} ...")
    documents = load_documents(DB_PATH)

    df = pd.DataFrame({'abstract': documents})
    df['processed'] = parallel_preprocess(df['abstract'].tolist())
    df = df[df['processed'].str.len() > 0]

    print(f"Vectorizing {len(df)} documents ...")
    X = vectorize_texts(df['processed'].tolist())

    k_values = list(range(5, 31))
    print(f"Computing silhouette scores in parallel for k in {k_values} ...")
    results = compute_silhouette_scores_parallel(X, k_values, n_trials=100, sil_sample_size=10000)

    output_svg = os.path.join(OUTPUT_DIR, 'silhouette_scores_distribution.svg')
    plot_silhouette_scores(results, output_svg)

    del X
    gc.collect()


if __name__ == "__main__":
    main()
