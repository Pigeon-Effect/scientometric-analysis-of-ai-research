# Updated script for human-in-the-loop clustering with improved naming and outlier filtering
import os
import json
import numpy as np
import pandas as pd
import sqlite3
import torch
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import umap.umap_ as umap
import plotly.express as px

# Configuration
DB_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\computer_science_sample.db"
OUTPUT_DIR = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_computer_science\resources"
UMAP_HTML_PATH = os.path.join(OUTPUT_DIR, "interactive_umap.html")
ABSTRACT_JSON_PATH = os.path.join(OUTPUT_DIR, "cluster_abstracts_summary.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Functions
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text

def name_clusters(vectorizer, tfidf_matrix, cluster_assignments, top_n_words=10, distinctiveness_weight=0.3):
    feature_names = vectorizer.get_feature_names_out()
    n_clusters = cluster_assignments.max() + 1
    cluster_centers = np.zeros((n_clusters, tfidf_matrix.shape[1]))
    for i in range(n_clusters):
        cluster_centers[i] = tfidf_matrix[cluster_assignments == i].mean(axis=0)
    term_presence = (cluster_centers > np.percentile(cluster_centers, 75, axis=1)[:, None]).astype(int)
    term_presence_sum = term_presence.sum(axis=0)
    icf = np.log((n_clusters + 1) / (1 + term_presence_sum))
    normalized_tf = cluster_centers / cluster_centers.max(axis=1, keepdims=True)
    normalized_icf = icf / icf.max()
    combined_scores = (distinctiveness_weight * normalized_icf * cluster_centers +
                       (1 - distinctiveness_weight) * normalized_tf * cluster_centers)
    min_presence = 0.1
    cluster_names = {}
    for i in range(n_clusters):
        top_indices = combined_scores[i].argsort()[::-1]
        top_terms = []
        used = set()
        for idx in top_indices:
            term = feature_names[idx]
            if term.isdigit() or len(term) < 3 or term in used:
                continue
            if cluster_centers[i][idx] < min_presence * cluster_centers[i].max():
                continue
            top_terms.append(term)
            used.add(term)
            if len(top_terms) >= top_n_words:
                break
        cluster_names[i] = " / ".join(top_terms)
        print(f"Cluster {i}: {cluster_names[i]}")
    return cluster_names

def export_cluster_summaries(df, embeddings, cluster_col, text_col, title_col, cluster_name_col, output_folder,
                              n_central=20, n_random=100):
    summaries = {}
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id]
        indices = cluster_df.index.tolist()
        cluster_embeddings = embeddings[indices]
        cluster_center = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        sorted_indices = np.argsort(distances)
        n_central_actual = min(n_central, len(indices))
        n_random_actual = min(n_random, len(indices))
        central_indices = [indices[i] for i in sorted_indices[:n_central_actual]]
        random_indices = list(np.random.choice(indices, size=n_random_actual, replace=False))
        central_docs = df.loc[central_indices, [title_col, text_col]].to_dict(orient='records')
        random_docs = df.loc[random_indices, [title_col, text_col]].to_dict(orient='records')
        summaries[str(cluster_id)] = {
            'cluster_name': cluster_df[cluster_name_col].iloc[0],
            'num_documents': len(cluster_df),
            'most_central_abstracts': central_docs,
            'random_abstracts': random_docs
        }
    with open(ABSTRACT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"[✓] JSON written to: {ABSTRACT_JSON_PATH}")

def plot_interactive_umap(df, cluster_col, cluster_name_col, x_col, y_col, output_file):
    custom_colors = [
        "#FF0000", "#0000FF", "#00FF00", "#FF7F00", "#8A2BE2", "#FF1493",
        "#00CED1", "#32CD32", "#B22222", "#4169E1", "#FFD700", "#DC143C",
        "#9932CC", "#00FA9A", "#FF6347", "#1E90FF", "#ADFF2F", "#8B0000",
        "#00BFFF", "#FF69B4", "#2E8B57", "#FA8072", "#9370DB", "#40E0D0",
        "#FF4500", "#7FFF00", "#D2691E", "#6A5ACD", "#20B2AA", "#FF00FF"
    ]
    df_plot = df.copy()
    df_plot['cluster_label'] = df_plot[cluster_col].astype(str) + ": " + df_plot[cluster_name_col]
    unique_cluster_ids = sorted(df[cluster_col].unique())
    color_discrete_map = {str(cluster_id) + ": " + df[df[cluster_col] == cluster_id][cluster_name_col].iloc[0]:
                         custom_colors[i % len(custom_colors)] for i, cluster_id in enumerate(unique_cluster_ids)}
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color='cluster_label',
        color_discrete_map=color_discrete_map,
        hover_data={'cluster_label': True},
        height=900,
        width=1200,
        title="Interactive UMAP Cluster Visualization"
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family="Times New Roman",
        title_font_family="Times New Roman",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(family="Times New Roman", size=10)
        )
    )
    fig.update_xaxes(gridcolor='lightgray', gridwidth=1, showgrid=True, title_font_family="Times New Roman")
    fig.update_yaxes(gridcolor='lightgray', gridwidth=1, showgrid=True, title_font_family="Times New Roman")
    fig.write_html(output_file)
    print(f"[✓] Interactive UMAP written to: {output_file}")

# Main Execution
if __name__ == "__main__":
    # Load and preprocess
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT title, cleaned_abstract FROM works_labeled WHERE cleaned_abstract IS NOT NULL AND language = 'en'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['processed_text'] = (df['title'].fillna('') + '. ' + df['cleaned_abstract']).apply(preprocess_text)

    # Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('allenai-specter', device=device)
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    window_size = 100
    stride = 50
    all_embeddings = []
    for text in tqdm(df['processed_text'], desc="Generating Embeddings"):
        tokens = tokenizer.tokenize(text)
        chunks = [tokens[i:i + window_size] for i in range(0, len(tokens), stride)]
        chunk_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks if chunk]
        if not chunk_texts:
            all_embeddings.append(model.encode([''], device=device))
        else:
            chunk_embeddings = model.encode(chunk_texts, device=device)
            all_embeddings.append(np.mean(chunk_embeddings, axis=0))
    embeddings = np.vstack(all_embeddings)

    # Clustering
    n_clusters = 50
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # UMAP + Improved Outlier Filtering
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    df[['x', 'y']] = umap_model.fit_transform(embeddings)
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    mask = iso_forest.fit_predict(df[['x', 'y']])
    df = df[mask == 1].reset_index(drop=True)
    embeddings = embeddings[mask == 1]
    df[['x', 'y']] = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings)

    # TF-IDF Cluster Naming
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(df['processed_text'])
    cluster_names = name_clusters(vectorizer, X_tfidf, df['cluster'].values)
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Export
    export_cluster_summaries(df, embeddings, 'cluster', 'cleaned_abstract', 'title', 'cluster_name', OUTPUT_DIR)
    plot_interactive_umap(df, 'cluster', 'cluster_name', 'y', 'x', UMAP_HTML_PATH)
