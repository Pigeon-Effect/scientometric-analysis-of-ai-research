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
from sklearn.neighbors import LocalOutlierFactor
import umap.umap_ as umap
import plotly.express as px


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text


def name_clusters(vectorizer, tfidf_matrix, cluster_assignments, top_n_words=4, distinctiveness_weight=0.7):
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

    min_presence = 0.15
    cluster_names = {}
    for i in range(n_clusters):
        top_indices = combined_scores[i].argsort()[::-1]
        top_terms = []
        for idx in top_indices:
            term = feature_names[idx]
            if (term.isdigit() or len(term) < 3):
                continue
            if cluster_centers[i][idx] < min_presence * cluster_centers[i].max():
                continue
            top_terms.append(term)
            if len(top_terms) >= top_n_words:
                break
        cluster_names[i] = " / ".join(top_terms)
    return cluster_names


def export_cluster_summaries(df, embeddings, cluster_col, text_col, title_col, cluster_name_col, output_folder,
                              n_central=20, n_random=100):
    os.makedirs(output_folder, exist_ok=True)
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

    output_path = os.path.join(output_folder, "cluster_abstracts_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"[✓] JSON written to: {output_path}")


def plot_interactive_umap(df, cluster_col, cluster_name_col, x_col, y_col, output_file):
    # Create a custom color palette with maximally different colors
    custom_colors = [
        "#FF0000",  # Red
        "#0000FF",  # Blue
        "#00FF00",  # Green
        "#FF7F00",  # Orange
        "#8A2BE2",  # Blue Violet
        "#FF1493",  # Deep Pink
        "#00CED1",  # Dark Turquoise
        "#32CD32",  # Lime Green
        "#B22222",  # Fire Brick
        "#4169E1",  # Royal Blue
        "#FFD700",  # Gold
        "#DC143C",  # Crimson
        "#9932CC",  # Dark Orchid
        "#00FA9A",  # Medium Spring Green
        "#FF6347",  # Tomato
        "#1E90FF",  # Dodger Blue
        "#ADFF2F",  # Green Yellow
        "#8B0000",  # Dark Red
        "#00BFFF",  # Deep Sky Blue
        "#FF69B4",  # Hot Pink
        "#2E8B57",  # Sea Green
        "#FA8072",  # Salmon
        "#9370DB",  # Medium Purple
        "#40E0D0",  # Turquoise
        "#FF4500",  # Orange Red
        "#7FFF00",  # Chartreuse
        "#D2691E",  # Chocolate
        "#6A5ACD",  # Slate Blue
        "#20B2AA",  # Light Sea Green
        "#FF00FF"   # Magenta
    ]

    # Create numeric cluster labels with names for better identification
    df_plot = df.copy()
    df_plot['cluster_label'] = df_plot[cluster_col].astype(str) + ": " + df_plot[cluster_name_col]

    # Create color mapping for clusters using the numeric cluster IDs
    unique_cluster_ids = sorted(df[cluster_col].unique())
    color_discrete_map = {str(cluster_id) + ": " + df[df[cluster_col] == cluster_id][cluster_name_col].iloc[0]:
                         custom_colors[i % len(custom_colors)]
                         for i, cluster_id in enumerate(unique_cluster_ids)}

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

    # Update axes to have white background
    fig.update_xaxes(
        gridcolor='lightgray',
        gridwidth=1,
        showgrid=True,
        title_font_family="Times New Roman"
    )
    fig.update_yaxes(
        gridcolor='lightgray',
        gridwidth=1,
        showgrid=True,
        title_font_family="Times New Roman"
    )

    fig.write_html(output_file)
    print(f"[✓] Interactive UMAP written to: {output_file}")


# ---------------- MAIN SCRIPT ----------------
if __name__ == "__main__":
    db_path = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\openalex_ai_filtered_dataset_sample_100k.db"
    output_dir = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\sklearn_feature_extraction"
    cluster_output_dir = os.path.join(output_dir, "u_map", "umap_scibert", "cluster_identifier_abstracts")
    umap_html_path = os.path.join(output_dir, "u_map", "umap_scibert", "semi_supervised_clusters_umap", "interactive_umap.html")

    os.makedirs(os.path.dirname(umap_html_path), exist_ok=True)
    os.makedirs(cluster_output_dir, exist_ok=True)

    # 1. Load data
    conn = sqlite3.connect(db_path)
    query = "SELECT title, cleaned_abstract FROM works WHERE cleaned_abstract IS NOT NULL AND language = 'en'"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['processed_text'] = (df['title'].fillna('') + '. ' + df['cleaned_abstract']).apply(preprocess_text)

    # 2. SciBERT embeddings with sliding windows
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

    # 3. Clustering
    n_clusters = 50
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # 4. Dimensionality reduction (UMAP) + LOF
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    df[['x', 'y']] = umap_model.fit_transform(embeddings)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    mask = lof.fit_predict(df[['x', 'y']])
    df = df[mask != -1].reset_index(drop=True)
    embeddings = embeddings[mask != -1]
    df[['x', 'y']] = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings)

    # 5. TF-IDF & Cluster Naming
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(df['processed_text'])
    cluster_names = name_clusters(vectorizer, X_tfidf, df['cluster'].values)
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # 6. Export JSON summaries
    export_cluster_summaries(
        df,
        embeddings,
        cluster_col='cluster',
        text_col='cleaned_abstract',
        title_col='title',
        cluster_name_col='cluster_name',
        output_folder=cluster_output_dir
    )

    # 7. Plot interactive UMAP
    plot_interactive_umap(
        df,
        cluster_col='cluster',
        cluster_name_col='cluster_name',
        x_col='y',
        y_col='x',
        output_file=umap_html_path
    )
