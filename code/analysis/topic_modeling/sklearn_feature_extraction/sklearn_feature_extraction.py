import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from tqdm import tqdm
import umap.umap_ as umap
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import scipy.sparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define output directory
output_dir = r"C:\Users\Admin\Documents\Master-Thesis\results\topic_modeling\sklearn_feature_extraction"
os.makedirs(output_dir, exist_ok=True)

# Custom color palette with enough colors for up to 30 clusters
custom_colors = [
    "#FF4500", "#ADFF2F", "#1E90FF", "#FFD700", "#FF1493",
    "#40E0D0", "#FF7F00", "#7FFF00", "#007FFF", "#FF007F",
    "#7F00FF", "#00FF7F", "#FFBF00", "#BF00FF", "#00FFBF",
    "#8A2BE2", "#FF69B4", "#00FA9A", "#D2691E", "#4682B4",
    "#A52A2A", "#5F9EA0", "#B22222", "#228B22", "#DAA520",
    "#20B2AA", "#9932CC", "#FF6347", "#3CB371", "#B8860B",
    "#4169E1"
]


def decompress_abstract(json_str):
    """Convert inverted index to full text"""
    inverted_index = json.loads(json_str)
    position_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word.append((pos, word))
    position_word.sort()
    words = [w for _, w in position_word]
    return ' '.join(words)


def load_documents(db_path):
    """Load documents from SQLite database with titles"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
    SELECT title, abstract_inverted_index 
    FROM works 
    WHERE abstract_inverted_index IS NOT NULL AND language = 'en'
    """
    rows = cursor.execute(query).fetchall()
    documents = []
    for title, abstract in rows:
        if title and abstract:
            full_text = f"{title} {title} {decompress_abstract(abstract)}"  # Title appears twice for 2x weight
            documents.append(full_text)
    conn.close()
    return documents


def preprocess_text(text):
    """Enhanced text preprocessing with phrase handling"""
    try:
        if not isinstance(text, str) or pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()


        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove very short words (1-2 characters)
        text = ' '.join([word for word in text.split() if len(word) > 2])

        return text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""


def name_clusters(vectorizer, kmeans, top_n_words=5):
    """Generate more meaningful names for clusters"""
    feature_names = vectorizer.get_feature_names_out()
    cluster_names = {}

    for i in range(len(kmeans.cluster_centers_)):
        # Get top terms
        top_terms = []
        term_indices = kmeans.cluster_centers_[i].argsort()[::-1]  # Sort descending

        for ind in term_indices:
            term = feature_names[ind]
            if not term.isdigit():
                top_terms.append(term)
                if len(top_terms) >= top_n_words:
                    break

        # Create a more readable name
        cluster_name = " / ".join(top_terms)
        cluster_names[i] = cluster_name  # Remove "Cluster {i + 1}:" prefix

    return cluster_names


def visualize_clusters(df, vectorizer, kmeans, cluster_names, output_path):
    """Enhanced visualization with better cluster representation"""
    print("\nCreating UMAP visualization...")

    # First apply LSA (Truncated SVD) to reduce dimensionality before UMAP
    svd = TruncatedSVD(100)  # Reduce to 100 dimensions first
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = vectorizer.transform(df['processed_text'])
    X_lsa = lsa.fit_transform(X)

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )

    print("Running UMAP dimensionality reduction...")
    embedding = reducer.fit_transform(X_lsa)

    # Add to dataframe
    df['x'] = embedding[:, 0]
    df['y'] = embedding[:, 1]
    df['cluster'] = kmeans.labels_
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Count papers per cluster and sort by count
    cluster_counts = df['cluster_name'].value_counts().to_dict()
    sorted_clusters = sorted(cluster_counts.keys(), key=lambda x: -cluster_counts[x])

    # Create the visualization
    plt.figure(figsize=(20, 16))

    # Plot each cluster in order of frequency
    for i, cluster in enumerate(sorted_clusters):
        subset = df[df['cluster_name'] == cluster]
        if len(subset) > 0:
            plt.scatter(
                subset['x'],
                subset['y'],
                color=custom_colors[i % len(custom_colors)],
                label=f"{cluster} ({len(subset)})",
                alpha=0.6,
                s=10
            )

    plt.title('UMAP Visualization of AI Paper Clusters (23 Clusters)', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)

    # Adjust legend - moved outside the plot area to avoid covering data
    legend = plt.legend(
        bbox_to_anchor=(1.02, 1),  # Place legend outside the right of the plot
        loc='upper left',
        borderaxespad=0.5,
        fontsize=10,
        title='Clusters (Count)',
        title_fontsize=12,
        markerscale=2.0  # Increase marker size in legend
    )

    # Make legend background slightly transparent
    legend.get_frame().set_alpha(0.8)

    plt.tight_layout()

    # Save as SVG
    svg_path = os.path.join(output_path, r'u_map\ai_papers_clusters_umap.svg')
    plt.savefig(svg_path, format='svg', dpi=1200, bbox_inches='tight')
    print(f"\nSaved UMAP visualization to {svg_path}")

    plt.show()

    return df


def print_top_words_per_cluster(vectorizer, kmeans, top_n_words=10):
    """Print the most important words for each cluster to the console."""
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop words per cluster:")
    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[::-1][:top_n_words]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"Cluster {i + 1}: {', '.join(top_words)}")


def detect_and_remove_outliers(df, X, contamination=0.05):
    """
    Enhanced outlier detection combining multiple methods to catch spatial outliers
    that cause UMAP visualization issues

    Args:
        df: DataFrame with documents
        X: TF-IDF matrix
        contamination: Expected proportion of outliers (default 5%)

    Returns:
        df_clean: DataFrame without outliers
        outlier_indices: Indices of detected outliers
    """
    print(f"\nDetecting outliers using multiple methods...")

    # Method 1: Text length outliers (very short or very long)
    text_lengths = df['processed_text'].str.len()
    length_q1, length_q3 = text_lengths.quantile([0.05, 0.95])
    length_outliers = set(df[(text_lengths < length_q1) | (text_lengths > length_q3)].index)
    print(f"Text length outliers: {len(length_outliers)} (too short: {len(df[text_lengths < length_q1])}, too long: {len(df[text_lengths > length_q3])})")

    # Method 2: Word count outliers
    word_counts = df['processed_text'].str.split().str.len()
    word_q1, word_q3 = word_counts.quantile([0.01, 0.99])
    word_outliers = set(df[(word_counts < word_q1) | (word_counts > word_q3)].index)
    print(f"Word count outliers: {len(word_outliers)} (too few: {len(df[word_counts < word_q1])}, too many: {len(df[word_counts > word_q3])})")

    # Method 3: Vocabulary diversity outliers (documents with very few unique words)
    unique_word_ratios = df['processed_text'].apply(lambda x: len(set(x.split())) / max(len(x.split()), 1))
    diversity_outliers = set(df[unique_word_ratios < 0.3].index)  # Less than 30% unique words
    print(f"Low vocabulary diversity outliers: {len(diversity_outliers)}")

    # Method 4: TF-IDF sparsity outliers (documents with very few non-zero features)
    feature_counts = np.array((X > 0).sum(axis=1)).flatten()
    sparse_threshold = np.percentile(feature_counts, 5)  # Bottom 5%
    sparse_outliers = set(np.where(feature_counts < sparse_threshold)[0])
    print(f"TF-IDF sparsity outliers: {len(sparse_outliers)} (fewer than {sparse_threshold:.0f} features)")

    # Method 5: Isolation Forest on TF-IDF (more aggressive)
    isolation_forest = IsolationForest(
        contamination=contamination * 2,  # More aggressive
        random_state=42,
        n_estimators=200
    )
    outlier_labels = isolation_forest.fit_predict(X)
    isolation_outliers = set(np.where(outlier_labels == -1)[0])
    print(f"Isolation Forest outliers: {len(isolation_outliers)}")

    # Method 6: Statistical outliers based on document similarity to corpus center
    # Calculate mean document vector
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    corpus_center = np.mean(X_dense, axis=0)

    # Calculate cosine similarities to corpus center
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(X_dense, corpus_center.reshape(1, -1)).flatten()
    similarity_threshold = np.percentile(similarities, 2)  # Bottom 2%
    similarity_outliers = set(np.where(similarities < similarity_threshold)[0])
    print(f"Low similarity outliers: {len(similarity_outliers)} (similarity < {similarity_threshold:.3f})")

    # Combine all outlier detection methods
    all_outliers = length_outliers | word_outliers | diversity_outliers | sparse_outliers | isolation_outliers | similarity_outliers
    print(f"\nTotal unique outliers detected: {len(all_outliers)} ({len(all_outliers)/len(df)*100:.2f}%)")

    # Get inlier indices
    inlier_indices = [i for i in range(len(df)) if i not in all_outliers]

    # Print some example outliers for inspection
    if len(all_outliers) > 0:
        print("\nExample outlier abstracts (first 5):")
        outlier_list = list(all_outliers)[:5]
        for i, idx in enumerate(outlier_list):
            text_preview = df.iloc[idx]['processed_text'][:150] + "..." if len(df.iloc[idx]['processed_text']) > 150 else df.iloc[idx]['processed_text']
            print(f"Outlier {i+1} (len={len(df.iloc[idx]['processed_text'])}): {text_preview}")

    # Return cleaned dataframe
    df_clean = df.iloc[inlier_indices].copy().reset_index(drop=True)

    return df_clean, list(all_outliers)


def main():
    # Load and preprocess the data
    db_path = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\openalex_ai_works_sample_100k.db"
    print(f"Loading data from {db_path}...")
    documents = load_documents(db_path)

    # Create DataFrame
    df = pd.DataFrame({'full_text': documents})

    # Preprocess text with enhanced preprocessing
    print("Preprocessing text...")
    df['processed_text'] = [preprocess_text(text) for text in tqdm(df['full_text'], desc="Processing")]

    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]

    print(f"\nSuccessfully loaded {len(df)} papers with valid text.")

    # Vectorize the text with more specific parameters
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=20,
        stop_words='english',
        max_features=15000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df['processed_text'])

    # Detect and remove outliers
    df_clean, outlier_indices = detect_and_remove_outliers(df, X, contamination=0.03)
    print(f"Proceeding with {len(df_clean)} documents after outlier removal.")

    # Re-vectorize the cleaned data
    X_clean = vectorizer.fit_transform(df_clean['processed_text'])

    # Fixed number of clusters
    n_clusters = 23
    print(f"\nRunning KMeans clustering with fixed k={n_clusters}...")

    # Perform final clustering on clean data
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        init='k-means++',
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X_clean)

    # Name the clusters
    cluster_names = name_clusters(vectorizer, kmeans, top_n_words=4)

    # Visualize clusters with clean data
    df_clean = visualize_clusters(df_clean, vectorizer, kmeans, cluster_names, output_dir)

    # Print top words per cluster
    print_top_words_per_cluster(vectorizer, kmeans, top_n_words=10)

if __name__ == "__main__":
    main()