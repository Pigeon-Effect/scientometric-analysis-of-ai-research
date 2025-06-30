import sqlite3
import json
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer


class DistinctiveCTFIDF(ClassTfidfTransformer):
    """Enhanced c-TF-IDF transformer that emphasizes distinctive terms"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        # Get standard c-TF-IDF scores
        tf_idf = super().transform(X)
        distinctiveness = self._calculate_distinctiveness(tf_idf)
        # Ensure distinctiveness is 2D for broadcasting with sparse matrix and convert to CSR
        combined = tf_idf.multiply(distinctiveness.reshape(1, -1)).tocsr()
        return combined

    def _calculate_distinctiveness(self, tf_idf):
        # Calculate distinctiveness score for each term
        # Terms get higher scores when they appear primarily in one topic
        topic_presence = (tf_idf > 0).astype(int)
        term_frequency_across_topics = topic_presence.sum(axis=0)
        distinctiveness = 1 / (term_frequency_across_topics + 1e-12)  # Avoid division by zero
        return distinctiveness


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
    cursor = conn.cursor()
    query = """
    SELECT abstract_inverted_index 
    FROM works 
    WHERE abstract_inverted_index IS NOT NULL AND language = 'en'
    """
    rows = cursor.execute(query).fetchall()
    documents = [decompress_abstract(row[0]) for row in rows]
    conn.close()
    return documents


class KMeansCluster(BaseCluster):
    def __init__(self, nr_clusters=18):
        super().__init__()
        self.kmeans = KMeans(n_clusters=nr_clusters)
        self.labels_ = None

    def fit(self, X):
        self.labels_ = self.kmeans.fit(X).labels_
        return self.labels_

    def fit_predict(self, X):
        self.labels_ = self.kmeans.fit_predict(X)
        return self.labels_

    def predict(self, X):
        return self.kmeans.predict(X)


def main():
    db_path = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\openalex_ai_works_sample_1k.db"
    docs = load_documents(db_path)
    print(f"Loaded {len(docs)} documents.")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    kmeans_model = KMeansCluster(nr_clusters=18)

    # Use our enhanced c-TF-IDF
    ctfidf_model = DistinctiveCTFIDF()

    topic_model = BERTopic(
        embedding_model=embed_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=kmeans_model,
        ctfidf_model=ctfidf_model,  # Add our custom model here
        calculate_probabilities=False,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)

    # Calculate and print percentage of documents in each topic cluster
    from collections import Counter
    topic_counts = Counter(topics)
    total_docs = len(topics)
    topic_info = topic_model.get_topic_info()

    topic_percentages = [
        (topic, count, (count / total_docs) * 100)
        for topic, count in topic_counts.items() if topic != -1
    ]
    topic_percentages.sort(key=lambda x: x[2], reverse=True)

    print("\nPercentage of documents in each topic cluster (sorted):")
    for topic, count, percent in topic_percentages:
        words = topic_model.get_topic(topic)
        top_words = [w for w, _ in words[:10]]
        best_label = top_words[0] if top_words else ""
        print(f"Topic {topic} ({best_label}) ({percent:.2f}%): {', '.join(top_words)}")


if __name__ == "__main__":
    main()