import sqlite3
import json
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic.representation import MaximalMarginalRelevance
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


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
    """Load documents from SQLite database"""
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


class DomainAwareBERTopic:
    def __init__(self):
        # Initialize with better parameters
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            min_df=5,
            max_df=0.85
        )

        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine'
        )

        self.hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        self.representation_model = MaximalMarginalRelevance(
            diversity=0.4
        )

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            representation_model=self.representation_model,
            calculate_probabilities=True,
            verbose=True
        )

    def fit_transform(self, documents):
        """Process documents and extract topics"""
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        topics, probs = self.topic_model.fit_transform(documents, embeddings)

        # Post-processing
        self._merge_similar_topics()
        self._generate_labels()

        return topics, probs

    def _merge_similar_topics(self):
        """Merge topics based on cosine similarity of their embeddings"""
        # Get topic embeddings
        topic_embeddings = np.array([self.topic_model.topic_embeddings_[i]
                                     for i in range(len(self.topic_model.topic_embeddings_))
                                     if i != -1])

        # Calculate similarity matrix
        sim_matrix = cosine_similarity(topic_embeddings)

        # Find similar topics (threshold = 0.85)
        to_merge = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i][j] > 0.85:
                    to_merge.append((i, j))

        # Merge topics
        for topic1, topic2 in to_merge:
            self.topic_model.merge_topics([topic1, topic2])

    def _generate_labels(self):
        """Generate better labels using top terms"""
        topic_info = self.topic_model.get_topic_info()

        for idx, row in topic_info.iterrows():
            if row.Topic == -1:
                continue

            # Get top terms
            top_terms = [term[0] for term in self.topic_model.get_topic(row.Topic)[:5]]

            # Create meaningful label
            label = " | ".join(top_terms[:3])
            self.topic_model.set_topic_labels({row.Topic: label})


def main():
    # Load documents
    db_path = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\openalex_ai_works_sample_1k.db"
    docs = load_documents(db_path)
    print(f"Loaded {len(docs)} documents.")

    # Initialize and run topic modeling
    model = DomainAwareBERTopic()
    topics, probs = model.fit_transform(docs)

    # Print results
    topic_info = model.topic_model.get_topic_info()
    topic_counts = Counter(topics)
    total_docs = len(topics)

    print("\nResearch Domains Identified:")
    for _, row in topic_info.iterrows():
        if row.Topic == -1:
            continue

        percentage = (topic_counts[row.Topic] / total_docs) * 100
        print(f"\nDomain {row.Topic} ({percentage:.1f}%): {row.Name}")
        print("Key Terms:", ", ".join([word[0] for word in model.topic_model.get_topic(row.Topic)][:10]))


if __name__ == "__main__":
    main()