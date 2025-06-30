import sqlite3
import json
import re
from collections import defaultdict
from gensim import corpora
from gensim.models import LdaModel, Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import nltk


def main():
    nltk.download('stopwords')

    db_path = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\openalex_ai_works_sample_10k.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT abstract_inverted_index 
    FROM works 
    WHERE abstract_inverted_index IS NOT NULL AND language = 'en'
    """
    rows = cursor.execute(query).fetchall()
    print(f"Fetched {len(rows)} rows.")

    def decompress_abstract(json_str):
        inverted_index = json.loads(json_str)
        position_word = []
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word.append((pos, word))
        position_word.sort()
        words = [w for _, w in position_word]
        return ' '.join(words)

    documents = [decompress_abstract(row[0]) for row in rows]

    tokenizer = TreebankWordTokenizer()
    stop_words = set(stopwords.words('english'))
    processed_docs = []

    for doc in documents:
        tokens = tokenizer.tokenize(doc.lower())
        tokens = [re.sub(r'\W+', '', t) for t in tokens]
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
        processed_docs.append(tokens)

    # === Create bigrams ===
    bigram = Phrases(processed_docs, min_count=5, threshold=10)
    bigram_mod = Phraser(bigram)

    bigram_docs = [bigram_mod[doc] for doc in processed_docs]

    dictionary = corpora.Dictionary(bigram_docs)
    corpus = [dictionary.doc2bow(doc) for doc in bigram_docs]
    print(f"Dictionary size: {len(dictionary)}")

    num_topics = 18  # fixed
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)

    # === Compute overall word probabilities ===
    word_freq = defaultdict(float)
    total_tokens = sum(sum(count for _, count in doc) for doc in corpus)
    for bow in corpus:
        for word_id, count in bow:
            word_freq[word_id] += count
    for word_id in word_freq:
        word_freq[word_id] /= total_tokens

    # === Extract top distinctive term for each topic ===
    used_titles = set()
    topic_titles = []

    for idx, topic in lda.show_topics(num_topics=num_topics, num_words=20, formatted=False):
        candidates = []
        for word, prob in topic:
            word_id = dictionary.token2id[word]
            p_w = word_freq[word_id]
            relevance = prob / p_w if p_w > 0 else 0
            candidates.append((relevance, word))
        candidates.sort(reverse=True)  # more distinctive first

        for _, candidate in candidates:
            if candidate not in used_titles:
                used_titles.add(candidate)
                topic_titles.append((idx, candidate))
                break
        else:
            topic_titles.append((idx, f"Topic{idx}"))

    print("\n=== Final Topics with Bigrams ===")
    for idx, title in topic_titles:
        top_words = ", ".join([w for w, _ in lda.show_topic(idx, topn=5)])
        print(f"Topic {idx} ({title}): {top_words}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
