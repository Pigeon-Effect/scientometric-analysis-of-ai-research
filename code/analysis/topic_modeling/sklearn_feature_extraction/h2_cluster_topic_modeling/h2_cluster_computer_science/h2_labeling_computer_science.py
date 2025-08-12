"""
h2_cluster_compsci.py
────────────────────────────────────────────────────────────────────────────
Adds a second-hierarchy label (column `h2_cluster`) to every English record
in the COMPUTER-SCIENCE SQLite corpus.

Macro-clusters (codes 0–7)

    0 : Computer Vision & Image Processing
    1 : Natural Language Processing
    2 : Speech Recognition & Audio Processing
    3 : Neuromorphic Hardware Accelerators
    4 : Machine Learning Foundations
    5 : Biomarkers for Health & Performance
    6 : Ethical & Creative AI
    7 : Recommendation Systems
"""

import os, json, re, sqlite3, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ── File locations ────────────────────────────────────────────────────────────
DB_PATH   = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\computer_science_dataset.db"
JSON_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_computer_science\resources\cluster_abstracts_summary.json"

# ── 1. Load curated reference abstracts ---------------------------------------
with open(JSON_PATH, encoding="utf-8") as f:
    curated = json.load(f)

ref_texts, ref_labels = [], []
for cid, entry in curated.items():
    lbl = f"cluster_{cid}" if cid.isdigit() else cid
    for section in ("most_central_abstracts", "random_abstracts"):
        for item in entry[section]:
            title = item.get("title", "") or ""
            abst  = item.get("cleaned_abstract", "") or ""
            title = title.strip() if title else ""
            abst = abst.strip() if abst else ""
            ref_texts.append(f"{title}. {abst}")
            ref_labels.append(lbl)

# ── 2. Embed reference set -----------------------------------------------------
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = SentenceTransformer("allenai-specter", device=device)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

def embed_sliding(texts, win=100, stride=50):
    vecs = []
    for txt in tqdm(texts, desc="Embedding reference docs"):
        toks   = tokenizer.tokenize(txt)
        chunks = [toks[i:i+win] for i in range(0, len(toks), stride)]
        chunk_txts = [tokenizer.convert_tokens_to_string(c) for c in chunks if c]
        emb = model.encode(chunk_txts or [""], device=device)
        vecs.append(emb.mean(axis=0))
    return np.vstack(vecs)

ref_emb = embed_sliding(ref_texts)

# ── 3. Load full social‑science dataset ---------------------------------------
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("""
        SELECT * FROM works_labeled
        WHERE cleaned_abstract IS NOT NULL
          AND language = 'en';
       """, conn)

# ── 4. Preprocess & embed corpus ---------------------------------------------
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if len(w) > 2)

df["processed_text"] = (df["title"].fillna("") + ". " +
                        df["cleaned_abstract"].fillna("")).apply(clean)

abs_emb = embed_sliding(df["processed_text"].tolist())

# ── 5. Assign nearest reference label -----------------------------------------
sim         = cosine_similarity(abs_emb, ref_emb)
nearest_idx = np.argmax(sim, axis=1)
df["ref_id"] = [ref_labels[i] for i in nearest_idx]

# ── 6. Map fine-grained cluster IDs → H2 codes --------------------------------
H2_MAP = {
    # 0 – Computer Vision & Image Processing
    **{str(i): 0 for i in (2, 4, 5, 9, 15, 19, 20, 22, 23, 30, 35, 36, 45, 48)},

    # 1 – Natural Language Processing
    **{str(i): 1 for i in (3, 24, 27, 34, 41, 49)},

    # 2 – Speech Recognition & Audio Processing
    **{str(i): 2 for i in (7, 28, 33, 46)},

    # 3 – Neuromorphic Hardware Accelerators
    **{str(i): 3 for i in (1, 38)},

    # 4 – Machine Learning Foundations
    **{str(i): 4 for i in (0, 6, 8, 11, 12, 29, 31, 39, 40, 42)},

    # 5 – Biomarkers for Health & Performance
    **{str(i): 5 for i in (10, 13, 14, 16, 18, 26, 44, 47)},

    # 6 – Ethical & Creative AI
    **{str(i): 6 for i in (17, 25, 43)},

    # 7 – Recommendation Systems
    **{str(i): 7 for i in (21, 32, 37)},
}


# Sanity‑check: every original cluster ID appears exactly once
if len(H2_MAP) != 50:
    missing  = set(str(i) for i in range(50)) - set(H2_MAP)
    dupes    = [k for k in H2_MAP if list(H2_MAP).count(k) > 1]
    raise ValueError(f"Mapping incomplete or duplicated. Missing: {missing}, Duplicates: {dupes}")

df["h2_cluster"] = [H2_MAP.get(lbl.split("_")[-1], -1) for lbl in df["ref_id"]]

# ── 7. Write back to database --------------------------------------------------
# Only add h2_cluster column, not ref_id
df_to_write = df.drop(columns=['ref_id', 'processed_text'])
df_to_write.to_sql("works_labeled", conn, if_exists="replace", index=False)
conn.close()

print("✓  Added 'h2_cluster' column (codes 0‑5) to social‑science dataset.")
