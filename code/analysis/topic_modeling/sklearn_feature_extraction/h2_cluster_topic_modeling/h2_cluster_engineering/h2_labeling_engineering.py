"""
h2_labeling_engineering.py
────────────────────────────────────────────────────────────────────────────
Adds a second‑hierarchy label (column `h2_cluster`) to every record
in the ENGINEERING SQLite corpus. Codes:

    0 : Cybersecurity & Privacy
    1 : IoT & Edge Computing
    2 : Anomaly & Fault Detection
    3 : Energy Technologies
    4 : Autonomous Mobility
    5 : Robotics & Mechatronics
"""
import os, json, re, sqlite3, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ── File locations ────────────────────────────────────────────────────────────
DB_PATH   = r"C:\Users\Admin\Documents\Master-Thesis\code\open_alex_api\data\working_dataset\h1_cluster_subsets\engineering_dataset.db"
JSON_PATH = r"C:\Users\Admin\Documents\Master-Thesis\code\analysis\topic_modeling\sklearn_feature_extraction\h2_cluster_engineering\resources\cluster_abstracts_summary.json"

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

# ── 6. Map fine‑grained cluster IDs → H2 codes --------------------------------
H2_MAP = {
    # 0 – Cybersecurity & Privacy
    **{str(i): "Cybersecurity & Privacy"
       for i in (6, 14, 18, 22, 30, 31, 35, 38, 39, 40, 46, 49, 48)},

    # 1 – IoT & Edge Computing
    **{str(i): "IoT & Edge Computing"
       for i in (3, 12, 16, 20, 28, 36, 4, 34, 13, 33, 45)},

    # 2 – Anomaly & Fault Detection
    **{str(i): "Anomaly & Fault Detection"
       for i in (2, 8, 23, 24)},

    # 3 – Energy Technologies
    **{str(i): "Energy Technologies"
       for i in (1, 10, 17, 19, 25, 37)},

    # 4 – Autonomous Mobility
    **{str(i): "Autonomous Mobility"
       for i in (0, 11, 26, 27, 32, 42, 43, 47)},

    # 5 – Robotics & Mechatronics
    **{str(i): "Robotics & Mechatronics"
       for i in (21, 41, 44, 5, 7, 9, 15, 29)}
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
