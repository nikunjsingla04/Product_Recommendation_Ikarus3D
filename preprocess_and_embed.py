# preprocess_and_embed.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# ------ CONFIG ------
DATA_CSV = "intern_data_ikarus.csv"   # <-- set to your CSV filename in the project folder
OUT_DIR = "embeddings_data"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128
# --------------------

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)
print("Loaded rows:", len(df))

# ensure uniq_id present
if "uniq_id" not in df.columns:
    raise RuntimeError("uniq_id column not found in dataset")

# Impute missing description with title + categories
df["description"] = df["description"].fillna("")
missing_desc_mask = df["description"].str.strip() == ""
df.loc[missing_desc_mask, "description"] = (
    df.loc[missing_desc_mask, "title"].fillna("") + " " +
    df.loc[missing_desc_mask, "categories"].fillna("")
)

# Build combined text for embedding
text_fields = ["title","brand","description","categories",
               "manufacturer","material","color","country_of_origin","package_dimensions"]
def build_text(row):
    """
    Build a rich, human-readable text used for embeddings.
    Prioritize: title, brand, explicit short description, then fallback to material/color/categories.
    Adds labeled metadata lines (Categories:, Material:, Color:, Manufacturer:, Origin:).
    """
    parts = []
    title = str(row.get("title","") or "").strip()
    brand = str(row.get("brand","") or "").strip()
    desc = str(row.get("description","") or "").strip()
    material = str(row.get("material","") or "").strip()
    color = str(row.get("color","") or "").strip()
    cats = str(row.get("categories","") or "").replace(">",",").strip()
    manu = str(row.get("manufacturer","") or "").strip()
    origin = str(row.get("country_of_origin","") or "").strip()
    pkg = str(row.get("package_dimensions","") or "").strip()

    if title:
        parts.append(title)
    if brand:
        parts.append(f"Brand: {brand}")
    # Prefer original description if it's non-trivial
    if desc and len(desc) > 30:
        parts.append(desc)
    else:
        # synthesize a short sentence using material/color/categories
        syn_chunks = []
        if material:
            syn_chunks.append(material)
        if color:
            syn_chunks.append(color)
        if cats:
            syn_chunks.append(cats.split(",")[0].strip())
        if syn_chunks:
            parts.append(" ".join(syn_chunks))

    # Add labeled metadata for context (helps embedding capture structured attributes)
    for label, val in [("Categories", cats), ("Material", material), ("Color", color),
                       ("Manufacturer", manu), ("Origin", origin), ("Package", pkg)]:
        if val and str(val).strip():
            parts.append(f"{label}: {val}")

    # limit length to avoid extremely long texts
    text = " || ".join(parts)
    if len(text) > 1200:
        text = text[:1200]
    return text

df["text_for_emb"] = df.apply(build_text, axis=1)

# Save metadata (keeps the requested columns + text_for_emb)
meta_cols = ["uniq_id","title","brand","description","price","categories","images",
             "manufacturer","package_dimensions","country_of_origin","material","color","text_for_emb"]
for c in meta_cols:
    if c not in df.columns:
        df[c] = ""
df[meta_cols].to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)

# Create embeddings
model = SentenceTransformer(MODEL_NAME)
emb_dim = model.get_sentence_embedding_dimension()
n = len(df)
embs = np.zeros((n, emb_dim), dtype=np.float32)

for start in tqdm(range(0, n, BATCH_SIZE)):
    end = min(n, start + BATCH_SIZE)
    texts = df["text_for_emb"].iloc[start:end].tolist()
    batch_emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs[start:end] = batch_emb

print("Computing title embeddings...")
titles = df["title"].fillna("").astype(str).tolist()
title_embs = model.encode(titles, convert_to_numpy=True, show_progress_bar=True)
# normalize
t_norms = np.linalg.norm(title_embs, axis=1, keepdims=True)
t_norms[t_norms == 0] = 1.0
title_embs = title_embs / t_norms
np.save(os.path.join(OUT_DIR, "embeddings_title.npy"), title_embs)
print("Saved title embeddings to", os.path.join(OUT_DIR, "embeddings_title.npy"))

np.save(os.path.join(OUT_DIR, "embeddings_text.npy"), embs)
print("Saved embeddings to", os.path.join(OUT_DIR, "embeddings_text.npy"))
print("Saved metadata to", os.path.join(OUT_DIR, "metadata.csv"))
