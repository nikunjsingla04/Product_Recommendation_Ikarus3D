# ================= DISABLED FOR RENDER =================
# Flan-T5 is disabled due to Render free plan memory limit (512MB)
# Alternative: Using template-based fallback in /generate-description endpoint
GENAI_PIPELINE = None
# ========================================================
import os
import re
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# If langchain is not installed, we fall back to SentenceTransformer directly.
try:
    from langchain.embeddings import SentenceTransformerEmbeddings
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

from sentence_transformers import SentenceTransformer

LOG = logging.getLogger("uvicorn.error")

OUT_DIR = "embeddings_data"
MODEL_NAME = "all-MiniLM-L6-v2"

EMB_PATH = os.path.join(OUT_DIR, "embeddings_text.npy")
META_PATH = os.path.join(OUT_DIR, "metadata.csv")
TITLE_EMB_PATH = os.path.join(OUT_DIR, "embeddings_title.npy")

if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("Missing embeddings_data. Run preprocess_and_embed.py first and ensure embeddings_data/ exists.")

embs = np.load(EMB_PATH).astype("float32")
meta = pd.read_csv(META_PATH).fillna("")

if os.path.exists(TITLE_EMB_PATH):
    emb_title = np.load(TITLE_EMB_PATH).astype("float32")
    tnorms = np.linalg.norm(emb_title, axis=1, keepdims=True)
    tnorms[tnorms == 0] = 1.0
    emb_title = emb_title / tnorms
else:
    emb_title = None

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

embs = normalize_rows(embs)

def parse_price(value):
    if pd.isna(value) or value is None: return None
    if isinstance(value, (int, float)) and not np.isnan(value): return float(value)
    s = str(value)
    m = re.search(r'[-+]?\d[\d,]*\.?\d*', s)
    if not m: return None
    try:
        return float(m.group(0).replace(",", ""))
    except:
        return None

if "price_num" not in meta.columns:
    meta["price_num"] = meta["price"].apply(parse_price)

if LANGCHAIN_AVAILABLE:
    try:
        lc_embedder = SentenceTransformerEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        LOG.warning("LangChain embedder init failed, falling back: %s", e)
        LANGCHAIN_AVAILABLE = False
        lc_embedder = None
else:
    lc_embedder = None

model = SentenceTransformer(MODEL_NAME)

def embed_query_text(q: str) -> np.ndarray:
    q = q or ""
    if LANGCHAIN_AVAILABLE and lc_embedder is not None:
        try:
            vec = lc_embedder.embed_query(q)
            arr = np.asarray(vec, dtype=np.float32)
            arr = arr / (np.linalg.norm(arr) + 1e-12)
            return arr
        except Exception as e:
            LOG.warning("LangChain embed_query failed, fallback: %s", e)
    vec = model.encode([q], convert_to_numpy=True)[0]
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype("float32")

def expand_query_text(q: str) -> str:
    if not q:
        return q
    q_low = q.lower()
    expansions = []
    if "dimmer" in q_low or "dimmable" in q_low:
        expansions += ["dimmable lamp", "dimmable light", "lamp with dimmer", "lamp dimmer"]
    if "floor" in q_low and "lamp" in q_low:
        expansions += ["floor lamp", "standing lamp", "floor light"]
    elif "lamp" in q_low:
        expansions += ["lamp", "lighting", "led lamp", "led light", "table lamp", "floor lamp"]
    if "sofa" in q_low or "couch" in q_low:
        expansions += ["sofa", "couch", "loveseat", "sectional", "sofa couch"]
    if "chair" in q_low:
        expansions += ["chair", "armchair", "accent chair", "lounge chair"]
    if "ottoman" in q_low or "storage ottoman" in q_low:
        expansions += ["ottoman", "storage ottoman", "storage bench"]
    if "bookshelf" in q_low or "bookcase" in q_low:
        expansions += ["bookshelf", "bookcase", "shelf", "storage shelf"]
    tokens = [t for t in re.split(r"\W+", q_low) if t and len(t) > 2]
    synonym_map = {
        "lamp": ["light", "lighting", "led"],
        "dimmer": ["dimmable", "dimmer switch", "dimming"],
        "sofa": ["couch", "loveseat"],
        "chair": ["seat", "armchair"]
    }
    for t in tokens:
        if t in synonym_map:
            expansions += synonym_map[t]
    expansions = list(dict.fromkeys([e for e in expansions if e]))
    if expansions:
        return q + " " + " ".join(expansions)
    return q

app = FastAPI(title="MVP Recommender (FastAPI)")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class RecommendReq(BaseModel):
    query: str
    top_k: Optional[int] = 5
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

class ProductOut(BaseModel):
    uniq_id: str
    title: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    score: Optional[float] = None
    image: Optional[str] = None

@app.post("/recommend", response_model=List[ProductOut])
def recommend(req: RecommendReq):
    q = (req.query or "").strip()
    k = req.top_k or 5
    cat = req.category
    min_p = req.min_price
    max_p = req.max_price

    try:
        q_proc = expand_query_text(q)
    except Exception:
        q_proc = q

    q_emb = embed_query_text(q_proc)
    mask = pd.Series(True, index=meta.index)
    if cat:
        mask = mask & meta["categories"].astype(str).str.lower().str.contains(str(cat).lower())
    if min_p is not None:
        mask = mask & (meta["price_num"].fillna(np.nan) >= float(min_p))
    if max_p is not None:
        mask = mask & (meta["price_num"].fillna(np.nan) <= float(max_p))

    cand_idxs = mask[mask].index.to_numpy()
    if cand_idxs.size == 0:
        return []

    sims_text = np.dot(embs[cand_idxs], q_emb)
    if emb_title is not None:
        sims_title = np.dot(emb_title[cand_idxs], q_emb)
        title_weight = 0.6
        sims_all = title_weight * sims_title + (1.0 - title_weight) * sims_text
    else:
        sims_all = sims_text

    tokens = [t.lower() for t in re.split(r"\W+", q_proc) if t and len(t) > 2]
    token_boost_map = {"lamp":0.16, "light":0.14, "dimmer":0.18, "sofa":0.10, "chair":0.10, "ottoman":0.10}
    important_tokens = set(token_boost_map.keys())
    boosts = np.zeros_like(sims_all, dtype=float)
    for i_local, idx_global in enumerate(cand_idxs):
        row = meta.iloc[int(idx_global)]
        text_fields = " ".join([str(row.get("title","") or ""), str(row.get("categories","") or ""), str(row.get("description","") or "")]).lower()
        boost_val = 0.0
        for tok in set(tokens):
            if tok in text_fields:
                if tok in token_boost_map:
                    boost_val += token_boost_map[tok]
                else:
                    boost_val += 0.04
        boosts[i_local] = float(boost_val)
    sims_boosted = sims_all + boosts

    overlap_boosts = np.zeros_like(sims_all, dtype=float)
    for i_local, idx_global in enumerate(cand_idxs):
        row = meta.iloc[int(idx_global)]
        text_fields = " ".join([str(row.get("title","") or ""), str(row.get("categories","") or ""), str(row.get("description","") or "")]).lower()
        overlap_count = sum(1 for tok in set(tokens) if tok in text_fields)
        if overlap_count >= 1:
            overlap_boosts[i_local] = 0.12 * overlap_count
    sims_boosted = sims_boosted + overlap_boosts

    top_n = min(len(cand_idxs), max(50, k*10))
    order_all = np.argsort(-sims_boosted)[:top_n]
    selected_idxs = cand_idxs[order_all]
    selected_scores = sims_boosted[order_all]

    top_k_pos = np.argsort(-selected_scores)[:k]
    final_idxs = selected_idxs[top_k_pos]
    final_scores = selected_scores[top_k_pos]

    if len(final_scores) == 0 or float(final_scores[0]) < 0.15:
        mask_kw = pd.Series(False, index=meta.index)
        for t in tokens:
            if len(t) < 3: continue
            mask_kw = mask_kw | meta["title"].astype(str).str.lower().str.contains(t) | meta["categories"].astype(str).str.lower().str.contains(t)
        mask_kw = mask_kw & mask
        fallback_ids = mask_kw[mask_kw].index.tolist()
        combined = list(final_idxs)
        for fid in fallback_ids:
            if fid not in combined:
                combined.append(fid)
            if len(combined) >= k:
                break
        final_idxs = combined[:k]
        final_scores = []
        for idx in final_idxs:
            if idx in cand_idxs:
                i_local = int(np.where(cand_idxs == idx)[0][0])
                final_scores.append(float(sims_boosted[i_local]))
            else:
                final_scores.append(float(np.dot(embs[int(idx)], q_emb)))

    out = []
    for idx, sc in zip(final_idxs, final_scores):
        row = meta.iloc[int(idx)].to_dict()
        imgs_field = str(row.get("images","") or "")
        img = None
        if imgs_field:
            m = re.search(r'https?://[^\s,\]\']+', imgs_field)
            if m:
                img = m.group(0).rstrip("',\"]")
            else:
                for p in re.split(r'[,|;]', imgs_field):
                    p2 = p.strip().strip("[]'\" ")
                    if p2.lower().startswith("http"):
                        img = p2
                        break
                if img is None:
                    img = imgs_field.split(",")[0].strip("[]'\" ")
        out.append(ProductOut(
            uniq_id=str(row.get("uniq_id","")),
            title=row.get("title"),
            brand=row.get("brand"),
            price=parse_price(row.get("price")),
            description=row.get("description"),
            score=float(sc),
            image=img
        ))
    return out

class GenReq(BaseModel):
    uniq_id: str
    use_llm: Optional[bool] = False

@app.post("/generate-description")
def generate_description(req: GenReq):
    uid = req.uniq_id
    row_df = meta[meta["uniq_id"].astype(str) == str(uid)]
    if row_df.shape[0] == 0:
        raise HTTPException(status_code=404, detail="uniq_id not found")
    row = row_df.iloc[0].to_dict()

    title = str(row.get("title","")).strip()
    brand = str(row.get("brand","")).strip()
    material = str(row.get("material","")).strip()
    color = str(row.get("color","")).strip()
    cats = str(row.get("categories","")).strip()
    orig_desc = str(row.get("description","")).strip()
    
    # Note: LLM is disabled on Render free plan to avoid memory errors.
    # Using template-based fallback instead.

    if req.use_llm and GENAI_PIPELINE:
        try:
            prompt = f"Write a short, engaging product description in 1-2 sentences:\nTitle: {title}\nBrand: {brand}\nMaterial: {material}\nColor: {color}\nCategories: {cats}\nExisting: {orig_desc}\n"
            outp = GENAI_PIPELINE(prompt, max_length=120, do_sample=False)
            text = outp[0]["generated_text"] if isinstance(outp, list) else str(outp)
            return {"uniq_id": uid, "creative_description": text}
        except Exception as e:
            LOG.error("LLM generation failed: %s", e)



    parts = []
    if title: parts.append(title)
    if brand: parts.append(f"by {brand}")
    if material or color:
        mc = " ".join([c for c in [color, material] if c])
        parts.append(f"a {mc}")
    if cats:
        parts.append(f"({cats})")
    summary = orig_desc.split(".")[0][:240] if orig_desc else ""
    if summary: parts.append("— " + summary)
    templ = " ".join(parts).strip()
    if not templ:
        templ = f"Product {uid}. Great quality item."
    if not templ.endswith("."):
        templ = templ + "."
    return {"uniq_id": uid, "creative_description": templ}

@app.get("/analytics")
def analytics():
    cats = meta["categories"].fillna("").astype(str)
    cat_counts = {}
    for s in cats:
        if not s: continue
        for part in re.split(r'[>,;|]', s):
            p = part.strip().lower()
            if not p: continue
            cat_counts[p] = cat_counts.get(p, 0) + 1

    brand_counts = meta["brand"].fillna("").astype(str).str.strip().value_counts().to_dict()
    price_nums = meta["price_num"].dropna().astype(float)
    price_stats = {"min": float(price_nums.min()) if not price_nums.empty else None,
                   "max": float(price_nums.max()) if not price_nums.empty else None,
                   "mean": float(price_nums.mean()) if not price_nums.empty else None}

    color_counts = meta["color"].fillna("").astype(str).str.strip().value_counts().to_dict()
    material_counts = meta["material"].fillna("").astype(str).str.strip().value_counts().to_dict()

    return {
        "categories": cat_counts,
        "brands": brand_counts,
        "price": price_stats,
        "colors": color_counts,
        "materials": material_counts,
        "n_items": int(len(meta))
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
