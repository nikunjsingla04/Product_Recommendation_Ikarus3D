import requests, json, math, random, sys, os
import pandas as pd
from collections import defaultdict

URL = "http://127.0.0.1:8000/recommend"
META = "embeddings_data/metadata.csv"

def map_to_coarse(catstr):
    s = str(catstr).lower()
    if any(x in s for x in ["sofa","couch","loveseat","sectional"]):
        return "sofa"
    if any(x in s for x in ["chair","seat","lounge","armchair"]):
        return "chair"
    if any(x in s for x in ["lamp","lighting","light"]):
        return "lamp"
    if any(x in s for x in ["table","desk","coffee table","dining"]):
        return "table"
    if any(x in s for x in ["ottoman","stool","storage","shelf","bookshelf","cabinet"]):
        return "storage"
    if any(x in s for x in ["bed","mattress","daybed"]):
        return "bed"
    if any(x in s for x in ["rug","mat"]):
        return "rug"
    return "other"

def run_query(q, top_k=5):
    try:
        r = requests.post(URL, json={"query": q, "top_k": top_k}, timeout=20)
    except Exception as e:
        return {"error": str(e)}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text}"}
    return {"items": r.json()}

def precision_at_k(results, true_label, k):
    if not results:
        return 0.0
    topk = results[:k]
    for it in topk:
        # if categories available, use them first
        cats = (it.get("description","") + " " + it.get("title","") + " " + (it.get("brand") or "") + " " + (it.get("image") or "")).lower()
        # coarse map of returned item
        ret_label = map_to_coarse(it.get("description","") or it.get("title",""))
        # check if returned coarse label equals true label OR any keyword present in title/description
        if ret_label == true_label or (true_label in cats):
            return 1.0
    return 0.0

def main():
    df = pd.read_csv(META)
    # build a pool of candidate queries using title and category
    queries = []
    for _, r in df.iterrows():
        title = str(r.get("title","") or "").strip()
        cats = str(r.get("categories","") or "").strip()
        if not title:
            continue
        # generate 2 queries per product: title alone, and title + first category token
        if cats:
            first_cat = cats.split(",")[0]
            queries.append((title, map_to_coarse(cats)))
            queries.append((f"{title} {first_cat}", map_to_coarse(cats)))
        else:
            queries.append((title, map_to_coarse(cats)))
    # deduplicate and sample up to N queries for speed
    seen = set()
    uniq_qs = []
    for q,l in queries:
        key = q.lower().strip()
        if key in seen: continue
        seen.add(key)
        uniq_qs.append((q,l))
    random.seed(42)
    SAMPLE_N = min(200, len(uniq_qs))
    sample = random.sample(uniq_qs, SAMPLE_N)
    print(f"Running evaluation on {SAMPLE_N} sampled queries...")

    stats = {"p1_sum":0.0, "p5_sum":0.0, "cases":0}
    per_label = defaultdict(lambda: {"p1":0, "p5":0, "n":0})
    for q, true_label in sample:
        res = run_query(q, top_k=5)
        if "error" in res:
            print("ERROR for query:", q, res["error"])
            continue
        items = res["items"]
        p1 = precision_at_k(items, true_label, 1)
        p5 = precision_at_k(items, true_label, 5)
        stats["p1_sum"] += p1
        stats["p5_sum"] += p5
        stats["cases"] += 1
        per_label[true_label]["p1"] += p1
        per_label[true_label]["p5"] += p5
        per_label[true_label]["n"] += 1

    if stats["cases"] == 0:
        print("No cases evaluated (server error?).")
        return
    p1 = stats["p1_sum"] / stats["cases"]
    p5 = stats["p5_sum"] / stats["cases"]
    print(f"Overall precision@1: {p1:.3f}, precision@5: {p5:.3f} (N={stats['cases']})")
    print("Per-label breakdown (label, n, p@1, p@5):")
    for lab,vals in per_label.items():
        print(lab, vals["n"], f"{vals['p1']/vals['n']:.3f}", f"{vals['p5']/vals['n']:.3f}" )

if __name__ == "__main__":
    main()
