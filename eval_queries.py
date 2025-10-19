import requests
import json

URL = "http://127.0.0.1:8000/recommend"

cases = [
    ("grey two-seater sofa", ["sofa","couch","loveseat","sectional"]),
    ("comfortable two seater sofa", ["sofa","couch","loveseat","sectional"]),
    ("ergonomic mesh office chair", ["chair","seat","armchair","lounge"]),
    ("LED floor lamp with dimmer", ["lamp","lighting","light"]),
    ("ottoman storage", ["ottoman","stool","storage","ottomans"]),
    ("wooden bookshelf", ["bookshelf","shelf","storage","bookcase"]),
    ("outdoor patio chair", ["outdoor","patio","garden","chair"]),
    ("mid century accent chair", ["chair","accent","mid-century","lounge"])
]

def run_case(query, keywords, top_k=5):
    payload = {"query": query, "top_k": top_k}
    r = requests.post(URL, json=payload, timeout=20)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text}"}
    items = r.json()
    hit = 0
    for it in items:
        cats = (it.get("description","") + " " + it.get("title","") + " " + (it.get("brand","") or "") + " " + str(it.get("image","") or "") + " " + (it.get("score") and str(it.get("score")) or "")).lower()
        # check categories field if available
        meta_cats = (it.get("description","") or "").lower()
        # check keywords in categories/title/description
        if any(k in (it.get("title","") + " " + (it.get("description","") or "") + " " + (it.get("brand","") or "")).lower() for k in keywords):
            hit = 1
            break
    return {"num_results": len(items), "hit": hit, "items": items}

if __name__ == "__main__":
    totals = {"cases": 0, "hits": 0}
    for q, kw in cases:
        print("="*80)
        print("Query:", q)
        res = run_case(q, kw, top_k=5)
        if "error" in res:
            print("ERROR:", res["error"])
            continue
        for i, it in enumerate(res["items"]):
            title = it.get("title","")[:120].replace("\n"," ")
            score = it.get("score", 0)
            print(f" {i+1}. score={score:.3f} | {title}")
        print("Proxied hit (keyword match in top-5?):", res["hit"])
        totals["cases"] += 1
        totals["hits"] += res["hit"]
    print("="*80)
    print(f"SUMMARY: {totals['hits']} / {totals['cases']} cases had a keyword-match in top-5 (proxy accuracy: {totals['hits']/totals['cases']:.2f})")
