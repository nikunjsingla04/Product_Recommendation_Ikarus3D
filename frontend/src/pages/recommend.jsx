import React, { useState } from "react";

const API = "http://127.0.0.1:8000";

const placeholder = "https://via.placeholder.com/220x160?text=No+Image";

export default function Recommend() {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [minp, setMinp] = useState("");
  const [maxp, setMaxp] = useState("");
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState([]);
  const [status, setStatus] = useState("");

  async function search(top_k = 10) {
    if (!q.trim()) {
      alert("Enter a query");
      return;
    }
    setLoading(true);
    setStatus("Searching...");
    try {
      const payload = { query: q.trim(), top_k };
      if (cat.trim()) payload.category = cat.trim();
      if (minp !== "") payload.min_price = Number(minp);
      if (maxp !== "") payload.max_price = Number(maxp);
      const res = await fetch(API + "/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || res.status);
      }
      const j = await res.json();
      setItems(j || []);
      setStatus(`Found ${j?.length || 0} results`);
    } catch (err) {
      console.error(err);
      setStatus("Error: " + (err.message || err));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  async function genDescription(uniq_id, setter) {
    setter("Generating...");
    try {
      const res = await fetch(API + "/generate-description", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uniq_id, use_llm: false })
      });
      const j = await res.json();
      setter(j.creative_description || j.creative || JSON.stringify(j));
    } catch (err) {
      setter("Error generating: " + (err.message || err));
    }
  }

  return (
    <div>
      <h2>Search & Recommendations</h2>

      <div className="controls">
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Describe what you want (e.g. 'comfortable grey two-seater sofa')" />
        <input value={cat} onChange={e => setCat(e.target.value)} placeholder="Category (optional)" />
        <input value={minp} onChange={e => setMinp(e.target.value)} placeholder="min price" type="number" />
        <input value={maxp} onChange={e => setMaxp(e.target.value)} placeholder="max price" type="number" />
        <button onClick={() => search(10)} disabled={loading}>Search</button>
      </div>

      <div className="status">{status}</div>

      <div className="results">
        {items.length === 0 && !loading && <div className="noresults">No results</div>}
        {items.map((it) => (
          <ResultCard key={it.uniq_id} item={it} onGenerate={genDescription} />
        ))}
      </div>
    </div>
  );
}

function ResultCard({ item, onGenerate }) {
  const [genText, setGenText] = useState("");
  const [genLoading, setGenLoading] = useState(false);
  const image = item.image || placeholder;

  return (
    <div className="card">
      <img
        className="thumb"
        src={image}
        alt={item.title || "product"}
        onError={(e) => { e.target.onerror = null; e.target.src = placeholder; }}
      />
      <div className="meta">
        <div className="title">{item.title || "(no title)"}</div>
        <div className="brand"><b>Brand:</b> {item.brand || "-"}</div>
        <div className="price"><b>Price:</b> {item.price || "-"}</div>
        <div className="desc">{item.description || ""}</div>
        <div className="score">score: {item.score ? Number(item.score).toFixed(3) : "-"}</div>

        <div style={{ marginTop: 8 }}>
          <button
            disabled={genLoading}
            onClick={async () => {
              setGenLoading(true);
              await onGenerate(item.uniq_id, (text) => setGenText(text));
              setGenLoading(false);
            }}
          >
            {genLoading ? "Generating..." : "Generate description"}
          </button>
        </div>

        {genText && <div className="gen">{genText}</div>}
      </div>
    </div>
  );
}
