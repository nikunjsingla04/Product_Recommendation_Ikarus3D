import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const API = "http://127.0.0.1:8000";

export default function Analytics() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const res = await fetch(API + "/analytics");
      const j = await res.json();
      setStats(j);
    } catch (err) {
      console.error(err);
      setStats(null);
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <div>Loading analytics...</div>;
  if (!stats) return <div>Error loading analytics.</div>;

  // prepare top categories and brands
  const categoriesArr = Object.entries(stats.categories || {}).map(([k, v]) => ({ name: k, value: v }));
  const brandsArr = Object.entries(stats.brands || {}).map(([k, v]) => ({ name: k || "(empty)", value: v }));
  categoriesArr.sort((a,b)=>b.value-a.value);
  brandsArr.sort((a,b)=>b.value-a.value);

  const topCats = categoriesArr.slice(0, 12);
  const topBrands = brandsArr.slice(0, 12);

  return (
    <div>
      <h2>Dataset Analytics</h2>

      <div className="analytics-grid">
        <div className="chart-card">
          <h3>Top Categories</h3>
          <div style={{ width: "100%", height: 340 }}>
            <ResponsiveContainer>
              <BarChart data={topCats} margin={{ top: 10, right: 20, left: 0, bottom: 50 }}>
                <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <h3>Top Brands</h3>
          <div style={{ width: "100%", height: 340 }}>
            <ResponsiveContainer>
              <BarChart data={topBrands} margin={{ top: 10, right: 20, left: 0, bottom: 50 }}>
                <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card small">
          <h3>Price Summary</h3>
          <div><b>Items:</b> {stats.n_items}</div>
          <div><b>Min:</b> {stats.price?.min ?? "-"}</div>
          <div><b>Mean:</b> {stats.price?.mean ?? "-"}</div>
          <div><b>Max:</b> {stats.price?.max ?? "-"}</div>
        </div>

      </div>
    </div>
  );
}
