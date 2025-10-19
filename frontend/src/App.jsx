import React from "react";
import { Routes, Route, Link, Navigate } from "react-router-dom";
import Recommend from "./pages/Recommend";
import Analytics from "./pages/Analytics";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="container">
          <h1 className="logo">MVP Recommender</h1>
          <nav>
            <Link to="/recommend" className="navlink">Recommend</Link>
            <Link to="/analytics" className="navlink">Analytics</Link>
          </nav>
        </div>
      </header>

      <main className="container">
        <Routes>
          <Route path="/" element={<Navigate to="/recommend" replace />} />
          <Route path="/recommend" element={<Recommend />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </main>

      <footer className="footer">
        <div className="container">FastAPI + React demo — backend: http://127.0.0.1:8000</div>
      </footer>
    </div>
  );
}
