# Product Recommendation & Analytics Web App

## Overview

This project is an ML-driven web application for recommending furniture products and generating creative product descriptions. It integrates multiple AI domains (ML, NLP, CV, GenAI) with full-stack development and analytics visualization.

The app has two main routes:

1. **Recommendation Page:** Conversational product recommendation with generated descriptions and images.
2. **Analytics Page:** Visual analytics on dataset items (top categories, brands, etc.).

---

## Project Structure

```
main.py                  # FastAPI entrypoint
clean_prices.py          # Helper script to clean prices
preprocess_and_embed.py  # Preprocessing and embeddings
eval_precision.py        # Evaluate recommendation precision
eval_queries.py          # Test queries and keyword hits

frontend/
│   src/                     # React app source files
│
intern_data_ikarus      # Dataset file

```

---

## Setup Instructions

### Backend

1. Create and activate virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:

```bash
python main.py
```

Server runs at: `http://127.0.0.1:8000`

---

### Frontend

1. Navigate to the frontend folder:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
npm install react-router-dom recharts
```

3. Run the React development server:

```bash
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## Usage

* Open the frontend URL in browser.
* Use the **Recommendation Page** to input queries and get product suggestions.
* Click **Generate Description** for creative AI-generated descriptions.
* Switch to **Analytics Page** to view visual stats (top categories, brands, etc.).

---

## Note
* Ensure backend is running before using frontend.

---

## Evaluation Scripts

* `eval_precision.py` - Computes overall precision@1 and precision@5 of recommendations.
* `eval_queries.py` - Runs sample queries and checks if expected keywords appear in top results.

---
