# yelp-two-stage-recsys

A production-style **two-stage recommender** on the Yelp Open Dataset: a two-tower retriever narrows tens of thousands of businesses to 100 candidates, and a learned ranker orders them. It's evaluated offline with a strict time-based protocol, then "shipped" behind a FastAPI endpoint and read out with a simulated A/B test.

```
user history ──► user tower ─┐
                             ├─ dot product ─► FAISS (HNSW) top-100 ─► features ─► LightGBM LambdaRank / DCN-v2 ─► top-10
business info ─► item tower ─┘                                          (co-visits, distance, category affinity, ...)
```

| Stage | What | Why |
|---|---|---|
| Data | Yelp reviews ≥ 4★ as positives, one state (default PA), 5-core, **leave-last-two-out by time** | Next-visit prediction without future leakage |
| Retrieval | Two-tower (PyTorch): history pooling + ID + category + city + numeric features; in-batch softmax with **logQ correction** and duplicate masking | Scales to any catalog; the user tower is inductive (new history, no retraining) |
| Index | FAISS exact for evaluation, **HNSW** for serving, plus a recall-vs-QPS sweep | Separates model quality from ANN error |
| Ranking | Same 18 features into **LightGBM LambdaRank** and **DCN-v2** (listwise softmax) | Cross features a dot product can't express |
| Metrics | Recall@10/100, NDCG@10, MRR, catalog coverage; popularity baseline | Accuracy and diversity together |
| Experiment | Hash bucketing, position-biased click simulation, z-test, bootstrap CI, **SRM check**, **A/A calibration**, power analysis | The workflow teams use to decide whether to ship |
| Serving | FastAPI `/recommend/{user_id}`, cold-start fallback, p50/p95/p99 `/metrics`, Docker | |

## Results on Yelp

Pennsylvania subset, 39,622 test users, 17,388 businesses, 100 retrieved candidates per user. Run on CPU (AMD Ryzen 7 5800H, 16 GB RAM), about 13 minutes for training plus evaluation (8 epochs). Full report in `artifacts/REPORT.md`.

| Model | Recall@10 | NDCG@10 | MRR | Recall@100 | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0234 | 0.0135 | 0.0127 | 0.0903 | 0.002 |
| Two-tower | 0.0473 | 0.0250 | 0.0231 | 0.2049 | 0.2765 |
| + LightGBM | 0.0527 | 0.0278 | 0.0252 | (same) | 0.3613 |
| + DCN-v2 | 0.0614 | 0.0327 | 0.0289 | (same) | 0.2875 |

- HNSW reaches 0.992 recall against exact search at efSearch=128 (about 13.6K QPS on CPU).
- Simulated A/B (two-tower vs. + LightGBM): CTR 0.0247 to 0.0294 (+19% relative, p = 0.004), SRM p = 0.79, A/A false-positive rate 0.02. The clicks are simulated with a position-bias model, not real traffic.

<details><summary>Sanity run on the built-in synthetic dataset (CPU, ~30 s). Not a real-data claim.</summary>

| Model | Recall@10 | NDCG@10 | Recall@100 |
|---|---|---|---|
| Popularity | 0.057 | 0.027 | 0.364 |
| Two-tower | 0.104 | 0.054 | 0.456 |
| + LightGBM | 0.112 | 0.056 | 0.456 |
| + DCN-v2 | 0.136 | 0.070 | 0.456 |

The synthetic generator has real structure (latent taste, home city, taste drift), so the ordering above is a meaningful smoke test rather than a coincidence.
</details>

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (or: source .venv/bin/activate)
pip install -r requirements.txt

# 1. Get the data: https://www.yelp.com/dataset (accept the terms), extract the JSON files to data/yelp/
python -m recsys.data --raw data/yelp --state PA --out artifacts/dataset.pkl
#    (no data yet? python -m recsys.data --synthetic --out artifacts/dataset.pkl)

# 2. Train + evaluate everything, write artifacts/REPORT.md
python -m recsys.pipeline --dataset artifacts/dataset.pkl --out artifacts --epochs 8

# 3. Serve
uvicorn recsys.serve:app --port 8080
curl localhost:8080/recommend/<yelp_user_id>?k=10
#    or: docker build -t yelp-recsys . && docker run -p 8080:8080 -v %cd%/artifacts:/app/artifacts yelp-recsys
```

Notes:
- The review file is about 5 GB. `recsys.data` streams it in chunks, so RAM stays reasonable. Expect several minutes.
- Training runs on CPU. A GPU isn't needed at PA scale, though more epochs help.

## Tests
```bash
pytest -q
```
Covers temporal-split leakage, k-core, metric math, deterministic and balanced bucketing, A/A false-positive calibration, power calculation, the full pipeline on synthetic data (including "the retriever beats popularity" and "re-ranking can't change recall@N"), and the API (never recommends an already-visited place, cold-start fallback, input validation).

See **[DESIGN.md](DESIGN.md)** for the reasoning behind every choice.
