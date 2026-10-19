"""FastAPI service: GET /recommend/{user_id} -> retrieve (HNSW) -> rank (LightGBM).

    uvicorn recsys.serve:app --port 8080     (ARTIFACTS env var, default ./artifacts)

Online path per request:
  1. user vector from the two-tower model using the user's latest history
  2. HNSW top-N, excluding places already visited
  3. features + LightGBM scores, sort, return top-k with names
Unknown users (cold start) get the popularity list; the response says so.
"""
from __future__ import annotations

import os
import pickle
import time
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from .data import Dataset, histories
from .ranker import FeatureBuilder
from .retrieval import ItemIndex
from .two_tower import Features, TwoTower, user_embeddings


class Recommender:
    def __init__(self, art: Path):
        self.ds = Dataset.load(art / "dataset.pkl")
        cfg = pickle.loads((art / "serving.pkl").read_bytes())
        self.hist_len, self.n_cands, self.popular = cfg["hist_len"], cfg["n_cands"], cfg["popular"]
        ds = self.ds
        self.feats = Features(ds)
        self.model = TwoTower(ds.n_users, ds.n_items, ds.n_cats, ds.user_num.shape[1],
                              ds.item_num.shape[1], dim=cfg["dim"], n_cities=self.feats.n_cities)
        self.model.load_state_dict(torch.load(art / "two_tower.pt", weights_only=True))
        self.model.eval()
        self.index = ItemIndex(np.load(art / "item_vecs.npy"), "hnsw")
        self.ranker = lgb.Booster(model_file=str(art / "ranker_lgbm.txt"))
        # Serving uses everything known (train+val+test) as history.
        self.hists = histories(ds.train, ds.n_users, pd.concat([ds.val, ds.test]))
        self.fb = FeatureBuilder(ds, self.hists)
        self.u_map = {u: k for k, u in enumerate(ds.user_raw_ids)}

    def recommend(self, raw_user: str, k: int = 10):
        u = self.u_map.get(raw_user)
        if u is None:
            return "popularity_fallback", [int(i) for i in self.popular[:k]]
        uv = user_embeddings(self.model, self.feats, [u], [self.hists[u]], self.hist_len)
        cands, scores = self.index.search(uv, self.n_cands, exclude=[set(self.hists[u])])
        X = self.fb.build([u], cands, scores)
        order = np.argsort(-self.ranker.predict(X), kind="stable")[:k]
        return "two_stage", [cands[0][j] for j in order]


app = FastAPI(title="Yelp two-stage recommender")
_latency_ms: deque = deque(maxlen=5000)
_rec: Recommender | None = None


def get_rec() -> Recommender:
    global _rec
    if _rec is None:
        _rec = Recommender(Path(os.environ.get("ARTIFACTS", "artifacts")))
    return _rec


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10):
    if not 1 <= k <= 100:
        raise HTTPException(422, "k must be in [1, 100]")
    t0 = time.perf_counter()
    rec = get_rec()
    source, items = rec.recommend(user_id, k)
    _latency_ms.append(1000 * (time.perf_counter() - t0))
    return {"user_id": user_id, "source": source,
            "items": [{"business_id": rec.ds.item_raw_ids[i], "name": rec.ds.item_names[i]}
                      for i in items]}


@app.get("/metrics")
def metrics():
    lat = np.array(_latency_ms) if _latency_ms else np.array([0.0])
    return {"requests": len(_latency_ms),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "latency_ms_p99": round(float(np.percentile(lat, 99)), 2)}
