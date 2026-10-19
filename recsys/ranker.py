"""Stage 2: re-rank the retriever's top-N with richer (user, item) features.

The retriever can only use what fits in a dot product of two independently
computed vectors. The ranker sees cross features the towers cannot express:
co-visitation with the user's recent items, distance from where they usually go,
category affinity, and the retriever's own score/rank.

Two rankers on identical features:
  * LightGBM LambdaRank: optimizes NDCG directly via pairwise gradients
  * DCN-v2 (deep & cross) with a listwise softmax loss over each user's candidates
"""
from __future__ import annotations

import warnings
from collections import Counter

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = ["tt_score", "tt_rank", "log_pop", "item_stars", "item_log_reviews",
            "item_price", "user_log_reviews", "user_avg_stars", "user_log_fans",
            "user_tenure", "hist_len", "cat_affinity", "covisit_sum", "covisit_max",
            "covisit_last", "geo_dist_centroid", "geo_dist_last", "city_match"]


def _km(a, b):
    """Equirectangular distance in km; plenty accurate within one metro."""
    lat = np.radians((a[..., 0] + b[..., 0]) / 2)
    dx = np.radians(b[..., 1] - a[..., 1]) * np.cos(lat)
    dy = np.radians(b[..., 0] - a[..., 0])
    return 6371 * np.sqrt(dx ** 2 + dy ** 2)


class FeatureBuilder:
    """Precomputes per-user and item-item statistics from the history available at
    prediction time. Built from TRAIN for the val stage and TRAIN+VAL for the test
    stage, so no feature ever peeks at the target."""

    def __init__(self, ds, hists: list[list[int]], covisit_window: int = 20):
        self.ds, self.hists, self.window = ds, hists, covisit_window
        rows = [u for u, h in enumerate(hists) for _ in h]
        cols = [i for h in hists for i in h]
        X = sp.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)),
                          shape=(ds.n_users, ds.n_items))
        X.data[:] = 1
        C = (X.T @ X).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        self.covisit = C
        self.pop = np.asarray(X.sum(0)).ravel()
        self.cat_dist = []
        self.centroid = np.zeros((ds.n_users, 2), np.float32)
        self.home_city = np.full(ds.n_users, -1)
        for u, h in enumerate(hists):
            cnt = Counter(c for i in h for c in ds.item_cats[i])
            tot = sum(cnt.values()) or 1
            self.cat_dist.append({c: v / tot for c, v in cnt.items()})
            if h:
                self.centroid[u] = np.median(ds.item_geo[h], axis=0)
                self.home_city[u] = Counter(ds.item_city[h].tolist()).most_common(1)[0][0]

    def build(self, users: list[int], cands: list[list[int]], scores: list[list[float]]):
        ds, feats = self.ds, []
        for u, items, sc in zip(users, cands, scores):
            items = np.asarray(items)
            h = self.hists[u]
            recent = h[-self.window:]
            if recent:
                co = self.covisit[recent][:, items].toarray()
                co_sum, co_max = co.sum(0), co.max(0)
                co_last = self.covisit[h[-1]][:, items].toarray().ravel()
                d_last = _km(ds.item_geo[h[-1]][None], ds.item_geo[items])
            else:
                co_sum = co_max = co_last = np.zeros(len(items))
                d_last = np.zeros(len(items))
            cd = self.cat_dist[u]
            aff = np.array([sum(cd.get(c, 0) for c in ds.item_cats[i]) for i in items])
            n = len(items)
            feats.append(np.column_stack([
                sc, np.arange(n), np.log1p(self.pop[items]), ds.item_num[items],
                np.repeat(ds.user_num[u][None], n, 0), np.full(n, len(h)), aff,
                np.log1p(co_sum), np.log1p(co_max), np.log1p(co_last),
                _km(self.centroid[u][None], ds.item_geo[items]), d_last,
                (ds.item_city[items] == self.home_city[u]).astype(float)]))
        return np.vstack(feats).astype(np.float32)


def labels_and_groups(cands, targets):
    y = np.concatenate([(np.asarray(c) == t).astype(np.int32) for c, t in zip(cands, targets)])
    return y, np.array([len(c) for c in cands])


# ---------------------------------------------------------------------- LightGBM
def train_lgbm(X, y, groups, X_val, y_val, g_val, seed=0, **params):
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05,
                           num_leaves=63, min_child_samples=50, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           lambdarank_truncation_level=20, random_state=seed,
                           verbose=-1, **params)
    # eval_set is deprecated in newest LightGBM but still the portable spelling.
    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")
    model.fit(X, y, group=groups, eval_set=[(X_val, y_val)], eval_group=[g_val],
              eval_at=[10], callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


# ---------------------------------------------------------------------- DCN-v2
class DCN(nn.Module):
    """Deep & Cross Network v2: explicit feature crosses x0 * (W x_l + b) + x_l,
    in parallel with a plain MLP."""

    def __init__(self, d, n_cross=3, hidden=128):
        super().__init__()
        self.cross = nn.ModuleList(nn.Linear(d, d) for _ in range(n_cross))
        self.deep = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(d + hidden, 1)

    def forward(self, x):
        x0, xl = x, x
        for layer in self.cross:
            xl = x0 * layer(xl) + xl
        return self.out(torch.cat([xl, self.deep(x)], -1)).squeeze(-1)


class DCNRanker:
    def __init__(self, d, seed=0):
        torch.manual_seed(seed)
        self.net = DCN(d)
        self.mu = self.sd = None

    def _norm(self, X):
        return torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)

    def fit(self, X, y, groups, X_val, y_val, g_val, epochs=30, lr=1e-3, batch_groups=256,
            patience=4):
        """Listwise loss: softmax over each user's N candidates, CE on the true one.
        Only fixed-size groups that contain the target are used for training."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xg, pos = self._grouped(X, y, groups)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        best, best_state, bad = -1, None, 0
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xg))
            for s in range(0, len(perm), batch_groups):
                idx = perm[s:s + batch_groups]
                logits = self.net(Xg[idx])
                loss = F.cross_entropy(logits, pos[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            score = ndcg_at(self.predict(X_val), y_val, g_val, 10)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def _grouped(self, X, y, groups):
        n = int(np.bincount(groups).argmax())  # the common (full) group size
        starts = np.r_[0, np.cumsum(groups)[:-1]]
        Xs, ps = [], []
        for s, g in zip(starts, groups):
            yy = y[s:s + g]
            if g == n and yy.any():
                Xs.append(X[s:s + g])
                ps.append(int(yy.argmax()))
        return self._norm(np.stack(Xs)), torch.tensor(ps)

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        return self.net(self._norm(X)).numpy()


def ndcg_at(scores, y, groups, k=10):
    """Mean NDCG@k over groups that contain a positive."""
    out, s = [], 0
    for g in groups:
        sc, yy = scores[s:s + g], y[s:s + g]
        s += g
        if not yy.any():
            continue
        order = np.argsort(-sc, kind="stable")[:k]
        hits = np.where(yy[order] > 0)[0]
        out.append(1 / np.log2(hits[0] + 2) if len(hits) else 0.0)
    return float(np.mean(out)) if out else 0.0


def rerank(cands, scores, groups):
    out, s = [], 0
    for c, g in zip(cands, groups):
        order = np.argsort(-scores[s:s + g], kind="stable")
        out.append([c[j] for j in order])
        s += g
    return out
