"""Ranking metrics for the one-held-out-item-per-user setting."""
from __future__ import annotations

import numpy as np


def rank_of(target: int, ranked: list[int] | np.ndarray) -> int | None:
    """0-based position of target in ranked, or None if absent."""
    ranked = list(ranked)
    return ranked.index(target) if target in ranked else None


def evaluate(targets: list[int], ranked_lists: list, ks=(10, 50, 100)) -> dict:
    """With one relevant item per user:
    Recall@K == HitRate@K; NDCG@K = 1/log2(rank+2) if hit; MRR = 1/(rank+1)."""
    ranks = [rank_of(t, r) for t, r in zip(targets, ranked_lists)]
    n = len(ranks)
    out = {"n_users": n}
    for k in ks:
        hits = [r is not None and r < k for r in ranks]
        out[f"recall@{k}"] = round(sum(hits) / n, 4)
        out[f"ndcg@{k}"] = round(float(sum(1 / np.log2(r + 2) for r in ranks
                                     if r is not None and r < k)) / n, 4)
    out["mrr"] = round(sum(1 / (r + 1) for r in ranks if r is not None) / n, 4)
    return out


def coverage(ranked_lists, n_items: int, k: int = 10) -> float:
    """Share of the catalog that appears in anyone's top-k — guards against a model
    that wins on accuracy by recommending the same 50 popular places to everyone."""
    return round(len({i for r in ranked_lists for i in list(r)[:k]}) / n_items, 4)
