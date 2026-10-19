"""ANN index over item embeddings + baseline retrievers."""
from __future__ import annotations

import faiss
import numpy as np


class ItemIndex:
    """Inner-product search (vectors are unit-norm, so IP == cosine).

    'flat' is exact and used for evaluation so retrieval quality isn't confounded
    with ANN error; 'hnsw' is what you'd serve. bench_ann() measures the gap.
    """

    def __init__(self, item_vecs: np.ndarray, kind: str = "flat", hnsw_m: int = 32,
                 ef_search: int = 128):
        d = item_vecs.shape[1]
        if kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)
        self.index.add(np.ascontiguousarray(item_vecs, dtype=np.float32))
        self.kind = kind

    def search(self, user_vecs: np.ndarray, k: int, exclude: list[set[int]] | None = None):
        """Top-k per user, skipping already-visited items (over-fetch then filter)."""
        extra = max((len(e) for e in exclude), default=0) if exclude else 0
        scores, ids = self.index.search(np.ascontiguousarray(user_vecs, dtype=np.float32),
                                        k + extra)
        out_ids, out_scores = [], []
        for r in range(len(ids)):
            ex = exclude[r] if exclude else set()
            keep = [(i, s) for i, s in zip(ids[r], scores[r]) if i >= 0 and i not in ex][:k]
            out_ids.append([int(i) for i, _ in keep])
            out_scores.append([float(s) for _, s in keep])
        return out_ids, out_scores


def popularity(train_items: np.ndarray, n_items: int):
    return np.argsort(-np.bincount(train_items, minlength=n_items), kind="stable")


def popular_topk(order: np.ndarray, exclude: list[set[int]], k: int):
    out = []
    for ex in exclude:
        lst = []
        for i in order:
            if i not in ex:
                lst.append(int(i))
                if len(lst) == k:
                    break
        out.append(lst)
    return out


def bench_ann(item_vecs, user_vecs, k=100, ef_values=(16, 32, 64, 128, 256)):
    """Recall of HNSW vs exact top-k, and query throughput, per efSearch value."""
    import time
    exact = ItemIndex(item_vecs, "flat")
    truth, _ = exact.search(user_vecs, k)
    hnsw = ItemIndex(item_vecs, "hnsw")
    rows = []
    for ef in ef_values:
        hnsw.index.hnsw.efSearch = ef
        t0 = time.perf_counter()
        got, _ = hnsw.search(user_vecs, k)
        dt = time.perf_counter() - t0
        rec = np.mean([len(set(a) & set(b)) / k for a, b in zip(got, truth)])
        rows.append({"efSearch": ef, "recall_vs_exact": round(float(rec), 4),
                     "qps": round(len(user_vecs) / dt, 1)})
    return rows
