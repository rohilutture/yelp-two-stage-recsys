"""Turn raw Yelp JSON (or a synthetic stand-in) into an encoded, time-split dataset.

Task framing: next-business prediction. A "positive" is a review with >= 4 stars.
For each user, ordered by time: the last positive is TEST, the second-to-last is
VALIDATION, everything earlier is TRAIN. Splitting by time per user (not
randomly) is what prevents future interactions from leaking into training.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_TOP_CATS = 200


@dataclass
class Dataset:
    n_users: int
    n_items: int
    n_cats: int
    train: pd.DataFrame          # columns: u, i, ts   (encoded ints)
    val: pd.DataFrame            # one row per user
    test: pd.DataFrame           # one row per user
    item_cats: list[list[int]]   # category ids per item
    item_num: np.ndarray         # [n_items, d] standardized numeric item features
    item_geo: np.ndarray         # [n_items, 2] lat, lon (raw)
    item_city: np.ndarray        # [n_items] int city id
    user_num: np.ndarray         # [n_users, d] standardized numeric user features
    item_names: list[str] = field(default_factory=list)
    user_raw_ids: list[str] = field(default_factory=list)
    item_raw_ids: list[str] = field(default_factory=list)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Dataset":
        with open(path, "rb") as f:
            return pickle.load(f)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    mu, sd = x.mean(0), x.std(0)
    return (x - mu) / np.where(sd > 0, sd, 1)


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users/items with < k interactions until stable."""
    while True:
        uc, ic = df["user"].map(df["user"].value_counts()), df["item"].map(df["item"].value_counts())
        keep = (uc >= k) & (ic >= k)
        if keep.all():
            return df
        df = df[keep]


def temporal_split(df: pd.DataFrame):
    """df: encoded positives (u, i, ts). Leave-last-two-out per user by time.
    Users with < 3 positives cannot supply train+val+test and are dropped."""
    df = df.sort_values(["u", "ts", "i"], kind="stable")
    df["rank_from_end"] = df.groupby("u").cumcount(ascending=False)
    n = df.groupby("u")["i"].transform("size")
    df = df[n >= 3]
    test = df[df.rank_from_end == 0][["u", "i", "ts"]]
    val = df[df.rank_from_end == 1][["u", "i", "ts"]]
    train = df[df.rank_from_end >= 2][["u", "i", "ts"]]
    # Items never seen in training can't be retrieved by an ID-based model; evaluate
    # only on items with training signal (item cold-start is a separate problem).
    seen = set(train.i.unique())
    val, test = val[val.i.isin(seen)], test[test.i.isin(seen)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _encode(pos: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame) -> Dataset:
    """pos: user, item, ts. items: item, name, categories(list), stars, review_count,
    price, lat, lon, city. users: user, review_count, average_stars, fans, tenure."""
    u_ids = sorted(pos.user.unique())
    i_ids = sorted(pos.item.unique())
    u_map = {u: k for k, u in enumerate(u_ids)}
    i_map = {i: k for k, i in enumerate(i_ids)}
    enc = pd.DataFrame({"u": pos.user.map(u_map).astype(np.int64),
                        "i": pos.item.map(i_map).astype(np.int64), "ts": pos.ts.values})
    train, val, test = temporal_split(enc)

    items = items.set_index("item").loc[i_ids]
    cat_counts = Counter(c for cs in items.categories for c in cs)
    top = [c for c, _ in cat_counts.most_common(N_TOP_CATS)]
    c_map = {c: k for k, c in enumerate(top)}
    item_cats = [[c_map[c] for c in cs if c in c_map] for cs in items.categories]
    item_num = _standardize(np.c_[items.stars, np.log1p(items.review_count),
                                  items.price.fillna(items.price.median())])
    city_map = {c: k for k, c in enumerate(sorted(items.city.unique()))}

    users = users.set_index("user").reindex(u_ids)
    user_num = _standardize(np.c_[np.log1p(users.review_count.fillna(0)),
                                  users.average_stars.fillna(users.average_stars.mean()),
                                  np.log1p(users.fans.fillna(0)), users.tenure.fillna(0)])
    return Dataset(len(u_ids), len(i_ids), len(top), train, val, test, item_cats, item_num,
                   items[["lat", "lon"]].to_numpy(np.float32),
                   items.city.map(city_map).to_numpy(np.int64), user_num,
                   items.name.tolist(), list(u_ids), list(i_ids))


# --------------------------------------------------------------------------- Yelp
def _read_jsonl(path: Path, cols: list[str], chunksize=500_000, filt=None):
    parts = []
    for chunk in pd.read_json(path, lines=True, chunksize=chunksize, dtype=False):
        chunk = chunk[cols]
        parts.append(filt(chunk) if filt else chunk)
    return pd.concat(parts, ignore_index=True)


def _price(attrs):
    if not isinstance(attrs, dict):  # null attributes arrive as NaN
        return np.nan
    try:
        return float(attrs.get("RestaurantsPriceRange2"))
    except (TypeError, ValueError):
        return np.nan


def load_yelp(raw_dir: Path, state: str | None = "PA", k: int = 5,
              min_stars: int = 4) -> Dataset:
    """Stream the official Yelp Open Dataset JSON. Restricting to one state keeps
    it laptop-sized (PA ~ Philadelphia metro is the largest) and makes geography
    a meaningful ranking feature."""
    raw = Path(raw_dir)
    biz = _read_jsonl(raw / "yelp_academic_dataset_business.json",
                      ["business_id", "name", "categories", "stars", "review_count",
                       "attributes", "latitude", "longitude", "city", "state"])
    if state:
        biz = biz[biz.state == state]
    keep_biz = set(biz.business_id)
    rev = _read_jsonl(raw / "yelp_academic_dataset_review.json",
                      ["user_id", "business_id", "stars", "date"],
                      filt=lambda c: c[c.business_id.isin(keep_biz) & (c.stars >= min_stars)])
    pos = pd.DataFrame({"user": rev.user_id, "item": rev.business_id,
                        "ts": pd.to_datetime(rev.date).astype("int64") // 10**9})
    pos = pos.drop_duplicates(["user", "item"], keep="first")  # first positive visit
    pos = k_core(pos, k)

    items = pd.DataFrame({
        "item": biz.business_id, "name": biz.name,
        "categories": biz.categories.fillna("").map(
            lambda s: [c.strip() for c in s.split(",") if c.strip()]),
        "stars": biz.stars, "review_count": biz.review_count,
        "price": biz.attributes.map(_price), "lat": biz.latitude, "lon": biz.longitude,
        "city": biz.city.str.strip().str.title()})
    keep_users = set(pos.user)
    usr = _read_jsonl(raw / "yelp_academic_dataset_user.json",
                      ["user_id", "review_count", "average_stars", "fans", "yelping_since"],
                      filt=lambda c: c[c.user_id.isin(keep_users)])
    now = pd.Timestamp("2022-01-01")
    users = pd.DataFrame({"user": usr.user_id, "review_count": usr.review_count,
                          "average_stars": usr.average_stars, "fans": usr.fans,
                          "tenure": (now - pd.to_datetime(usr.yelping_since)).dt.days / 365})
    return _encode(pos, items, users)


# ---------------------------------------------------------------------- synthetic
def synthetic(n_users=2000, n_items=600, n_cats=30, dim=8, per_user=(8, 30),
              n_cities=5, seed=0) -> Dataset:
    """A Yelp-shaped dataset with real structure, for tests and a CPU demo.

    Users and items have latent taste vectors; items belong to categories and
    cities; users mostly stay in a home city and drift slowly in taste. Choices are
    sampled from softmax(taste . item + popularity + locality), so models that use
    history, geography, and co-visitation SHOULD beat popularity — which makes the
    tests meaningful rather than just "code runs".
    """
    rng = np.random.default_rng(seed)
    cat_vec = rng.normal(size=(n_cats, dim))
    item_cat = [list(rng.choice(n_cats, size=rng.integers(1, 4), replace=False))
                for _ in range(n_items)]
    item_vec = np.stack([cat_vec[c].mean(0) for c in item_cat]) + 0.5 * rng.normal(size=(n_items, dim))
    pop = rng.normal(0, 1.0, n_items)
    city_xy = rng.uniform(0, 10, size=(n_cities, 2))
    item_city = rng.integers(0, n_cities, n_items)
    item_geo = city_xy[item_city] + rng.normal(0, 0.3, (n_items, 2))
    rows, users = [], []
    for u in range(n_users):
        taste = rng.normal(size=dim)
        home = rng.integers(0, n_cities)
        n = rng.integers(*per_user)
        seen = set()
        t = int(rng.integers(1.4e9, 1.5e9))
        for _ in range(n):
            local = np.where(item_city == home, 1.5, 0.0)
            logits = item_vec @ taste / np.sqrt(dim) * 1.5 + 0.6 * pop + local
            logits[list(seen)] = -np.inf
            p = np.exp(logits - logits.max())
            i = rng.choice(n_items, p=p / p.sum())
            seen.add(i)
            t += int(rng.integers(86400, 86400 * 60))
            rows.append((f"u{u}", f"b{i}", t))
            taste = 0.9 * taste + 0.1 * item_vec[i]  # taste drifts toward what you liked
        users.append((f"u{u}", n * 3, rng.uniform(3, 5), rng.poisson(2), rng.uniform(0, 10)))
    pos = pd.DataFrame(rows, columns=["user", "item", "ts"])
    items = pd.DataFrame({
        "item": [f"b{i}" for i in range(n_items)], "name": [f"Business {i}" for i in range(n_items)],
        "categories": [[f"cat{c}" for c in cs] for cs in item_cat],
        "stars": np.clip(3.5 + 0.5 * pop + rng.normal(0, 0.3, n_items), 1, 5).round(1),
        "review_count": np.exp(3 + pop).astype(int), "price": rng.integers(1, 5, n_items).astype(float),
        "lat": item_geo[:, 0], "lon": item_geo[:, 1], "city": [f"City{c}" for c in item_city]})
    users = pd.DataFrame(users, columns=["user", "review_count", "average_stars", "fans", "tenure"])
    return _encode(pos, items, users)


def histories(train: pd.DataFrame, n_users: int, extra: pd.DataFrame | None = None):
    """Per-user chronological item lists (train, optionally + extra split)."""
    df = train if extra is None else pd.concat([train, extra])
    df = df.sort_values(["u", "ts", "i"], kind="stable")
    out = [[] for _ in range(n_users)]
    for u, i in zip(df.u.to_numpy(), df.i.to_numpy()):
        out[u].append(int(i))
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw", help="dir with the Yelp JSON files")
    p.add_argument("--state", default="PA")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="artifacts/dataset.pkl")
    a = p.parse_args(argv)
    ds = synthetic() if a.synthetic else load_yelp(Path(a.raw), a.state or None)
    ds.save(Path(a.out))
    print(json.dumps({"users": ds.n_users, "items": ds.n_items, "categories": ds.n_cats,
                      "train": len(ds.train), "val": len(ds.val), "test": len(ds.test)}))


if __name__ == "__main__":
    main()
