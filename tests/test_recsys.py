import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import ab_sim
from recsys.data import synthetic, temporal_split, k_core
from recsys.metrics import coverage, evaluate
from recsys.pipeline import run


# ---------------------------------------------------------------- data
def test_temporal_split_is_leave_last_two_out():
    df = pd.DataFrame({"u": [0] * 5 + [1] * 2 + [2] * 5,
                       "i": [10, 11, 12, 13, 14, 10, 11, 10, 13, 15, 16, 99],
                       "ts": [5, 1, 3, 4, 2, 1, 2, 1, 2, 3, 4, 5]})
    tr, va, te = temporal_split(df)
    assert te.set_index("u").loc[0, "i"] == 10          # latest ts=5
    assert va.set_index("u").loc[0, "i"] == 13          # ts=4
    assert set(tr[tr.u == 0].i) == {11, 12, 14}
    assert 1 not in set(tr.u)                            # <3 positives: dropped
    assert 2 not in set(te.u)                            # item 99 never in train: not evaluable
    # Every train interaction for a user is strictly before their val and test.
    m = tr.merge(va, on="u", suffixes=("", "_v"))
    assert (m.ts < m.ts_v).all()


def test_k_core():
    df = pd.DataFrame({"user": ["a", "a", "b", "b", "c"], "item": ["x", "y", "x", "y", "z"]})
    out = k_core(df, 2)
    assert set(out.user) == {"a", "b"} and set(out.item) == {"x", "y"}


def test_synthetic_shapes():
    ds = synthetic(n_users=200, n_items=80, seed=1)
    assert len(ds.val) <= ds.n_users and ds.item_num.shape == (ds.n_items, 3)
    assert set(ds.test.i) <= set(ds.train.i)  # eval items have training signal


# ---------------------------------------------------------------- metrics
def test_metrics_single_relevant():
    r = evaluate([5, 7, 9], [[5, 1, 2], [1, 7, 2], [1, 2, 3]], ks=(1, 3))
    assert r["recall@1"] == round(1 / 3, 4)
    assert r["recall@3"] == round(2 / 3, 4)
    assert r["ndcg@3"] == round((1 + 1 / np.log2(3)) / 3, 4)
    assert r["mrr"] == round((1 + 0.5) / 3, 4)
    assert coverage([[1, 2], [2, 3]], 10, k=2) == 0.3


# ---------------------------------------------------------------- A/B
def test_bucketing_is_deterministic_and_balanced():
    arms = [ab_sim.arm_of(f"user{i}", "s") for i in range(20000)]
    assert arms == [ab_sim.arm_of(f"user{i}", "s") for i in range(20000)]
    assert abs(np.mean(arms) - 0.5) < 0.02
    assert arms != [ab_sim.arm_of(f"user{i}", "other") for i in range(20000)]


def test_ab_detects_real_lift_and_aa_is_calibrated():
    rng = np.random.default_rng(0)
    n = 6000
    ids, tgt = [f"u{i}" for i in range(n)], list(range(n))
    bad = [[t if rng.random() < 0.2 else -1] + list(range(-10, -1)) for t in tgt]
    good = [[t if rng.random() < 0.4 else -1] + list(range(-10, -1)) for t in tgt]
    res = ab_sim.run_ab(ids, tgt, bad, good, seed=1)
    assert res["significant"] and res["rel_lift"] > 0.5 and res["srm_p"] > 0.01
    fpr = ab_sim.aa_false_positive_rate(ids[:2000], tgt[:2000], bad[:2000], n_runs=100)
    assert fpr <= 0.12  # ~alpha=0.05; loose bound for 100 runs


def test_power_calc_monotonic():
    assert ab_sim.users_per_arm(0.05, 0.10) < ab_sim.users_per_arm(0.05, 0.05)
    assert ab_sim.users_per_arm(0.10, 0.05) < ab_sim.users_per_arm(0.05, 0.05)


# ---------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    ds = synthetic(n_users=900, n_items=250, seed=3)
    ds.save(out / "dataset.pkl")
    rep = run(ds, out, epochs=4, dim=32, n_cands=50, log=lambda *_: None)
    return out, rep


def test_pipeline_beats_popularity(trained):
    _, rep = trained
    r = rep["results"]
    assert r["two_tower"]["recall@10"] > r["popularity"]["recall@10"]
    best_ranker = max(r["two_tower+lgbm"]["ndcg@10"], r["two_tower+dcn"]["ndcg@10"])
    assert best_ranker > r["popularity"]["ndcg@10"]
    # Re-ranking cannot change recall@N: it only reorders the same N candidates.
    assert r["two_tower+lgbm"]["recall@50"] == r["two_tower"]["recall@50"]
    assert (trained[0] / "REPORT.md").exists()


def test_api(trained, monkeypatch):
    from fastapi.testclient import TestClient
    out, _ = trained
    monkeypatch.setenv("ARTIFACTS", str(out))
    import recsys.serve as serve
    serve._rec = None
    c = TestClient(serve.app)
    r = c.get("/recommend/u5", params={"k": 5}).json()
    assert r["source"] == "two_stage" and len(r["items"]) == 5
    seen = set(serve.get_rec().hists[serve.get_rec().u_map["u5"]])
    ids = {serve.get_rec().ds.item_raw_ids.index(x["business_id"]) for x in r["items"]}
    assert not ids & seen  # never recommend a place the user already visited
    assert c.get("/recommend/nobody").json()["source"] == "popularity_fallback"
    assert c.get("/recommend/u5", params={"k": 0}).status_code == 422
    assert c.get("/metrics").json()["requests"] >= 2


def test_yelp_loader_on_fake_json(tmp_path):
    """Exercise the real-data path with tiny files in the official Yelp schema."""
    import json
    from recsys.data import load_yelp
    rng = np.random.default_rng(0)
    biz, rev, usr = [], [], []
    for b in range(12):
        biz.append({"business_id": f"B{b}", "name": f"Place {b}",
                    "categories": "Restaurants, Pizza" if b % 2 else "Coffee & Tea, Cafes",
                    "stars": 4.0, "review_count": 10 + b,
                    "attributes": {"RestaurantsPriceRange2": "2"} if b % 3 else None,
                    "latitude": 39.9 + b / 100, "longitude": -75.1, "city": " philadelphia",
                    "state": "PA" if b < 10 else "NJ"})
    for u in range(8):
        usr.append({"user_id": f"U{u}", "review_count": 5, "average_stars": 4.1, "fans": 1,
                    "yelping_since": "2015-03-01 10:00:00"})
        for b in rng.choice(10, 7, replace=False):
            rev.append({"user_id": f"U{u}", "business_id": f"B{b}", "stars": 5,
                        "date": f"2019-0{rng.integers(1, 9)}-1{rng.integers(0, 9)} 12:00:00"})
        rev.append({"user_id": f"U{u}", "business_id": "B0", "stars": 1, "date": "2019-01-01 00:00:00"})
    for name, rows in [("business", biz), ("review", rev), ("user", usr)]:
        with open(tmp_path / f"yelp_academic_dataset_{name}.json", "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
    ds = load_yelp(tmp_path, state="PA", k=5)
    assert set(ds.item_raw_ids) <= {f"B{b}" for b in range(10)}   # NJ businesses filtered out
    assert 0 < ds.n_users <= 8                                     # 5-core may drop some
    assert all(n.startswith("Place") for n in ds.item_names)
    assert len(ds.test) > 0 and ds.item_num.shape[1] == 3
