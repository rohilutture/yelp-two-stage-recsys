"""Train and evaluate the full two-stage system.

    python -m recsys.pipeline --dataset artifacts/dataset.pkl --out artifacts/

Protocol (no leakage):
  retriever      trained on TRAIN interactions; best epoch picked on VAL recall@N
  ranker         trained on retriever candidates for VAL targets (features from TRAIN)
  final metrics  on TEST targets, with history and features from TRAIN+VAL
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from . import ab_sim
from . import ranker as rk
from .data import Dataset, histories
from .metrics import coverage, evaluate
from .retrieval import ItemIndex, bench_ann, popular_topk, popularity
from .two_tower import item_embeddings, train_two_tower, user_embeddings


def retrieve(model, feats, index, users, hists, k, hist_len):
    uv = user_embeddings(model, feats, users, [hists[u] for u in users], hist_len)
    return index.search(uv, k, exclude=[set(hists[u]) for u in users])


def stage_lists(model, feats, ds, split, hists, n_cands, hist_len):
    users = split.u.tolist()
    index = ItemIndex(item_embeddings(model, feats), "flat")
    cands, scores = retrieve(model, feats, index, users, hists, n_cands, hist_len)
    return users, split.i.tolist(), cands, scores


def run(ds: Dataset, out: Path, epochs=8, dim=64, n_cands=100, hist_len=50, seed=0,
        log=print):
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    train_h = histories(ds.train, ds.n_users)
    trainval_h = histories(ds.train, ds.n_users, ds.val)
    results, timings = {}, {}

    # ---- Stage 1 -------------------------------------------------------------
    val_users = ds.val.u.tolist()

    def val_recall(model, feats):
        _, tgt, cands, _ = stage_lists(model, feats, ds, ds.val, train_h, n_cands, hist_len)
        return evaluate(tgt, cands, ks=(n_cands,))[f"recall@{n_cands}"]

    t0 = time.perf_counter()
    model, feats = train_two_tower(ds, train_h, epochs=epochs, dim=dim, hist_len=hist_len,
                                   seed=seed, log=log, val_fn=val_recall)
    timings["two_tower_train_s"] = round(time.perf_counter() - t0, 1)

    # ---- Baseline: popularity --------------------------------------------------
    pop = popularity(ds.train.i.to_numpy(), ds.n_items)
    test_users, test_tgt = ds.test.u.tolist(), ds.test.i.tolist()
    pop_lists = popular_topk(pop, [set(trainval_h[u]) for u in test_users], n_cands)
    results["popularity"] = {**evaluate(test_tgt, pop_lists),
                             "coverage@10": coverage(pop_lists, ds.n_items)}

    # ---- Two-tower alone (retrieval order) ----------------------------------------
    _, _, val_cands, val_scores = stage_lists(model, feats, ds, ds.val, train_h, n_cands, hist_len)
    _, _, test_cands, test_scores = stage_lists(model, feats, ds, ds.test, trainval_h,
                                                n_cands, hist_len)
    results["two_tower"] = {**evaluate(test_tgt, test_cands),
                            "coverage@10": coverage(test_cands, ds.n_items)}

    # ---- Stage 2: features ---------------------------------------------------------
    t0 = time.perf_counter()
    fb_val = rk.FeatureBuilder(ds, train_h)
    X = fb_val.build(val_users, val_cands, val_scores)
    y, g = rk.labels_and_groups(val_cands, ds.val.i.tolist())
    fb_test = rk.FeatureBuilder(ds, trainval_h)
    X_test = fb_test.build(test_users, test_cands, test_scores)
    y_test, g_test = rk.labels_and_groups(test_cands, test_tgt)
    timings["features_s"] = round(time.perf_counter() - t0, 1)

    # Hold out 10% of VAL users (whole groups) for ranker early stopping.
    rng = np.random.default_rng(seed)
    is_es = rng.random(len(g)) < 0.1
    row_es = np.repeat(is_es, g)
    Xtr, ytr, gtr = X[~row_es], y[~row_es], g[~is_es]
    Xes, yes, ges = X[row_es], y[row_es], g[is_es]

    t0 = time.perf_counter()
    lgbm = rk.train_lgbm(Xtr, ytr, gtr, Xes, yes, ges, seed=seed)
    timings["lgbm_train_s"] = round(time.perf_counter() - t0, 1)
    lgbm_lists = rk.rerank(test_cands, lgbm.predict(X_test), g_test)
    results["two_tower+lgbm"] = {**evaluate(test_tgt, lgbm_lists),
                                 "coverage@10": coverage(lgbm_lists, ds.n_items)}

    t0 = time.perf_counter()
    dcn = rk.DCNRanker(X.shape[1], seed=seed).fit(Xtr, ytr, gtr, Xes, yes, ges)
    timings["dcn_train_s"] = round(time.perf_counter() - t0, 1)
    dcn_lists = rk.rerank(test_cands, dcn.predict(X_test), g_test)
    results["two_tower+dcn"] = {**evaluate(test_tgt, dcn_lists),
                                "coverage@10": coverage(dcn_lists, ds.n_items)}

    importance = dict(sorted(zip(rk.FEATURES, lgbm.booster_.feature_importance("gain").round(1)),
                             key=lambda kv: -kv[1]))
    uv = user_embeddings(model, feats, test_users[:2000],
                         [trainval_h[u] for u in test_users[:2000]], hist_len)
    ann = bench_ann(item_embeddings(model, feats), uv, k=min(n_cands, ds.n_items - 1))

    # Simulated online experiment: control = retrieval order, treatment = + LightGBM.
    ids = [ds.user_raw_ids[u] for u in test_users]
    ab = ab_sim.run_ab(ids, test_tgt, test_cands, lgbm_lists, salt="ranker-v1", seed=seed)
    ab["aa_false_positive_rate"] = ab_sim.aa_false_positive_rate(ids, test_tgt, test_cands,
                                                                n_runs=100)
    if ab["ctr_control"]:
        ab["users_per_arm_for_5pct_rel_lift"] = ab_sim.users_per_arm(ab["ctr_control"], 0.05)

    report = {"dataset": {"users": ds.n_users, "items": ds.n_items,
                          "train_interactions": len(ds.train), "test_users": len(test_users)},
              "n_candidates": n_cands, "results": results,
              "retrieval_recall_at_N": results["two_tower"][f"recall@{n_cands}"],
              "lgbm_feature_gain": importance, "ab_test": ab, "ann_hnsw_vs_exact": ann,
              "timings": timings}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    # Serving artifacts
    torch.save(model.state_dict(), out / "two_tower.pt")
    np.save(out / "item_vecs.npy", item_embeddings(model, feats))
    lgbm.booster_.save_model(str(out / "ranker_lgbm.txt"))
    with open(out / "serving.pkl", "wb") as f:
        pickle.dump({"dim": dim, "hist_len": hist_len, "n_cands": n_cands,
                     "popular": pop[:500].tolist()}, f)
    write_markdown(report, out / "REPORT.md")
    return report


def write_markdown(rep, path: Path):
    n = rep["n_candidates"]
    cols = ["recall@10", "ndcg@10", "mrr", f"recall@{n}", "coverage@10"]
    lines = ["# Results\n", f"Test users: {rep['dataset']['test_users']:,} · "
             f"items: {rep['dataset']['items']:,} · candidates per user: {n}\n",
             "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in rep["results"].items():
        lines.append(f"| {name} | " + " | ".join(str(r.get(c, "—")) for c in cols) + " |")
    lines += ["\n## Ranker feature importance (LightGBM gain)\n", "| Feature | Gain |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rep["lgbm_feature_gain"].items()]
    lines += ["\n## HNSW vs exact search\n", "| efSearch | Recall vs exact | QPS |", "|---|---|---|"]
    lines += [f"| {r['efSearch']} | {r['recall_vs_exact']} | {r['qps']} |"
              for r in rep["ann_hnsw_vs_exact"]]
    ab = rep["ab_test"]
    lines += ["\n## Simulated A/B: two-tower (control) vs two-tower + LightGBM (treatment)\n",
              f"- Users: {ab['n_control']:,} control / {ab['n_treatment']:,} treatment "
              f"(SRM p = {ab['srm_p']})",
              f"- CTR: {ab['ctr_control']} → {ab['ctr_treatment']} "
              f"(relative lift {ab['rel_lift']}, p = {ab['p_value']}, "
              f"95% CI of abs. lift {ab['ci95_abs_lift']})",
              f"- A/A false-positive rate over 100 salts: {ab['aa_false_positive_rate']} "
              "(should be close to 0.05)",
              f"- Users per arm to detect a 5% relative lift: "
              f"{ab.get('users_per_arm_for_5pct_rel_lift', '—'):,}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="artifacts/dataset.pkl")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n-cands", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rep = run(Dataset.load(Path(a.dataset)), Path(a.out), a.epochs, a.dim, a.n_cands, seed=a.seed)
    print(json.dumps(rep["results"], indent=2))


if __name__ == "__main__":
    main()
