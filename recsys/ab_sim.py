"""Simulated online A/B test, to practice the experimentation workflow end to end.

Offline NDCG is not what a product team ships on: they run an experiment. With
no live traffic, this module simulates one honestly:

  * Bucketing: users go to control/treatment by a salted hash of their ID
    (deterministic, sticky, and independent of anything about the user).
  * Exposure: each user sees their arm's top-10.
  * Outcome: a click on the user's real held-out next business, with
    position bias: P(examine rank r) = 1/log2(r+2). Ranks past 10 aren't shown.
  * Readout: CTR per arm, absolute/relative lift, two-proportion z-test,
    bootstrap 95% CI, and a sample-ratio-mismatch (SRM) check.
  * A/A runs: the same model in both arms, many salts; the false-positive rate
    should be ~alpha. If it isn't, the analysis is broken, not the model.
  * Power: users per arm needed to detect a given lift.

Limitation, stated up front: the simulated user only "wants" one item (their
real next visit), so absolute CTRs are low and only RELATIVE comparisons mean
anything. That is the right level of claim for a portfolio project.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import stats


def arm_of(user_id: str, salt: str, treatment_share: float = 0.5) -> int:
    h = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:15], 16)
    return int((h / 16 ** 15) < treatment_share)


def examine_prob(rank: int) -> float:
    return 1 / math.log2(rank + 2)


def simulate_clicks(targets, lists, rng, k=10):
    out = np.zeros(len(targets), dtype=np.int8)
    for j, (t, lst) in enumerate(zip(targets, lists)):
        top = list(lst)[:k]
        if t in top:
            out[j] = rng.random() < examine_prob(top.index(t))
    return out


def srm_pvalue(n_c, n_t, share=0.5):
    total = n_c + n_t
    return float(stats.chisquare([n_c, n_t], [total * (1 - share), total * share]).pvalue)


def two_prop_z(c_clicks, c_n, t_clicks, t_n):
    p_c, p_t = c_clicks / c_n, t_clicks / t_n
    p = (c_clicks + t_clicks) / (c_n + t_n)
    se = math.sqrt(p * (1 - p) * (1 / c_n + 1 / t_n)) or 1e-12
    z = (p_t - p_c) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def bootstrap_ci(c, t, rng, n_boot=2000, alpha=0.05):
    diffs = [rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean() for _ in range(n_boot)]
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def run_ab(user_ids, targets, control_lists, treatment_lists, salt="exp-001", seed=0,
           alpha=0.05):
    rng = np.random.default_rng(seed)
    arms = np.array([arm_of(str(u), salt) for u in user_ids])
    idx_c, idx_t = np.where(arms == 0)[0], np.where(arms == 1)[0]
    c = simulate_clicks([targets[i] for i in idx_c], [control_lists[i] for i in idx_c], rng)
    t = simulate_clicks([targets[i] for i in idx_t], [treatment_lists[i] for i in idx_t], rng)
    z, p = two_prop_z(c.sum(), len(c), t.sum(), len(t))
    lo, hi = bootstrap_ci(c, t, rng)
    ctr_c, ctr_t = c.mean(), t.mean()
    return {"n_control": len(c), "n_treatment": len(t),
            "srm_p": round(srm_pvalue(len(c), len(t)), 4),
            "ctr_control": round(float(ctr_c), 5), "ctr_treatment": round(float(ctr_t), 5),
            "abs_lift": round(float(ctr_t - ctr_c), 5),
            "rel_lift": round(float((ctr_t - ctr_c) / ctr_c), 4) if ctr_c else None,
            "z": round(float(z), 3), "p_value": round(float(p), 5),
            "ci95_abs_lift": [round(lo, 5), round(hi, 5)],
            "significant": bool(p < alpha)}


def aa_false_positive_rate(user_ids, targets, lists, n_runs=200, alpha=0.05):
    sig = sum(run_ab(user_ids, targets, lists, lists, salt=f"aa-{r}", seed=r,
                     alpha=alpha)["significant"] for r in range(n_runs))
    return sig / n_runs


def users_per_arm(base_rate, rel_mde, alpha=0.05, power=0.8):
    """Classic two-proportion sample size for a relative minimum detectable effect."""
    p1, p2 = base_rate, base_rate * (1 + rel_mde)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
         / (p2 - p1)) ** 2
    return math.ceil(n)
