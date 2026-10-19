# Design notes

## Why two stages
Scoring every business for every request with a rich model costs O(catalog) per request. Retrieval brings that down to about 100 candidates using a cheap dot product plus approximate nearest-neighbor search. The ranker can then afford expensive, cross (user×item) features on those 100. YouTube, Pinterest, and TikTok all describe this same funnel.

## Data and evaluation protocol
- **Positive = review ≥ 4★.** Yelp has no click logs, so a good review stands in for "would go here".
- **One state.** This keeps it laptop-sized and makes geography a real signal.
- **5-core filtering.** This removes users and items too sparse to learn from. The tradeoff: it biases evaluation toward active users. That's a known limitation and should be stated.
- **Leave-last-two-out by time.** Per user, the last positive is test, the second-to-last is validation, and everything earlier is train. A random split would let the model "see the future" (for example, train on a 2021 review to predict a 2019 one), which inflates metrics badly.
- **Items absent from train are excluded from evaluation.** An ID-based model cannot retrieve them. Item cold-start is a separate problem; the content features in the item tower are a first step toward it.
- **No leakage anywhere in the pipeline:**
  - The retriever picks its best epoch on validation.
  - The ranker trains on validation targets, with features built only from train.
  - Final numbers use test targets, with features from train+val.

## Two-tower details
- **History pooling in the user tower.** Users are represented by what they've liked, so a new review changes their vector immediately, with no retraining. The user-ID embedding is dropped for 50% of users during training so the tower can't just memorize IDs.
- **In-batch negatives.** Other users' positives in the same batch serve as negatives, so it's free.
  - Popular items appear as negatives more often, so the model learns to push them down.
  - **logQ correction** subtracts log(item frequency) from the logits to undo that bias (Yi et al., 2019).
- **Duplicate masking.** If the same item appears twice in a batch, the copy is not a negative for the other.
- **Learned temperature,** clamped to [0.01, 1].
- **City embedding in the item tower.** Without it, the retriever barely beat popularity on the synthetic data (recall@100 0.40 vs 0.36). With it: 0.46. Where a business is matters more than almost anything else on Yelp.

## Ranker features (18)
Retriever score and rank · item popularity, stars, review count, price · user activity and tenure · history length · **category affinity** (overlap with the user's category distribution) · **co-visitation** (sum, max, and with the last item: how often the candidate co-occurs with the user's recent places) · **distance** from the user's median location and from the last place · same-city flag.

On synthetic data, the top gains were co-visitation, distance, and the retriever score. In other words, the ranker leans on exactly the cross features the dot product can't represent.

## LightGBM LambdaRank vs DCN-v2
- **LambdaRank.** Pairwise gradients weighted by the NDCG change of swapping two items, so it optimizes the ranking metric directly. Strong on tabular features with little tuning; the standard first ranker.
- **DCN-v2.** Explicit feature crosses `x0 ⊙ (W·xl + b) + xl` alongside an MLP. Trained with a **listwise softmax** over each user's candidates, which matches "pick the right one out of 100".
- Both use identical features, so the comparison is about the learner, not the inputs.
- Re-ranking can't change Recall@100: it only reorders the same candidates. A test enforces this. It also means the retriever's recall@N is the ceiling for the whole system.

## ANN
Exact search is used for evaluation so model quality isn't confounded with ANN error. `bench_ann` reports HNSW recall-vs-exact and QPS across efSearch, which is the knob you'd tune for a latency budget.

## Simulated A/B test
Offline gains don't guarantee online gains. Without live traffic, the module simulates an experiment and states its assumptions:
- **Bucketing:** a salted hash of user ID. Deterministic, sticky, and independent of user traits. A new salt per experiment avoids carryover.
- **Clicks:** only on the user's real held-out next place, with position bias 1/log2(rank+2), top-10 only.
- **Readout:** CTR per arm, lift, two-proportion z-test, and a bootstrap CI.
- **SRM check:** a chi-square test against the planned 50/50 split. A sample-ratio mismatch means the assignment is broken and the result can't be trusted.
- **A/A runs:** across many salts, the rate of "significant" results should be about α (5%). If not, the analysis pipeline is wrong.
- **Power:** users per arm needed to detect a given lift. On the default synthetic set, the ranker's +15% lift was *not* significant with only ~1,000 users per arm. The power calculation says why. That is a real lesson about underpowered tests, not a failure.

## Serving
- **Cost per request:** one user-tower forward pass, an HNSW search, 100 feature rows, and one LightGBM predict. All in-process, sub-10 ms on CPU at PA scale.
- **Cold start:** unknown users get the popularity list, and the response labels the source so dashboards can segment it.
- **Already-visited places** are always filtered out. A test checks this.

## What I'd do next
- Hard-negative mining for the retriever.
- A sequence user tower (a GRU or a transformer over history).
- Time-decayed co-visitation.
- Calibrated scores for blending with business rules.
- Replay or off-policy evaluation instead of a simulator.
