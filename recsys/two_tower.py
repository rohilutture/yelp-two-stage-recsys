"""Stage 1: two-tower candidate retrieval.

User tower: mean-pooled embeddings of the user's recent history items
            + user ID embedding + numeric user features  -> MLP -> unit vector
Item tower: item ID embedding + mean of category embeddings + city embedding
            + numeric item features                        -> MLP -> unit vector

Training: each training positive (u, i at time t) becomes an example whose history
is u's positives BEFORE t, so the model learns "next item given past items".
Loss: sampled softmax with in-batch negatives, with two standard fixes:
  * logQ correction: popular items show up as in-batch negatives more often, so
    subtract log(sampling prob) from their logits; otherwise the model learns to
    suppress popular items.
  * duplicate masking: if the same item appears twice in a batch, the copy must
    not be treated as a negative for the other.

Because the user tower is driven mostly by history, it is inductive: at test time
we feed train+val history without retraining.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_hidden, d_out, dropout=0.1):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(d_hidden, d_out))


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_cats, d_user_num, d_item_num, dim=64,
                 hidden=256, user_id_dropout=0.5, n_cities=1):
        super().__init__()
        self.city = nn.Embedding(n_cities, dim)  # where a place is matters a lot on Yelp
        self.n_items, self.n_cats = n_items, n_cats
        self.item_id = nn.Embedding(n_items + 1, dim, padding_idx=n_items)  # last = pad
        self.cat = nn.EmbeddingBag(n_cats + 1, dim, mode="mean", padding_idx=n_cats)
        self.user_id = nn.Embedding(n_users, dim)
        self.uid_drop = user_id_dropout
        self.item_mlp = mlp(3 * dim + d_item_num, hidden, dim)
        self.user_mlp = mlp(2 * dim + d_user_num, hidden, dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def item_vec(self, items, cat_idx, cat_offsets, item_num, city):
        x = torch.cat([self.item_id(items), self.cat(cat_idx, cat_offsets), self.city(city),
                       item_num], -1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_vec(self, users, hist, user_num):
        mask = (hist != self.n_items).unsqueeze(-1).float()
        pooled = (self.item_id(hist) * mask).sum(1) / mask.sum(1).clamp(min=1)
        uid = self.user_id(users)
        if self.training and self.uid_drop:
            # Drop the ID embedding for whole users so the tower can't lean on ID
            # memorization and must use history (keeps it inductive).
            keep = (torch.rand(len(users), 1, device=uid.device) > self.uid_drop).float()
            uid = uid * keep
        x = torch.cat([pooled, uid, user_num], -1)
        return F.normalize(self.user_mlp(x), dim=-1)


class Features:
    """Tensors for item and user side inputs, shared by training and serving."""

    def __init__(self, ds):
        self.ds = ds
        self.item_num = torch.tensor(ds.item_num)
        self.user_num = torch.tensor(ds.user_num)
        self.item_city = torch.tensor(ds.item_city)
        self.n_cities = int(ds.item_city.max()) + 1
        self.cats = [c if c else [ds.n_cats] for c in ds.item_cats]  # pad empty

    def cat_bag(self, items: torch.Tensor):
        lists = [self.cats[i] for i in items.tolist()]
        offsets = torch.tensor([0] + [len(c) for c in lists[:-1]]).cumsum(0)
        return torch.tensor([c for cs in lists for c in cs]), offsets

    def items(self, items: torch.Tensor):
        idx, off = self.cat_bag(items)
        return items, idx, off, self.item_num[items], self.item_city[items]

    def pad_hist(self, hists: list[list[int]], hist_len: int):
        out = torch.full((len(hists), hist_len), self.ds.n_items, dtype=torch.long)
        for r, h in enumerate(hists):
            h = h[-hist_len:]
            if h:
                out[r, -len(h):] = torch.tensor(h)
        return out


def training_examples(train_hists: list[list[int]]):
    """(user, position) for every training positive with at least one prior item."""
    return [(u, p) for u, h in enumerate(train_hists) for p in range(1, len(h))]


def train_two_tower(ds, train_hists, epochs=5, batch_size=1024, dim=64, hist_len=50,
                    lr=2e-3, weight_decay=1e-6, seed=0, log=print, val_fn=None):
    torch.manual_seed(seed)
    random.seed(seed)
    feats = Features(ds)
    model = TwoTower(ds.n_users, ds.n_items, ds.n_cats, ds.user_num.shape[1],
                     ds.item_num.shape[1], dim=dim, n_cities=feats.n_cities)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    counts = np.bincount(ds.train.i, minlength=ds.n_items) + 1
    log_q = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
    examples = training_examples(train_hists)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        random.shuffle(examples)
        total = 0.0
        for s in range(0, len(examples), batch_size):
            batch = examples[s:s + batch_size]
            users = torch.tensor([u for u, _ in batch])
            pos = torch.tensor([train_hists[u][p] for u, p in batch])
            hist = feats.pad_hist([train_hists[u][:p] for u, p in batch], hist_len)
            uv = model.user_vec(users, hist, feats.user_num[users])
            iv = model.item_vec(*feats.items(pos))
            logits = uv @ iv.T / model.log_tau.exp() - log_q[pos][None, :]
            dup = (pos[None, :] == pos[:, None]) & ~torch.eye(len(pos), dtype=torch.bool)
            logits = logits.masked_fill(dup, float("-inf"))
            loss = F.cross_entropy(logits, torch.arange(len(pos)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.log_tau.clamp_(math.log(0.01), math.log(1.0))
            total += loss.item() * len(batch)
        msg = f"epoch {ep + 1}: loss {total / len(examples):.4f}"
        if val_fn:
            score = val_fn(model, feats)
            msg += f"  val recall@100 {score:.4f}"
            if score > best:  # keep the best epoch (early-stopping style)
                best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
        log(msg)
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, feats


@torch.no_grad()
def item_embeddings(model, feats, batch=4096) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, feats.ds.n_items, batch):
        items = torch.arange(s, min(s + batch, feats.ds.n_items))
        out.append(model.item_vec(*feats.items(items)))
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def user_embeddings(model, feats, users: list[int], hists: list[list[int]],
                    hist_len=50) -> np.ndarray:
    model.eval()
    u = torch.tensor(users)
    return model.user_vec(u, feats.pad_hist(hists, hist_len),
                          feats.user_num[u]).numpy().astype(np.float32)
