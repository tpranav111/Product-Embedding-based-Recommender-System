import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Embedding, Input, Reshape, Dense, Dot
from keras.models import Model
from sklearn.manifold import TSNE

pd.set_option("chained_assignment", None)

RANDOM_SEED = 100
WINDOW_SIZE = 2
EMBEDDING_SIZE = 50
POSITIVE_SAMPLES = 1_000
NEGATIVE_RATIO = 4
EPOCHS = 50


def unix_to_dt(ts):
    return datetime.utcfromtimestamp(ts / 1000)


def load_events(path: str) -> pd.DataFrame:
    df = (
        pd.read_csv(path)
        .query("event == 'view'")
        .drop(columns=["event", "transactionid"])
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"].apply(unix_to_dt))
    df["date"] = df["timestamp"].dt.date
    return df


def filter_sessions(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    df["view_count"] = df.groupby(["visitorid", "date"])["itemid"].transform("count")
    df = df.loc[df["view_count"] > 1].sort_values(["visitorid", "timestamp"])
    df["timestamp_lag1"] = df.groupby(["visitorid", "date"])["timestamp"].shift()
    df["time_elapsed"] = (df["timestamp"] - df["timestamp_lag1"]).dt.total_seconds()
    df["time_elapsed_cum"] = df.groupby(["visitorid", "date"])["time_elapsed"].cumsum()
    df.fillna({"time_elapsed_cum": 0}, inplace=True)
    return df.loc[df["time_elapsed_cum"] <= minutes * 60]


def build_sequences(df: pd.DataFrame, min_len: int = 5):
    df["itemid"] = df["itemid"].astype(str)
    seqs = (
        df.groupby(["visitorid", "date"])["itemid"]
        .apply(list)
        .tolist()
    )

    def dedup(seq):
        deduped = [seq[0]]
        deduped.extend(i for i in seq[1:] if i != deduped[-1])
        return deduped

    return [dedup(seq) for seq in seqs if len(seq) >= min_len]


def create_mappings(seqs):
    items = sorted({i for s in seqs for i in s})
    to_int = {it: idx for idx, it in enumerate(items)}
    to_item = {idx: it for it, idx in to_int.items()}
    return items, to_int, to_item


def generate_pairs(seqs, to_int, window):
    pairs = []
    for seq in seqs:
        n = len(seq)
        for i, target in enumerate(seq):
            ctx = seq[max(0, i - window) : min(i + window + 1, n)]
            pairs.extend((target, c) for c in ctx if c != target)
    idx_pairs = [(to_int[a], to_int[b]) for a, b in pairs]
    return idx_pairs, set(idx_pairs)


def batch_gen(idx_pairs, idx_set, item_cnt, n_pos, neg_ratio, seed):
    random.seed(seed)
    batch_sz = n_pos * (1 + neg_ratio)
    while True:
        batch = np.zeros((batch_sz, 3), dtype=np.int32)
        for i, (t, c) in enumerate(random.sample(idx_pairs, n_pos)):
            batch[i] = t, c, 1
        j = i + 1
        while j < batch_sz:
            t = random.randrange(item_cnt)
            c = random.randrange(item_cnt)
            if t != c and (t, c) not in idx_set:
                batch[j] = t, c, 0
                j += 1
        np.random.shuffle(batch)
        yield {"target": batch[:, 0], "context": batch[:, 1]}, batch[:, 2]


def build_model(item_cnt, emb_dim):
    target = Input(shape=(1,), name="target")
    context = Input(shape=(1,), name="context")
    t_emb = Embedding(item_cnt, emb_dim, name="target_emb")(target)
    c_emb = Embedding(item_cnt, emb_dim, name="context_emb")(context)
    dot = Dot(axes=2, normalize=True)([t_emb, c_emb])
    out = Dense(1, activation="sigmoid")(Reshape((1,))(dot))
    model = Model([target, context], out)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model


def normalize_embeddings(model):
    w = model.get_layer("target_emb").get_weights()[0]
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def similar_items(item, k, emb, to_int, to_item):
    idx = to_int.get(item)
    if idx is None:
        return []
    sims = emb @ emb[idx]
    top = np.argsort(sims)[-(k + 1) :][::-1]
    return [(to_item[i], sims[i]) for i in top if i != idx][:k]


def main():
    events = load_events("events.csv")
    sessions = filter_sessions(events)
    seqs = build_sequences(sessions)
    items, to_int, to_item = create_mappings(seqs)
    pairs, pair_set = generate_pairs(seqs, to_int, WINDOW_SIZE)
    gen = batch_gen(pairs, pair_set, len(items), POSITIVE_SAMPLES, NEGATIVE_RATIO, RANDOM_SEED)
    model = build_model(len(items), EMBEDDING_SIZE)
    steps = len(pairs) // POSITIVE_SAMPLES
    model.fit(gen, steps_per_epoch=steps, epochs=EPOCHS, verbose=2)
    emb = normalize_embeddings(model)
    item0 = items[0]
    print(f"Most similar to {item0}:")
    for itm, score in similar_items(item0, 10, emb, to_int, to_item):
        print(itm, f"{score:.2f}")


if __name__ == "__main__":
    main()
