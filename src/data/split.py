import numpy as np
def shuffle_items(xs, seed=42):
    xs = list(xs)
    rng = np.random.default_rng(seed)
    rng.shuffle(xs)
    return xs
# 这里需要按照对话进行划分数据集，否则会数据泄露。
def split_conversations(conv_ids, seed=42):
    ids = sorted(set(conv_ids))
    ids = shuffle_items(ids, seed)
    n = len(ids)
    train_end = int(n * 0.7)
    valid_end = train_end + int(n * 0.1)
    mp = {}
    for cid in ids[:train_end]:
        mp[cid] = "train"
    for cid in ids[train_end:valid_end]:
        mp[cid] = "valid"
    for cid in ids[valid_end:]:
        mp[cid] = "test"
    return mp
