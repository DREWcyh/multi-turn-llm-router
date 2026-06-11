import argparse
import json
import numpy as np
from tqdm import tqdm
from features.context_selectors import CONTEXT_STRATEGIES, select_context
from features.encoders import encode_texts, load_encoder
from features.attribute_features import build_attribute_features
from features.text_utils import pair_history_turns, passage_text, query_text, turn_to_text
from utils.io_utils import ensure_dir, read_jsonl
def build_turn_texts(samples):
    all_txt = []
    ranges = []
    for s in samples:
        ts = pair_history_turns(s["history_messages"])
        st = len(all_txt)
        for t in ts:
            all_txt.append(passage_text(turn_to_text(t["messages"])))
        ranges.append((st, len(all_txt)))
    return all_txt, ranges
def build_feature_matrix(q_emb, ctx_emb, attr_rows):
    cos = np.sum(q_emb * ctx_emb, axis=1, keepdims=True).astype(np.float32)
    attr = np.vstack(attr_rows).astype(np.float32)
    x = np.hstack([
        q_emb,
        ctx_emb,
        np.abs(q_emb - ctx_emb),
        q_emb * ctx_emb,
        cos,
        attr,
    ])
    return x.astype(np.float32)
def build_feature_meta(dim, attr_names, attr_groups):
    names = []
    groups = []
    for i in range(dim):
        names.append(f"query_emb_{i}")
        groups.append("query_semantic")
    for i in range(dim):
        names.append(f"context_emb_{i}")
        groups.append("context_semantic")
    for i in range(dim):
        names.append(f"abs_diff_{i}")
        groups.append("relation_semantic")
    for i in range(dim):
        names.append(f"emb_product_{i}")
        groups.append("relation_semantic")
    names.append("cosine")
    groups.append("relation_semantic")
    names += attr_names or []
    groups += attr_groups or []
    return names, groups
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="artifacts/samples/samples.jsonl")
    p.add_argument("--output-dir", default="artifacts/features")
    p.add_argument("--strategy", default="recent_semantic", choices=CONTEXT_STRATEGIES)
    p.add_argument("--embedding-model", default="models/multilingual-e5-small")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--recent-k", type=int, default=2)
    p.add_argument("--semantic-k", type=int, default=2)
    p.add_argument("--output-name", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()
def main():
    args = get_args()
    out_dir = ensure_dir(args.output_dir)
    tag = args.output_name or f"{args.strategy}_n{args.max_samples or 'all'}"
    if tag.endswith(".npz"):
        tag = tag[:-4]
    npz_path = out_dir / f"{tag}.npz"
    meta_path = out_dir / f"{tag}.jsonl"
    if npz_path.exists() and meta_path.exists() and not args.force:
        print(str(npz_path) + " exists, use --force if rebuild")
        return
    data = list(read_jsonl(args.samples))
    if args.max_samples is not None:
        data = data[:args.max_samples]
    if not data:
        raise SystemExit("no samples")
    enc = load_encoder(args.embedding_model)
    q_texts = []
    for s in data:
        q_texts.append(query_text(s["query"]))
    q_emb = encode_texts(enc, q_texts, args.batch_size)
    turn_texts, ranges = build_turn_texts(data)
    if turn_texts:
        turn_emb = encode_texts(enc, turn_texts, args.batch_size)
    else:
        turn_emb = np.zeros((0, q_emb.shape[1]), dtype=np.float32)
    ctx_texts = []
    attr_rows = []
    rows = []
    attr_names = None
    attr_groups = None
    for i, s in enumerate(tqdm(data, desc="select_context")):
        a, b = ranges[i]
        ctx, chosen, mx, av, sims = select_context(
            args.strategy,
            s["history_messages"],
            q_emb[i],
            turn_emb[a:b],
            args.recent_k,
            args.semantic_k,
        )
        ctx_texts.append(passage_text(ctx))
        arr, names, groups, vals = build_attribute_features(
            s["query"],
            s["history_messages"],
            ctx,
            len(chosen),
            mx,
            av,
        )
        attr_names = names
        attr_groups = groups
        attr_rows.append(arr)
        rows.append({
            "sample_id": s["sample_id"],
            "conversation_hash": s["conversation_hash"],
            "turn_idx": s["turn_idx"],
            "split": s["split"],
            "selected_turns": chosen,
            "selected_turn_similarities": sims,
            "compressed_context_preview": ctx[:1200],
            "attribute_features": vals,
            "attribute_feature_groups": dict(zip(names, groups)),
        })
    ctx_emb = encode_texts(enc, ctx_texts, args.batch_size)
    x = build_feature_matrix(q_emb, ctx_emb, attr_rows)
    dim = int(q_emb.shape[1])
    fnames, fgroups = build_feature_meta(dim, attr_names, attr_groups)
    np.savez_compressed(
        npz_path,
        X=x,
        y=np.array([s["label"] for s in data], dtype=np.int64),
        weak_score=np.array([s["weak_score"] for s in data], dtype=np.float32),
        strong_score=np.array([s["strong_score"] for s in data], dtype=np.float32),
        split=np.array([s["split"] for s in data]),
        sample_id=np.array([s["sample_id"] for s in data]),
        embedding_dim=np.array([dim], dtype=np.int64),
        feature_names=np.array(fnames),
        feature_groups=np.array(fgroups),
        strategy=np.array([args.strategy]),
        embedding_model=np.array([args.embedding_model]),
        recent_k=np.array([args.recent_k], dtype=np.int64),
        semantic_k=np.array([args.semantic_k], dtype=np.int64),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("features:", npz_path)
    print("meta:", meta_path)
    print("shape:", x.shape)
if __name__ == "__main__":
    main()
