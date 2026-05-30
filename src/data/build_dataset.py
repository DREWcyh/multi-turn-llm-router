import argparse
from pathlib import Path
from data.dataset_utils import iter_turn_samples, load_labels
from data.split import split_conversations
from utils.io_utils import write_json, write_jsonl
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conversations", default="data/conversations.jsonl")
    p.add_argument("--features", default="data/features/qwen06b_20k.jsonl")
    p.add_argument("--output", default="artifacts/samples/samples.jsonl")
    p.add_argument("--label-field", default="label_t2")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()
def build_samples(args):
    labs = load_labels(args.features, args.label_field)
    rows = list(iter_turn_samples(args.conversations, labs, args.max_samples))
    mp = split_conversations([x["conversation_hash"] for x in rows], seed=args.seed)
    for x in rows:
        x["split"] = mp[x["conversation_hash"]]
    return rows, mp
def main():
    args = get_args()
    out = Path(args.output)
    split_out = out.with_name(out.stem + "_splits.json")
    if out.exists() and not args.force:
        print(str(out) + " exists, use --force if rebuild")
        return
    rows, mp = build_samples(args)
    n = write_jsonl(out, rows)
    write_json(split_out, mp)
    print("samples:", n, out)
    print("splits:", split_out)
if __name__ == "__main__":
    main()
