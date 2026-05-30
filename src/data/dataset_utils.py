from tqdm import tqdm
from utils.io_utils import read_jsonl
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
def load_labels(path, label_field="label_t2"):
    ans = {}
    for item in tqdm(read_jsonl(path), desc="labels"):
        k = (item["conversation_hash"], int(item["turn_idx"]))
        ans[k] = {
            "label": int(item[label_field]),
            "weak_score": float(item["weak_score"]),
            "strong_score": float(item["strong_score"]),
        }
    return ans
def clean_text(text):
    if text is None:
        return ""
    return str(text).strip()
def iter_turn_samples(conversations_path, labels, max_samples=None):
    cnt = {
        "total_user_turns": 0,
        "aligned": 0,
        "missing": 0,
    }

    for conv in tqdm(read_jsonl(conversations_path), desc="conversations"):
        cid = conv["conversation_hash"]
        old_msgs = []
        uidx = 0

        for msg in conv.get("conversation", []):
            if msg.get("role") == ROLE_USER:
                cnt["total_user_turns"] += 1
                k = (cid, uidx)
                lab = labels.get(k)

                if lab is None:
                    cnt["missing"] += 1
                else:
                    cnt["aligned"] += 1
                    yield {
                        "sample_id": f"{cid}:{uidx}",
                        "conversation_hash": cid,
                        "turn_idx": uidx,
                        "query": clean_text(msg.get("content")),
                        "history_messages": list(old_msgs),
                        **lab,
                    }

                    if max_samples is not None and cnt["aligned"] >= max_samples:
                        print_stats(cnt)
                        return

                uidx += 1

            old_msgs.append(msg)

    print_stats(cnt)
def print_stats(stats):
    print("alignment stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
