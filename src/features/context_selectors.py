import numpy as np
from features.text_utils import pair_history_turns, turn_to_text
CONTEXT_STRATEGIES = ["query_only", "full", "recent_k", "semantic_top_k", "recent_semantic"]
def select_context(strategy, history_msgs, query_emb=None, turn_embs=None, recent_k=3, semantic_k=3):
    ts = pair_history_turns(history_msgs)
    raw = turn_to_text(history_msgs)
    if len(ts) == 0 or strategy == "query_only":
        return "", [], 0.0, 0.0, []
    if strategy == "full":
        return raw, [x["turn_idx"] for x in ts], 0.0, 0.0, []
    ss = []
    if turn_embs is not None and query_emb is not None and len(turn_embs) > 0:
        ss = (turn_embs @ query_emb).astype(float).tolist()
    choose = set()
    if strategy in ["recent_k", "recent_semantic"]:
        st = len(ts) - recent_k
        if st < 0:
            st = 0
        for i in range(st, len(ts)):
            choose.add(i)
    if strategy in ["semantic_top_k", "recent_semantic"] and ss:
        left = []
        for i in range(len(ts)):
            if i not in choose:
                left.append(i)
        left.sort(key=lambda x: ss[x], reverse=True)
        for i in left[:semantic_k]:
            choose.add(i)
    all_msg = []
    turn_ids = []
    picked_ss = []
    for i in sorted(choose):
        all_msg += ts[i]["messages"]
        turn_ids.append(ts[i]["turn_idx"])
        if ss:
            picked_ss.append(ss[i])
    txt = turn_to_text(all_msg)
    mx = max(picked_ss) if picked_ss else 0.0
    av = float(np.mean(picked_ss)) if picked_ss else 0.0
    tmp = []
    for i in range(len(ss)):
        tmp.append({"turn_idx": ts[i]["turn_idx"], "similarity": ss[i]})
    return txt, turn_ids, float(mx), av, tmp
