from data.dataset_utils import ROLE_USER
def msg_text(msg):
    role = msg.get("role", "")
    txt = msg.get("content", "") or ""
    return f"{role}: {txt}".strip()
def turn_to_text(msgs):
    lines = []
    for m in msgs:
        if m.get("content"):
            lines.append(msg_text(m))
    return "\n".join(lines)
def pair_history_turns(history_msgs):
    res = []
    now = []
    user_i = -1
    for m in history_msgs:
        if m.get("role") == ROLE_USER:
            if now:
                res.append({"turn_idx": user_i, "messages": now})
            user_i += 1
            now = [m]
        elif now:
            now.append(m)
    if now:
        res.append({"turn_idx": user_i, "messages": now})
    return res
def query_text(q):
    return "query: " + (q or "")
def passage_text(txt):
    if not txt:
        return ""
    return "passage: " + txt
