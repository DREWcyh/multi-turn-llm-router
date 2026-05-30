import re
import numpy as np
from data.dataset_utils import ROLE_USER
from features.text_utils import turn_to_text
def word_cnt(txt):
    return len(re.findall(r"\w+", txt, flags=re.UNICODE))
def log1p(x):
    if x < 0:
        x = 0
    return float(np.log1p(x))
def zh_cnt(txt):
    n = 0
    for ch in txt:
        if "\u4e00" <= ch <= "\u9fff":
            n += 1
    return n
def zh_ratio(txt):
    if not txt:
        return 0.0
    return zh_cnt(txt) / len(txt)
def has_code(txt):
    ps = [
        r"```",
        r"\b(def|class|import|return|function|var|let|const|for|while|if)\b",
        r"[{};]\s*$",
        r"Traceback \(most recent call last\)",
    ]
    for p in ps:
        if re.search(p, txt, flags=re.I | re.M):
            return 1
    return 0
def has_math(txt):
    p = r"(\$.*\$|\\frac|\\sum|\\int|[=<>]\s*-?\d|\bsolve\b|\bequation\b|公式|方程)"
    return int(bool(re.search(p, txt, re.I)))
def has_url(txt):
    return int(bool(re.search(r"https?://|www\.", txt, re.I)))
def has_words(txt, en_words, cn_words):
    low = txt.lower()
    for w in en_words:
        p = r"(?<![a-z])" + re.escape(w.lower()) + r"(?![a-z])"
        if re.search(p, low):
            return 1
    for w in cn_words:
        if w in txt:
            return 1
    return 0
def has_pronoun_reference(txt):
    en = ["this", "that", "it", "they", "them", "these", "those", "above", "previous", "earlier", "last", "former", "same", "the above", "previous answer", "last answer", "your answer", "your response", "that answer", "this answer"]
    cn = ["这些", "这种", "这里", "这段", "这份", "这个", "那个", "上面", "上述", "刚才", "之前", "前面", "上一", "上一次", "上文", "下文", "原文", "它", "它们"]
    return has_words(txt, en, cn)
def has_continuation_marker(txt):
    en = ["continue", "next", "more", "elaborate", "expand", "go on", "keep going", "carry on", "also", "add", "add more", "more detail", "more details", "in addition", "then", "further", "another"]
    cn = ["继续", "接着", "然后", "进一步", "补充", "再", "再来", "还有", "另外", "扩展", "展开", "详细", "继续写", "接下来", "下一步", "续写"]
    return has_words(txt, en, cn)
def has_correction_marker(txt):
    en = ["fix", "correct", "revise", "rewrite", "modify", "change", "debug", "edit", "update", "improve", "replace", "remove", "delete", "instead", "make it", "turn it", "convert it", "shorter", "longer", "simplify", "rephrase", "polish"]
    cn = ["改", "修改", "纠正", "重写", "修复", "优化", "调整", "替换", "删除", "去掉", "改成", "换成", "变成", "重新", "润色", "简化", "扩写", "缩短", "翻译成"]
    return has_words(txt, en, cn)
def q_features(c):
    q = c["query"]
    ctx = c["compressed_context"]
    return {
        "query_len_chars": float(len(q)),
        "query_len_words": float(word_cnt(q)),
        "query_log_len_chars": log1p(len(q)),
        "query_log_len_words": log1p(word_cnt(q)),
        "has_code": float(has_code(q) or has_code(ctx)),
        "has_math": float(has_math(q) or has_math(ctx)),
        "has_url": float(has_url(q) or has_url(ctx)),
    }
def ctx_features(c):
    return {
        "num_prior_turns": float(c["num_prior_turns"]),
        "log_num_prior_turns": log1p(c["num_prior_turns"]),
        "raw_context_len": float(c["raw_len"]),
        "raw_context_log_len": log1p(c["raw_len"]),
        "is_first_turn": float(1 if not c["history_messages"] else 0),
    }
def compress_features(c):
    n = c["num_prior_turns"]
    r = c["retained_turn_count"]
    return {
        "compressed_context_len": float(c["compressed_len"]),
        "compressed_context_log_len": log1p(c["compressed_len"]),
        "compression_ratio": float(c["compressed_len"] / c["raw_len"]) if c["raw_len"] else 0.0,
        "retained_turn_count": float(r),
        "retained_turn_ratio": float(r / n) if n else 0.0,
    }
def sim_features(c):
    return {
        "max_history_similarity": float(c["max_history_similarity"]),
        "avg_history_similarity": float(c["avg_history_similarity"]),
    }
def marker_features(c):
    q = c["query"]
    return {
        "has_pronoun_reference": float(has_pronoun_reference(q)),
        "has_continuation_marker": float(has_continuation_marker(q)),
        "has_correction_marker": float(has_correction_marker(q)),
    }
def lang_features(c):
    q = c["query"]
    ctx = c["compressed_context"]
    return {
        "has_chinese": float(1 if zh_cnt(q) or zh_cnt(ctx) else 0),
        "query_chinese_char_ratio": float(zh_ratio(q)),
        "context_chinese_char_ratio": float(zh_ratio(ctx)),
    }
FEATURE_GROUP_BUILDERS = [
    ("query_attributes", q_features),
    ("context_scale", ctx_features),
    ("compression_attributes", compress_features),
    ("relation_semantic", sim_features),
    ("multi_turn_markers", marker_features),
    ("language_attributes", lang_features),
]
def build_attribute_features(query, history_messages, compressed_context, retained_turn_count, max_history_similarity, avg_history_similarity):
    raw = turn_to_text(history_messages)
    c = {
        "query": query or "",
        "history_messages": history_messages,
        "compressed_context": compressed_context or "",
        "retained_turn_count": retained_turn_count,
        "max_history_similarity": max_history_similarity,
        "avg_history_similarity": avg_history_similarity,
        "raw_len": len(raw),
        "compressed_len": len(compressed_context or ""),
        "num_prior_turns": sum(1 for m in history_messages if m.get("role") == ROLE_USER),
    }
    vals = {}
    names = []
    groups = []
    for g, fn in FEATURE_GROUP_BUILDERS:
        one = fn(c)
        for k, v in one.items():
            names.append(k)
            groups.append(g)
            vals[k] = float(v)
    return np.array([vals[k] for k in names], dtype=np.float32), names, groups, vals
