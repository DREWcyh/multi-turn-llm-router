import numpy as np
SEMANTIC_GROUPS = {"query_semantic", "context_semantic", "relation_semantic"}
ATTRIBUTE_GROUPS = {"query_attributes", "context_scale", "compression_attributes", "multi_turn_markers", "language_attributes"}
FEATURE_SETS = {
    "full_features": set(),
    "remove_semantic": SEMANTIC_GROUPS,
    "remove_attribute": ATTRIBUTE_GROUPS,
    "remove_query_semantic": {"query_semantic"},
    "remove_context_semantic": {"context_semantic"},
    "remove_relation_semantic": {"relation_semantic"},
    "remove_query_attributes": {"query_attributes"},
    "remove_context_scale": {"context_scale"},
    "remove_compression_attributes": {"compression_attributes"},
    "remove_multi_turn_markers": {"multi_turn_markers"},
    "remove_language_attributes": {"language_attributes"},
}
def feature_set_suffix(feature_set):
    if feature_set == "full_features":
        return ""
    return "_" + feature_set
def get_feature_meta(data, x):
    if "feature_names" not in data.files or "feature_groups" not in data.files:
        raise ValueError("feature file has no feature_names or feature_groups")
    names = data["feature_names"].astype(str)
    groups = data["feature_groups"].astype(str)
    if len(groups) != x.shape[1]:
        raise ValueError("feature meta length is not same as X")
    return names, groups
def select_feature_matrix(x, feature_groups, feature_set):
    if not feature_set:
        feature_set = "full_features"
    if feature_set not in FEATURE_SETS:
        opts = ", ".join(sorted(FEATURE_SETS))
        raise ValueError("unknown feature_set: " + feature_set + ", choices: " + opts)
    remove = FEATURE_SETS[feature_set]
    gs = np.asarray(feature_groups).astype(str)
    keep = []
    for g in gs:
        keep.append(g not in remove)
    keep = np.array(keep, dtype=bool)
    if not keep.any():
        raise ValueError(feature_set + " removes all columns")
    return x[:, keep].astype(np.float32)
