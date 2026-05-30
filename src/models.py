from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
DEFAULT_MODELS = ["logreg", "svm", "gnb", "knn", "dt", "rf", "xgboost", "mlp"]
def build_models(seed=42, names=None):
    if names is None:
        names = DEFAULT_MODELS
    ms = {}
    if "logreg" in names:
        ms["logreg"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
    if "svm" in names:
        base = LinearSVC(class_weight="balanced", random_state=seed)
        ms["svm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=3)),
        ])
    if "gnb" in names:
        ms["gnb"] = GaussianNB()
    if "knn" in names:
        ms["knn"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        ])
    if "dt" in names:
        ms["dt"] = DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=seed)
    if "rf" in names:
        ms["rf"] = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if "xgboost" in names:
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise SystemExit("need xgboost") from e
        ms["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    if "mlp" in names:
        ms["mlp"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True, random_state=seed)),
        ])
    return ms
