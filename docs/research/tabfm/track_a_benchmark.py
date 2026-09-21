"""
Track A — TabFM/TabPFN research benchmark for RuVector (ADR-272).

Question 1 (primary): Is the SHIPPABLE model (TabPFN-v2, Apache-2.0 weights)
competitive zero-shot with HEAVILY-TUNED GBDTs on standard tabular data?

Question 2 (ceiling): How much accuracy do we forgo vs the newer, NON-COMMERCIAL
TabPFN weights (v2.5/v3)? Try to load them; if HF-gated, record and defer to
published numbers.

Protocol: nested CV. Tree baselines get RandomizedSearchCV on each training fold
(fair "tuned" comparison). TabPFN gets NO tuning (its value proposition).
Metric: ROC AUC (binary: roc_auc; multiclass: macro roc_auc_ovr) + accuracy.
License note: internal benchmarking is explicitly exempt under both the TabFM and
TabPFN non-commercial licenses, so running gated weights here is permitted.
"""
import time, json, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from tabpfn import TabPFNClassifier

RNG = 42
N_OUTER = 5
DATASETS = [   # (openml name/id, label) — small, standard, CPU-friendly
    ("credit-g", 31), ("diabetes", 37), ("blood-transfusion-service-center", 1464),
    ("breast-w", 15), ("qsar-biodeg", 1494), ("vehicle", 54),
]

def load(did):
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X, y = d.data.copy(), d.target
    y = y.astype(str)
    classes = sorted(y.unique())
    y = y.map({c: i for i, c in enumerate(classes)}).to_numpy()
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    num = [c for c in X.columns if c not in cat]
    return X, y, num, cat, len(classes)

def auc(y_true, proba, n_cls):
    if n_cls == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")

def tree_prep(num, cat):  # ordinal-encode categoricals, impute — for XGB/HistGBM/RF
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
    ])

def lin_prep(num, cat):   # one-hot + scale — for LogReg
    from sklearn.preprocessing import OneHotEncoder
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

def tabpfn_prep(X, num, cat):  # numeric matrix; ordinal-encode categoricals
    Xn = X.copy()
    for c in cat:
        Xn[c] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1
               ).fit_transform(Xn[[c]].astype(str))
    return Xn.apply(lambda s: s.fillna(s.median() if s.dtype != object else 0)).to_numpy(dtype=float)

def tuned_xgb(n_cls):
    est = xgb.XGBClassifier(tree_method="hist", eval_metric="logloss",
                            objective="multi:softprob" if n_cls > 2 else "binary:logistic",
                            n_jobs=4, verbosity=0, random_state=RNG)
    grid = {"n_estimators": [100, 300, 600], "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.02, 0.05, 0.1, 0.2], "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5]}
    return RandomizedSearchCV(est, grid, n_iter=15, cv=3, scoring="roc_auc_ovr",
                              random_state=RNG, n_jobs=4)

def tuned_hist():
    est = HistGradientBoostingClassifier(random_state=RNG)
    grid = {"max_iter": [200, 400], "learning_rate": [0.03, 0.06, 0.1, 0.2],
            "max_depth": [None, 3, 6, 10], "l2_regularization": [0.0, 0.1, 1.0],
            "max_leaf_nodes": [15, 31, 63]}
    return RandomizedSearchCV(est, grid, n_iter=12, cv=3, scoring="roc_auc_ovr",
                              random_state=RNG, n_jobs=4)

# try to load a ceiling (non-commercial) model once
def load_ceiling():
    for mp, tag in [("tabpfn-v2.5-classifier.ckpt", "v2.5"), (None, "v3-auto")]:
        try:
            c = TabPFNClassifier(device="cpu", **({"model_path": mp} if mp else {}))
            from sklearn.datasets import load_breast_cancer
            Xb, yb = load_breast_cancer(return_X_y=True)
            c.fit(Xb[:100], yb[:100]); c.predict_proba(Xb[:3])
            return c, tag
        except Exception as e:
            print(f"  [ceiling {tag}] unavailable: {type(e).__name__}: {str(e)[:90]}")
    return None, None

def main():
    print("Loading ceiling (non-commercial) model for Q2 ...")
    ceiling_clf, ceiling_tag = load_ceiling()
    results = {}
    for name, did in DATASETS:
        print(f"\n=== {name} (openml {did}) ===")
        X, y, num, cat, n_cls = load(did)
        print(f"  n={len(y)} feat={X.shape[1]} ({len(num)} num/{len(cat)} cat) classes={n_cls}")
        Xtab = tabpfn_prep(X, num, cat)
        skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RNG)
        per = {m: {"auc": [], "acc": [], "t": []} for m in
               ["TabPFN-v2(ship)", "TabPFN-"+(ceiling_tag or "ceiling"), "XGB-tuned", "HistGBM-tuned", "RandomForest", "LogReg"]}
        for tr, te in skf.split(X, y):
            ytr, yte = y[tr], y[te]
            # --- TabPFN v2 (shippable, zero-shot) ---
            t = time.time()
            m = TabPFNClassifier(device="cpu", model_path="tabpfn-v2-classifier.ckpt")
            m.fit(Xtab[tr], ytr); p = m.predict_proba(Xtab[te])
            per["TabPFN-v2(ship)"]["auc"].append(auc(yte, p, n_cls))
            per["TabPFN-v2(ship)"]["acc"].append(accuracy_score(yte, p.argmax(1)))
            per["TabPFN-v2(ship)"]["t"].append(time.time() - t)
            # --- ceiling (non-commercial), if available ---
            ck = "TabPFN-"+(ceiling_tag or "ceiling")
            if ceiling_clf is not None:
                t = time.time()
                ceiling_clf.fit(Xtab[tr], ytr); p = ceiling_clf.predict_proba(Xtab[te])
                per[ck]["auc"].append(auc(yte, p, n_cls)); per[ck]["acc"].append(accuracy_score(yte, p.argmax(1)))
                per[ck]["t"].append(time.time() - t)
            # --- tuned tree baselines (nested search on train fold) ---
            for mname, mk, prep in [("XGB-tuned", tuned_xgb(n_cls), tree_prep(num, cat)),
                                    ("HistGBM-tuned", tuned_hist(), tree_prep(num, cat))]:
                t = time.time()
                Xt = prep.fit_transform(X.iloc[tr]); Xv = prep.transform(X.iloc[te])
                mk.fit(Xt, ytr); p = mk.predict_proba(Xv)
                per[mname]["auc"].append(auc(yte, p, n_cls)); per[mname]["acc"].append(accuracy_score(yte, p.argmax(1)))
                per[mname]["t"].append(time.time() - t)
            # --- untuned references ---
            prep = tree_prep(num, cat)
            Xt = prep.fit_transform(X.iloc[tr]); Xv = prep.transform(X.iloc[te])
            rf = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=RNG).fit(Xt, ytr)
            p = rf.predict_proba(Xv); per["RandomForest"]["auc"].append(auc(yte, p, n_cls)); per["RandomForest"]["acc"].append(accuracy_score(yte, p.argmax(1))); per["RandomForest"]["t"].append(0)
            lp = lin_prep(num, cat); Xt = lp.fit_transform(X.iloc[tr]); Xv = lp.transform(X.iloc[te])
            lr = LogisticRegression(max_iter=2000).fit(Xt, ytr)
            p = lr.predict_proba(Xv); per["LogReg"]["auc"].append(auc(yte, p, n_cls)); per["LogReg"]["acc"].append(accuracy_score(yte, p.argmax(1))); per["LogReg"]["t"].append(0)
        results[name] = {m: {"auc_mean": float(np.mean(v["auc"])) if v["auc"] else None,
                             "auc_std": float(np.std(v["auc"])) if v["auc"] else None,
                             "acc_mean": float(np.mean(v["acc"])) if v["acc"] else None,
                             "t_mean": float(np.mean(v["t"])) if v["t"] else None} for m, v in per.items()}
        for m, r in results[name].items():
            if r["auc_mean"] is not None:
                print(f"  {m:22s} AUC {r['auc_mean']:.4f}±{r['auc_std']:.3f}  acc {r['acc_mean']:.4f}  {r['t_mean']:.1f}s")
    json.dump(results, open("track_a_results.json", "w"), indent=2)
    print("\nSaved track_a_results.json")

if __name__ == "__main__":
    main()
