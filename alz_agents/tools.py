"""Deterministic tools shared by the agents: data, audit, models, SHAP/LIME, what-if."""
import json, os, hashlib
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
from . import config as C

BUNDLE = os.path.join(C.ART_DIR, "model_bundle.joblib")

def load_df(path=None):
    df = pd.read_csv(path or C.DATA_PATH).rename(columns=C.RENAME)
    return df

def file_hash(path=None):
    return hashlib.md5(open(path or C.DATA_PATH, "rb").read()).hexdigest()[:12]

# ---------------- Agent 1: audit ----------------
def audit(df):
    y = (df[C.TARGET] == "Yes").astype(int)
    rep = {"rows": int(len(df)), "columns": int(df.shape[1]), "missing_total": int(df.isna().sum().sum()),
           "duplicate_rows": int(df.duplicated().sum()), "countries": int(df["Country"].nunique()),
           "positive_rate": round(float(y.mean()), 4), "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4)}
    warns, errs = [], []
    if C.TARGET not in df.columns: errs.append("target column missing")
    if rep["missing_total"] > 0: warns.append(f"{rep['missing_total']} missing values (imputed with train median/mode)")
    if rep["duplicate_rows"] > 0: warns.append(f"{rep['duplicate_rows']} duplicate rows")
    if rep["positive_rate"] < 0.1 or rep["positive_rate"] > 0.9: warns.append("severe class imbalance")
    # leakage / signal screen: single-feature AUC, flag suspiciously high
    X, _ = encode_frame(df)
    aucs = {}
    for c in X.columns:
        try: a = roc_auc_score(y, X[c]); aucs[c] = round(max(a, 1 - a), 3)
        except Exception: pass
    top = dict(sorted(aucs.items(), key=lambda kv: -kv[1])[:5])
    rep["top_single_feature_auc"] = top
    if top and max(top.values()) > 0.95: errs.append(f"possible target leakage: single feature AUC {max(top.values())}")
    weak = [k for k, v in aucs.items() if v < 0.52]
    rep["n_features_without_signal"] = len(weak)
    if len(weak) > 0.5 * len(aucs): warns.append("most features carry no univariate signal; data may be simulated, expect AUC≈0.80 at most")
    rep["warnings"], rep["errors"], rep["ok"] = warns, errs, len(errs) == 0
    rep["data_hash"] = file_hash()
    return rep

def encode_frame(df, cat_maps=None):
    """Ordinal-encode categoricals (stable category order). Returns X, cat_maps."""
    X = df.drop(columns=[C.TARGET], errors="ignore").copy(); fit = cat_maps is None; cat_maps = cat_maps or {}
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            if fit: cat_maps[c] = sorted(X[c].astype(str).unique())
            X[c] = pd.Categorical(X[c].astype(str), categories=cat_maps[c]).codes.astype(float)
            X.loc[X[c] < 0, c] = np.nan
    return X, cat_maps

def make_splits(df, seed=C.SEED):
    y = (df[C.TARGET] == "Yes").astype(int); strat = y.astype(str) + "|" + df["Country"].astype(str)
    idx = np.arange(len(df)); tr, rest = train_test_split(idx, test_size=0.30, stratify=strat, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=strat.iloc[rest], random_state=seed)
    return tr, va, te

# ---------------- Agent 2: models ----------------
def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)

def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def train_models(df, seed=C.SEED):
    import lightgbm as lgb
    if C.FAST: df = df.sample(min(len(df), 15000), random_state=seed).reset_index(drop=True)
    y = (df[C.TARGET] == "Yes").astype(int).values; X, cat_maps = encode_frame(df)
    tr, va, te = make_splits(df, seed)
    fill = X.iloc[tr].median(); X = X.fillna(fill)
    Xtr, Xva, Xte, ytr, yva, yte = X.iloc[tr], X.iloc[va], X.iloc[te], y[tr], y[va], y[te]
    cands = {"LogisticRegression": LogisticRegression(max_iter=2000),
             "LightGBM": lgb.LGBMClassifier(n_estimators=150 if C.FAST else 400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)}
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cands["LogisticRegression"] = make_pipeline(StandardScaler(), cands["LogisticRegression"])
    table, fitted = [], {}
    for n, m in cands.items():
        m.fit(Xtr, ytr); fitted[n] = m
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        table.append({"model": n, "val_auc": roc_auc_score(yva, pv), "test_auc": roc_auc_score(yte, pt), "test_acc@0.5": accuracy_score(yte, pt > .5)})
    tab = pd.DataFrame(table).set_index("model"); best = tab["val_auc"].idxmax(); model = fitted[best]       # selection on validation only
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(model.predict_proba(Xva)[:, 1]), yva)
    cal = lambda Xd: platt.predict_proba(_logit(model.predict_proba(Xd)[:, 1]))[:, 1]
    pv, pt = cal(Xva), cal(Xte)
    ths = np.linspace(0.1, 0.9, 81); thr = float(ths[int(np.argmax([f1_score(yva, pv >= t) for t in ths]))])
    final = {"model": best, "threshold": round(thr, 3), "test_auc": round(roc_auc_score(yte, pt), 4), "test_acc": round(accuracy_score(yte, pt >= thr), 4),
             "test_f1": round(f1_score(yte, pt >= thr), 4), "brier": round(brier_score_loss(yte, pt), 4), "ece": round(ece(yte, pt), 4),
             "majority_baseline": round(max(yte.mean(), 1 - yte.mean()), 4), "n_train": int(len(tr)), "n_test": int(len(te))}
    # leave-one-country-out AUC (a few countries for speed)
    countries = df["Country"].astype(str).values; loco = []
    for cty in sorted(set(countries))[: (4 if C.FAST else 20)]:
        m_te = countries == cty; Xa, ya = X[~m_te], y[~m_te]
        mm = cands[best].__class__(**cands[best].get_params()) if best == "LightGBM" else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        mm.fit(Xa, ya); loco.append(roc_auc_score(y[m_te], mm.predict_proba(X[m_te])[:, 1]))
    final["loco_auc_mean"], final["loco_auc_sd"] = round(float(np.mean(loco)), 4), round(float(np.std(loco)), 4)
    # background + ranges for explanations / validation
    num_cols = [c for c in X.columns if c not in cat_maps]
    bundle = {"model": model, "platt": platt, "threshold": thr, "columns": list(X.columns), "cat_maps": cat_maps, "fill": fill.to_dict(),
              "background": Xtr.sample(min(500, len(Xtr)), random_state=seed), "num_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in num_cols},
              "metrics": final, "comparison": tab.round(4).reset_index().to_dict("records"), "data_hash": file_hash()}
    joblib.dump(bundle, BUNDLE)
    return final, tab.round(4)

def load_bundle(): return joblib.load(BUNDLE) if os.path.exists(BUNDLE) else None

def patient_to_row(b, patient):
    """Validate a patient dict, fill missing fields with train median/mode, return (row DataFrame, assumptions)."""
    assumptions, row = [], {}
    for c in b["columns"]:
        v = patient.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[c] = b["fill"][c]; assumptions.append(f"{c} missing -> training median/mode used")
        elif c in b["cat_maps"]:
            if str(v) not in b["cat_maps"][c]:
                raise ValueError(f"{c}: '{v}' not one of {b['cat_maps'][c]}")
            row[c] = float(b["cat_maps"][c].index(str(v)))
        else:
            lo, hi = b["num_ranges"][c]
            if not (lo <= float(v) <= hi): raise ValueError(f"{c}: {v} outside training range [{lo}, {hi}]")
            row[c] = float(v)
    return pd.DataFrame([row])[b["columns"]], assumptions

def risk(b, X): return b["platt"].predict_proba(_logit(b["model"].predict_proba(X)[:, 1]))[:, 1]

def decode(b, c, code): return b["cat_maps"][c][int(code)] if c in b["cat_maps"] else code

# ---------------- Agent 3: explanations ----------------
def shap_local(b, X):
    import shap
    m = b["model"]
    if b["metrics"]["model"] == "LightGBM":
        sv = shap.TreeExplainer(m).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    else:
        f = lambda A: m.predict_proba(pd.DataFrame(A, columns=b["columns"]))[:, 1]
        sv = shap.KernelExplainer(f, b["background"].iloc[:50].values).shap_values(X.values, nsamples=200)
    return pd.Series(np.asarray(sv).reshape(-1), index=b["columns"])

def lime_local(b, X, seed=C.SEED):
    from lime.lime_tabular import LimeTabularExplainer
    cols = b["columns"]; cat_idx = [i for i, c in enumerate(cols) if c in b["cat_maps"]]
    names = {i: [str(v) for v in b["cat_maps"][cols[i]]] for i in cat_idx}
    ex = LimeTabularExplainer(b["background"].values, feature_names=cols, categorical_features=cat_idx, categorical_names=names,
                              class_names=["No", "Yes"], discretize_continuous=True, random_state=seed)
    f = lambda A: b["model"].predict_proba(pd.DataFrame(A, columns=cols))
    e = ex.explain_instance(X.values[0], f, num_features=len(cols), num_samples=1500)
    return pd.Series({cols[i]: w for i, w in e.as_map()[1]}).reindex(cols).fillna(0.0)

def agreement(shap_s, lime_s, k=5):
    ts, tl = set(shap_s.abs().nlargest(k).index), set(lime_s.abs().nlargest(k).index)
    jac = len(ts & tl) / len(ts | tl)
    common = list(ts | tl); sign = float(np.mean(np.sign(shap_s[common]) == np.sign(lime_s[common])))
    conflicts = [f for f in common if np.sign(shap_s[f]) != np.sign(lime_s[f]) and max(abs(shap_s[f]), 0) > 0]
    return {"top_k": k, "jaccard": round(jac, 3), "sign_agreement": round(sign, 3), "conflict_features": conflicts}

def what_if(b, X, base_risk):
    """One-at-a-time change of modifiable features; model-based association, not causal."""
    out = []
    for f in C.MODIFIABLE:
        if f not in b["columns"]: continue
        alts = range(len(b["cat_maps"][f])) if f in b["cat_maps"] else [X[f].iloc[0] - 2.0, X[f].iloc[0] + 2.0] if f == "BMI" else []
        best = None
        for a in alts:
            if f in b["cat_maps"] and float(a) == X[f].iloc[0]: continue
            Xa = X.copy(); Xa[f] = float(a); d = float(risk(b, Xa)[0] - base_risk)
            if best is None or d < best[1]: best = (a, d)
        if best: out.append({"feature": f, "from": decode(b, f, X[f].iloc[0]), "to": decode(b, f, best[0]) if f in b["cat_maps"] else round(best[0], 1),
                             "delta_risk": round(best[1], 4)})
    return sorted(out, key=lambda r: r["delta_risk"])
