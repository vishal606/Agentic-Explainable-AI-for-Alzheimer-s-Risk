"""The four agents. Each is a function state -> partial state update (a LangGraph node)."""
import json, os, re, numpy as np
from . import config as C, tools as T, memory as M
from .kb import HybridRetriever

_RETR = None
def retriever():
    global _RETR
    if _RETR is None: _RETR = HybridRetriever(C.KB_DIR)
    return _RETR

AUDIT_PATH = os.path.join(C.ART_DIR, "audit.json")

def _py(o):
    """Convert numpy types to plain Python so LangGraph can checkpoint the state."""
    if isinstance(o, dict): return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_py(v) for v in o]
    if isinstance(o, np.generic): return o.item()
    return o

def safe(fn):
    def w(state): return _py(fn(state))
    w.__name__ = fn.__name__; return w

# ---------- Agent 1: Data Auditor ----------
@safe
def data_auditor(state):
    df = T.load_df(); rep = T.audit(df)
    json.dump(rep, open(AUDIT_PATH, "w"), indent=2)
    msg = f"[Auditor] {rep['rows']} rows, positive rate {rep['positive_rate']}, majority baseline {rep['majority_baseline']}; " \
          f"{'OK' if rep['ok'] else 'BLOCKED: ' + '; '.join(rep['errors'])}"
    return {"audit": rep, "log": [msg]}

# ---------- Agent 2: Predictor ----------
@safe
def predictor(state):
    final, tab = T.train_models(T.load_df())
    msg = f"[Predictor] best={final['model']} test AUC {final['test_auc']} acc {final['test_acc']} (majority {final['majority_baseline']}), " \
          f"LOCO AUC {final['loco_auc_mean']}±{final['loco_auc_sd']}, ECE {final['ece']}"
    return {"train_metrics": final, "comparison": tab.reset_index().to_dict("records"), "log": [msg]}

# ---------- Agent 3: Explainer ----------
@safe
def explainer(state):
    b = T.load_bundle()
    try: X, assumptions = T.patient_to_row(b, state["patient"])
    except ValueError as e: return {"error": f"invalid patient data: {e}", "log": [f"[Explainer] rejected input: {e}"]}
    p = float(T.risk(b, X)[0]); thr = b["threshold"]
    sh, li = T.shap_local(b, X), T.lime_local(b, X); ag = T.agreement(sh, li)
    wi = T.what_if(b, X, p)
    vals = {c: T.decode(b, c, X[c].iloc[0]) for c in b["columns"]}
    top = [{"feature": f, "value": vals[f], "shap": round(float(sh[f]), 4), "lime": round(float(li[f]), 4)} for f in sh.abs().nlargest(5).index]
    msg = f"[Explainer] risk {p:.3f} (threshold {thr:.2f}); SHAP–LIME Jaccard {ag['jaccard']}, sign agreement {ag['sign_agreement']}, conflicts {ag['conflict_features']}"
    return {"risk": p, "risk_label": "high" if p >= thr else "lower", "threshold": thr, "top": top, "agreement": ag,
            "what_if": wi[:5], "assumptions": assumptions, "log": [msg]}

# ---------- Agent 4: Evidence Writer + Checker ----------
ALIAS = {"BMI": "body mass index obesity", "APOE-e4 Genetic Risk": "APOE e4 allele genetic", "Country": "regional differences prevalence country income settings",
         "Family History": "family history parent sibling", "Urban vs Rural Living": "urban rural living", "Stress Levels": "chronic psychological stress",
         "Income Level": "income socioeconomic position", "Employment Status": "employment socioeconomic position", "Education Level": "education cognitive reserve"}
def _q(f): return f"{ALIAS.get(f, f)} Alzheimer's dementia risk"

def _evidence(state):
    r = retriever(); ev = {}
    for t in state["top"]:
        hits = [h for h in r.search(_q(t["feature"]), k=8, min_cos=0.03)]
        hits = [h for h in hits if t["feature"] in h["features"]] or hits[:0]          # feature-tagged evidence only; otherwise abstain
        ev[t["feature"]] = [h["id"] for h in hits[:2]]
    return ev

def _draft(state, strict=False):
    r = retriever(); ev = _evidence(state); ag = state["agreement"]; claims, lines = [], []
    for t in state["top"]:
        ids = ev[t["feature"]]; direction = "raises" if t["shap"] > 0 else "lowers"
        conflict = t["feature"] in ag["conflict_features"]
        claim = {"feature": t["feature"], "value": str(t["value"]), "direction": direction, "evidence": ids, "conflict": conflict, "abstained": not ids}
        if strict and not ids: continue
        claims.append(claim)
        cite = " ".join(f"[{i}]" for i in ids) if ids else "(no supporting evidence in the knowledge base; stated as model output only)"
        note = " The two explanation methods disagree on this feature, so treat it with caution [KB-XAI]." if conflict else ""
        lines.append(f"- **{t['feature']} = {t['value']}** {direction} the model's risk estimate {cite}.{note}")
    advice = []
    for w in state.get("what_if", []):
        if w["delta_risk"] < -0.005 and w["feature"] in C.MODIFIABLE:
            ids = r.search(_q(w["feature"]), k=8, min_cos=0.03)
            ids = [h["id"] for h in ids if w["feature"] in h["features"]][:1]
            if ids or not strict: advice.append({"feature": w["feature"], "from": str(w["from"]), "to": str(w["to"]), "delta": w["delta_risk"], "evidence": ids})
    txt = [f"**Estimated risk: {state['risk']:.1%}** (model-based; flag threshold {state['threshold']:.0%}; category: {state['risk_label']}).", "", "**Main contributing factors**"] + lines
    if advice:
        txt += ["", "**Modifiable factors where the model's estimate changes** (association in this dataset, not a causal effect)"]
        for a in advice:
            cite = " ".join(f"[{i}]" for i in a["evidence"]); txt.append(f"- {a['feature']}: {a['from']} → {a['to']} changes the estimate by {a['delta']:+.1%} {cite}".rstrip())
    prev = M.last(state["patient_id"]) if state.get("patient_id") else None
    if prev:
        txt += ["", f"**Change since last assessment:** {prev['risk']:.1%} → {state['risk']:.1%} ({state['risk'] - prev['risk']:+.1%})."]
    if state.get("assumptions"): txt += ["", f"_Assumptions: {len(state['assumptions'])} field(s) were missing and filled with training medians/modes._"]
    txt += ["", "_Limitations: population-data model; associations, not diagnosis or causation [KB-LIM]. Consult a clinician for any health decision._"]
    return "\n".join(txt), {"claims": claims, "advice": advice, "conflicts": ag["conflict_features"]}

def _llm_rewrite(draft):
    import anthropic
    msg = anthropic.Anthropic().messages.create(model=C.LLM_MODEL, max_tokens=700,
        system="Rewrite the report in plain, calm language for a non-expert. Use ONLY facts in the draft, keep every [KBxxx] citation exactly, "
               "do not add numbers, causes, diagnoses, or advice.", messages=[{"role": "user", "content": draft}])
    return msg.content[0].text

@safe
def evidence_writer(state):
    strict = state.get("retries", 0) > 0
    text, struct = _draft(state, strict=strict); log = []
    if C.USE_LLM and not strict:
        try: text = _llm_rewrite(text); log.append("[Writer] LLM rewrite applied")
        except Exception as e: log.append(f"[Writer] LLM rewrite failed ({type(e).__name__}); using template draft")
    # ---- checker ----
    kb_ids = set(retriever().by_id); problems = []
    cited = set(re.findall(r"\[(KB[\w\-]*)\]", text))
    allowed = {i for c in struct["claims"] for i in c["evidence"]} | {i for a in struct["advice"] for i in a["evidence"]} | {"KB-XAI", "KB-LIM"}
    if cited - kb_ids: problems.append(f"citation to unknown id: {sorted(cited - kb_ids)}")
    if cited - allowed: problems.append(f"citation not backed by retrieved evidence: {sorted(cited - allowed)}")
    for a in struct["advice"]:
        if a["feature"] not in C.MODIFIABLE: problems.append(f"advice on non-modifiable feature {a['feature']}")
    for f in struct["conflicts"]:
        if f in {c["feature"] for c in struct["claims"]} and "disagree" not in text: problems.append(f"conflict on {f} not disclosed")
    if "[KB-LIM]" not in text: problems.append("limitations statement missing")
    unsupported = [c["feature"] for c in struct["claims"] if c["abstained"]]
    check = {"passed": not problems, "problems": problems, "n_claims": len(struct["claims"]), "abstained_features": unsupported,
             "citation_validity": 1.0 if not (cited - kb_ids) else 0.0}
    log.append(f"[Writer/Checker] claims={check['n_claims']} abstained={len(unsupported)} passed={check['passed']} {problems or ''}")
    upd = {"report": text, "claims": struct, "check": check, "retries": state.get("retries", 0) + (0 if check["passed"] else 1), "log": log}
    if check["passed"] and state.get("patient_id"):
        M.save_assessment(state["patient_id"], state["risk"], state["risk_label"], [t["feature"] for t in state["top"]], text)
    return upd
