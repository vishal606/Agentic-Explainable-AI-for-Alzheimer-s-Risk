"""LangGraph wiring: supervisor routes between the four agents; SqliteSaver gives per-thread memory."""
import operator, os, sqlite3, json
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from . import config as C, tools as T
from .agents import data_auditor, predictor, explainer, evidence_writer, AUDIT_PATH

def _log_reducer(a, b):
    """Append log lines; a leading '__reset__' starts a fresh log for a new run on the same thread."""
    if b and b[0] == "__reset__": return b[1:]
    return (a or []) + b

class State(TypedDict, total=False):
    task: str; patient_id: str; patient: dict; next: str; error: str; retries: int
    audit: dict; train_metrics: dict; comparison: list
    risk: float; risk_label: str; threshold: float; top: list; agreement: dict; what_if: list; assumptions: list
    report: str; claims: dict; check: dict
    log: Annotated[list, _log_reducer]

def _fresh_audit():
    if not os.path.exists(AUDIT_PATH): return False
    a = json.load(open(AUDIT_PATH)); return a.get("data_hash") == T.file_hash() and a.get("ok", False)

def _fresh_model():
    b = T.load_bundle(); return b is not None and b["data_hash"] == T.file_hash()

def supervisor(state: State):
    """Rule-based router (deterministic, auditable). Returns the next agent name."""
    if state.get("error"): return {"next": "END", "log": ["[Supervisor] -> END (error)"]}
    ran = state.get("log", [])
    did = lambda tag: any(l.startswith(tag) for l in ran)
    if state.get("task") == "train":
        if not did("[Auditor]"): nxt = "auditor"
        elif (state.get("audit") or {}).get("ok") is False: nxt = "END"
        elif not did("[Predictor]"): nxt = "predictor"
        else: nxt = "END"
    else:                                                    # assess
        if not _fresh_audit() and not did("[Auditor]"): nxt = "auditor"
        elif (state.get("audit") or {}).get("ok") is False: nxt = "END"
        elif not _fresh_model() and not did("[Predictor]"): nxt = "predictor"
        elif state.get("risk") is None: nxt = "explainer"
        elif state.get("check") is None: nxt = "writer"
        elif not state["check"]["passed"] and state.get("retries", 0) < 2: nxt = "writer"     # strict rewrite once
        else: nxt = "END"
    return {"next": nxt, "log": [f"[Supervisor] -> {nxt}"]}

def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("supervisor", supervisor); g.add_node("auditor", data_auditor); g.add_node("predictor", predictor)
    g.add_node("explainer", explainer); g.add_node("writer", evidence_writer)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], {"auditor": "auditor", "predictor": "predictor", "explainer": "explainer", "writer": "writer", "END": END})
    for n in ["auditor", "predictor", "explainer", "writer"]: g.add_edge(n, "supervisor")
    return g.compile(checkpointer=checkpointer)

def get_app():
    conn = sqlite3.connect(os.path.join(C.ART_DIR, "checkpoints.sqlite"), check_same_thread=False)
    return build_graph(SqliteSaver(conn))

def run_train(app=None):
    app = app or get_app(); return app.invoke({"task": "train", "log": ["__reset__"], "audit": None, "error": None}, {"configurable": {"thread_id": "system-train"}})

def run_assess(patient_id, patient, app=None):
    app = app or get_app()
    return app.invoke({"task": "assess", "patient_id": patient_id, "patient": patient, "retries": 0, "log": ["__reset__"],
                       "error": None, "audit": None, "risk": None, "check": None, "report": None},
                      {"configurable": {"thread_id": f"patient-{patient_id}"}, "recursion_limit": 25})
