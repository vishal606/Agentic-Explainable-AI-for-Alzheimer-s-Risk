"""FastAPI service: uvicorn alz_agents.api:app --port 8000"""
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import graph as G, memory as M, tools as T

app = FastAPI(title="Alzheimer's risk multi-agent system", version="1.0")
_graph = None
def graph():
    global _graph
    if _graph is None: _graph = G.get_app()
    return _graph

class AssessRequest(BaseModel):
    patient_id: str
    patient: Dict[str, Any]

@app.get("/health")
def health(): return {"status": "ok", "model_ready": T.load_bundle() is not None}

@app.post("/train")
def train():
    s = G.run_train(graph()); return {"log": s["log"], "metrics": s.get("train_metrics"), "audit": s.get("audit")}

@app.post("/assess")
def assess(req: AssessRequest):
    s = G.run_assess(req.patient_id, req.patient, graph())
    if s.get("error"): raise HTTPException(422, s["error"])
    if not s.get("report"): raise HTTPException(409, {"message": "pipeline stopped", "log": s.get("log")})
    return {"risk": s["risk"], "category": s["risk_label"], "report": s["report"], "check": s["check"], "agreement": s["agreement"], "log": s["log"]}

@app.get("/history/{patient_id}")
def hist(patient_id: str): return M.history(patient_id)
