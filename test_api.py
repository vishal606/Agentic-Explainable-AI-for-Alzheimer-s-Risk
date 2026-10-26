import os; os.environ["FAST_MODE"]="1"
from fastapi.testclient import TestClient
from alz_agents import api, agents, config as C
from sample_patient import patient
c = TestClient(api.app)
print(c.get("/health").json())
r = c.post("/assess", json={"patient_id":"A1","patient":patient}); print(r.status_code, r.json()["category"], r.json()["check"]["passed"])
print("bad category ->", c.post("/assess", json={"patient_id":"A2","patient":dict(patient, Country="Mars")}).status_code)
print("history rows:", len(c.get("/history/A1").json()))
C.USE_LLM = True; agents._llm_rewrite = lambda d: d + "\nThis is supported by [KB999]."     # simulate a hallucinated citation
r = c.post("/assess", json={"patient_id":"A3","patient":patient}).json()
print([l for l in r["log"] if "Writer" in l or "Supervisor" in l]); print("final passed:", r["check"]["passed"], "| KB999 in report:", "KB999" in r["report"])
