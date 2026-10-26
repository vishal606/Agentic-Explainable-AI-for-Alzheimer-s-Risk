"""Long-term memory (SQLite). Short-term/thread memory is the LangGraph SqliteSaver checkpointer in graph.py."""
import sqlite3, json, time
from . import config as C

def _conn():
    c = sqlite3.connect(C.DB_PATH, check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS assessments(id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, ts REAL, risk REAL,
                 label TEXT, top_features TEXT, report TEXT)""")
    return c

def save_assessment(patient_id, risk, label, top_features, report):
    c = _conn(); c.execute("INSERT INTO assessments(patient_id, ts, risk, label, top_features, report) VALUES(?,?,?,?,?,?)",
                           (patient_id, time.time(), float(risk), label, json.dumps(top_features), report)); c.commit(); c.close()

def history(patient_id, limit=10):
    c = _conn(); rows = c.execute("SELECT ts, risk, label, top_features FROM assessments WHERE patient_id=? ORDER BY ts DESC LIMIT ?",
                                  (patient_id, limit)).fetchall(); c.close()
    return [{"ts": r[0], "risk": r[1], "label": r[2], "top_features": json.loads(r[3])} for r in rows]

def last(patient_id):
    h = history(patient_id, 1); return h[0] if h else None
