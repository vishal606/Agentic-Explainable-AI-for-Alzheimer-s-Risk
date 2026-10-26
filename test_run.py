import os, json, time
os.environ.setdefault("FAST_MODE", "1")
from alz_agents import graph as G, memory as M
t=time.time(); app = G.get_app()
s = G.run_train(app); print("\n".join(s["log"])); print(f"train {time.time()-t:.0f}s")
patient = {"Country":"Spain","Age":78,"Gender":"Female","Education Level":6,"BMI":31.0,"Physical Activity Level":"Low","Smoking Status":"Former",
 "Alcohol Consumption":"Regularly","Diabetes":"Yes","Hypertension":"Yes","Cholesterol Level":"High","Family History":"Yes","Cognitive Test Score":55,
 "Depression Level":"High","Sleep Quality":"Poor","Dietary Habits":"Unhealthy","Air Pollution Exposure":"High","Employment Status":"Retired",
 "Marital Status":"Widowed","APOE-e4 Genetic Risk":"Yes","Social Engagement Level":"Low","Income Level":"Low","Stress Levels":"High","Urban vs Rural Living":"Urban"}
t=time.time(); r = G.run_assess("P001", patient, app); print("\n".join(r["log"])); print(r["report"]); print("check", r["check"], f"{time.time()-t:.0f}s")
p2 = dict(patient, **{"Physical Activity Level":"High","Age":79}); r2 = G.run_assess("P001", p2, app); print(r2["report"].split("**Change")[-1][:120]); print(M.history("P001"))
bad = G.run_assess("P002", dict(patient, Age=300), app); print("bad:", bad.get("error"))
