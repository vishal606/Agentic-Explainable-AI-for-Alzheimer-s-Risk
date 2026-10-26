# %% [markdown]
# # Multi-agent Alzheimer's risk system (LangGraph): demo
# Four agents (Auditor, Predictor, Explainer, Evidence Writer + Checker) coordinated by a supervisor, with checkpoint memory, long-term memory and a FastAPI service.
# Run from the folder that contains `alz_agents/` and the dataset CSV. `FAST_MODE=1` is a quick test; use 0 for reportable numbers.

# %%
import os; os.environ.setdefault("FAST_MODE", "1")
from alz_agents import graph as G, memory as M
from IPython.display import Markdown, display
app = G.get_app()

# %% [markdown]
# ## Agent graph

# %%
print(G.build_graph().get_graph().draw_mermaid())

# %% [markdown]
# ## 1. Train (Auditor -> Predictor)

# %%
s = G.run_train(app); print("\n".join(s["log"]))

# %% [markdown]
# ## 2. Assess a patient (Explainer -> Writer/Checker). Edit the fields freely.

# %%
patient = {"Country":"Spain","Age":78,"Gender":"Female","Education Level":6,"BMI":31.0,"Physical Activity Level":"Low","Smoking Status":"Former",
 "Alcohol Consumption":"Regularly","Diabetes":"Yes","Hypertension":"Yes","Cholesterol Level":"High","Family History":"Yes","Cognitive Test Score":55,
 "Depression Level":"High","Sleep Quality":"Poor","Dietary Habits":"Unhealthy","Air Pollution Exposure":"High","Employment Status":"Retired",
 "Marital Status":"Widowed","APOE-e4 Genetic Risk":"Yes","Social Engagement Level":"Low","Income Level":"Low","Stress Levels":"High","Urban vs Rural Living":"Urban"}
r = G.run_assess("P001", patient, app); print("\n".join(r["log"])); display(Markdown(r["report"])); print(r["check"])

# %% [markdown]
# ## 3. Memory: reassess the same patient later

# %%
r2 = G.run_assess("P001", dict(patient, Age=79, **{"Physical Activity Level": "High"}), app)
display(Markdown(r2["report"])); M.history("P001")

# %% [markdown]
# ## 4. Input validation: out-of-range data is rejected by the Explainer

# %%
print(G.run_assess("P002", dict(patient, Age=300), app)["error"])

# %% [markdown]
# ## 5. Serve it as an API
# ```
# uvicorn alz_agents.api:app --port 8000
# curl -X POST localhost:8000/assess -H 'content-type: application/json' -d '{"patient_id":"P1","patient":{...}}'
# ```
