# Alzheimer's risk: multi-agent system (LangGraph)

```
START -> supervisor -> {auditor | predictor | explainer | writer} -> supervisor -> ... -> END
```
| Agent | Job |
|---|---|
| Data Auditor | quality, duplicates, class balance, leakage screen; blocks the run on errors |
| Predictor | logistic regression vs LightGBM, validation-based selection, Platt calibration, validation threshold, leave-one-country-out AUC |
| Explainer | local SHAP + LIME, agreement score and conflicts, one-at-a-time what-if on modifiable features |
| Evidence Writer + Checker | hybrid retrieval (BM25 + TF-IDF) with abstention, cited report, then checks citations/advice rules; failed check sends the supervisor back to a strict rewrite |

**Memory**: (1) LangGraph `SqliteSaver` checkpointer, one thread per patient (`artifacts/checkpoints.sqlite`); (2) long-term SQLite store of past assessments (`artifacts/memory.sqlite`) used for "change since last assessment" and the `/history` endpoint.
**APIs**: FastAPI (`/health`, `/train`, `/assess`, `/history/{id}`). Optional Claude rewrite with `USE_LLM=1` + `ANTHROPIC_API_KEY` (the checker still validates the output).

## Run
```
pip install -r requirements.txt
# put alzheimers_prediction_dataset.csv next to this file (or set DATA_PATH)
python -c "from alz_agents import graph as G; print(G.run_train()['log'])"
uvicorn alz_agents.api:app --port 8000        # then POST /assess  {"patient_id": "P1", "patient": {...all 23 fields...}}
python test_api.py                            # smoke test
```
`FAST_MODE=1` subsamples for quick tests. Knowledge base = 28 paraphrased starter statements in `alz_agents/kb.py`: verify them, and add real documents to `kb_docs/`.

## Limits
Dataset appears simulated (only Age, APOE-e4, Family History, Country carry signal; AUC ceiling ≈0.80). Output is a model-based estimate, not a diagnosis; what-if results are associations, not causal. Supervisor is rule-based by design (auditable). The LLM rewrite path is only exercised with a stub in tests.
