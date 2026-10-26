import os
ROOT = os.environ.get("ALZ_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(ROOT, "alzheimers_prediction_dataset.csv"))
ART_DIR = os.environ.get("ART_DIR", os.path.join(ROOT, "artifacts"))
KB_DIR = os.environ.get("KB_DIR", os.path.join(ROOT, "kb_docs"))
DB_PATH = os.path.join(ART_DIR, "memory.sqlite")          # long-term memory + LangGraph checkpoints
SEED = int(os.environ.get("SEED", "42"))
FAST = os.environ.get("FAST_MODE", "0") == "1"             # 1 = subsample for quick tests
USE_LLM = os.environ.get("USE_LLM", "0") == "1"            # optional: Claude rewrites the draft (needs ANTHROPIC_API_KEY)
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
os.makedirs(ART_DIR, exist_ok=True)

TARGET = "Diagnosis"
RENAME = {"Family History of Alzheimer’s": "Family History", "Genetic Risk Factor (APOE-ε4 allele)": "APOE-e4 Genetic Risk",
          "Alzheimer’s Diagnosis": TARGET}
# modifiability groups (association only, NOT causal advice)
MODIFIABLE = ["Physical Activity Level", "Smoking Status", "Alcohol Consumption", "Diabetes", "Hypertension", "Cholesterol Level",
              "Depression Level", "Sleep Quality", "Dietary Habits", "Air Pollution Exposure", "Social Engagement Level", "BMI", "Stress Levels"]
NON_MODIFIABLE = ["Age", "Gender", "Family History", "APOE-e4 Genetic Risk", "Country", "Education Level"]
