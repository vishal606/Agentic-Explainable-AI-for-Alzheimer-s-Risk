"""Knowledge base + hybrid retriever (BM25 + TF-IDF fused by reciprocal rank fusion, with a relevance gate).
STARTER set: paraphrased statements. VERIFY each against its source before publishing; add real documents to kb_docs/."""
import os, re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STARTER_KB = [
 ("KB001", "WHO dementia fact sheet", "Dementia burden", "WHO estimates that more than 57 million people were living with dementia in 2021, and Alzheimer's disease may account for roughly 60-70% of cases.", []),
 ("KB002", "Alzheimer's Association; Lancet Commission", "Age", "Age is the strongest known risk factor for Alzheimer's disease, and risk rises steeply after about 65 years and continues to increase in older age groups.", ["Age"]),
 ("KB003", "Alzheimer's Association; genetics literature", "APOE e4 allele", "The APOE e4 allele is the strongest common genetic risk factor for late-onset Alzheimer's disease; carrying one or two copies raises risk, but carriers do not always develop the disease and non-carriers can.", ["APOE-e4 Genetic Risk"]),
 ("KB004", "Alzheimer's Association", "Family history", "Having a parent or sibling with Alzheimer's disease increases an individual's risk, reflecting shared genetics and possibly shared environment.", ["Family History"]),
 ("KB005", "Alzheimer's Association", "Sex differences", "More women than men are living with Alzheimer's disease, partly because women live longer on average; research on sex-specific biological factors is ongoing.", ["Gender"]),
 ("KB006", "Lancet Commission on dementia (2024)", "Modifiable risk factors", "The 2024 Lancet Commission identified 14 potentially modifiable risk factors that together may account for about 45% of dementia cases worldwide, including less education, high LDL cholesterol, depression, physical inactivity, diabetes, smoking, hypertension, obesity, excessive alcohol use, social isolation and air pollution.", ["Education Level", "Cholesterol Level", "Depression Level", "Physical Activity Level", "Diabetes", "Smoking Status", "Hypertension", "BMI", "Alcohol Consumption", "Social Engagement Level", "Air Pollution Exposure"]),
 ("KB007", "Lancet Commission on dementia (2024)", "Physical activity", "Regular physical activity is associated with a lower risk of cognitive decline and dementia in observational studies, and physical inactivity is a modifiable risk factor.", ["Physical Activity Level"]),
 ("KB008", "Lancet Commission on dementia (2024)", "Hypertension", "Hypertension, especially in midlife, is associated with a higher risk of dementia, and blood pressure control is considered an important preventive measure.", ["Hypertension"]),
 ("KB009", "Lancet Commission on dementia (2024)", "Diabetes", "Diabetes is associated with an increased risk of dementia, and managing diabetes and other vascular risk factors is recommended for brain health.", ["Diabetes"]),
 ("KB010", "Lancet Commission on dementia (2024)", "Cholesterol", "High LDL cholesterol, particularly in midlife, is listed among the modifiable risk factors for dementia.", ["Cholesterol Level"]),
 ("KB011", "Lancet Commission on dementia (2024)", "Smoking", "Smoking is associated with a higher risk of dementia, and former smokers generally have lower risk than current smokers, although evidence on the timing of benefit is limited.", ["Smoking Status"]),
 ("KB012", "Lancet Commission on dementia (2024)", "Alcohol", "Heavy or excessive alcohol consumption is associated with an increased risk of dementia.", ["Alcohol Consumption"]),
 ("KB013", "Lancet Commission on dementia (2024)", "Depression", "Depression is associated with a higher risk of dementia and may be both a risk factor and an early symptom, so the relationship is complex.", ["Depression Level"]),
 ("KB014", "Sleep and dementia literature", "Sleep quality", "Poor sleep quality and sleep disturbances are associated with cognitive decline and Alzheimer's pathology in observational research, and the direction of causation is still being studied.", ["Sleep Quality"]),
 ("KB015", "Nutrition and dementia literature", "Diet", "Dietary patterns such as the Mediterranean and MIND diets are associated with slower cognitive decline in observational studies, while evidence from randomised trials is mixed.", ["Dietary Habits"]),
 ("KB016", "Lancet Commission on dementia (2024)", "Social engagement", "Social isolation and low social contact are associated with a higher risk of dementia, and staying socially and cognitively engaged is considered beneficial.", ["Social Engagement Level"]),
 ("KB017", "Lancet Commission on dementia (2024)", "Air pollution", "Long-term exposure to air pollution, particularly fine particulate matter, is associated with a higher risk of dementia and is a modifiable risk factor.", ["Air Pollution Exposure"]),
 ("KB018", "Lancet Commission on dementia (2024)", "Education", "Lower levels of education in early life are associated with a higher risk of dementia, consistent with the cognitive reserve hypothesis.", ["Education Level"]),
 ("KB019", "Lancet Commission on dementia (2024)", "Body mass index and obesity", "Obesity in midlife is associated with a higher risk of dementia, while the relationship in late life is more complex because weight often falls before diagnosis.", ["BMI"]),
 ("KB020", "Clinical cognitive screening literature", "Cognitive test score", "Cognitive screening tests provide a snapshot of cognitive function; low scores can indicate impairment but are not diagnostic on their own, and results vary with education and language.", ["Cognitive Test Score"]),
 ("KB021", "Stress and cognition literature", "Stress", "Chronic psychological stress has been linked to cognitive decline in some studies, although evidence for a causal role in Alzheimer's disease is limited.", ["Stress Levels"]),
 ("KB022", "Social determinants of dementia literature", "Income and socioeconomic position", "Lower socioeconomic position is associated with a higher risk of dementia in many populations, likely reflecting differences in education, access to health care and vascular risk factors.", ["Income Level", "Employment Status"]),
 ("KB023", "Social determinants of dementia literature", "Marital status", "Being unmarried or widowed has been associated with a higher risk of dementia in some cohort studies, possibly through social isolation and lifestyle factors.", ["Marital Status"]),
 ("KB024", "Epidemiological literature", "Urban and rural living", "Some studies report differences in dementia prevalence between rural and urban settings, which may reflect education, access to care and environmental factors, and findings vary by country.", ["Urban vs Rural Living"]),
 ("KB025", "WHO; Alzheimer's Disease International", "Regional differences", "Dementia prevalence and risk-factor profiles differ across regions and income settings, and a majority of people with dementia live in low- and middle-income countries.", ["Country"]),
 ("KB026", "Lancet Commission on dementia (2024)", "Prevention", "Reducing modifiable risk factors across the life course may prevent or delay a substantial share of dementia cases, although there is currently no cure for Alzheimer's disease.", []),
 ("KB-XAI", "Authors' method note", "Explanation disagreement", "When two explanation methods disagree on a feature, its attribution is unreliable and should be interpreted with caution rather than treated as established.", []),
 ("KB-LIM", "Authors' method note", "Limitations", "Model-based risk estimates from population datasets describe statistical associations, are not diagnoses and do not establish causation; clinical assessment is required for any diagnosis.", []),
]

def _tok(s): return re.findall(r"[a-z0-9\-]+", s.lower())

def load_user_docs(folder, words=120):
    out = []
    if not os.path.isdir(folder): return out
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn); txt = ""
        if fn.lower().endswith((".txt", ".md")): txt = open(p, encoding="utf-8", errors="ignore").read()
        elif fn.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                txt = "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
            except Exception as e: print("skip pdf:", fn, e)
        w = txt.split()
        for k in range(0, len(w), words):
            out.append({"id": f"{os.path.splitext(fn)[0][:12]}-{k // words:03d}", "source": fn, "title": fn,
                        "text": " ".join(w[k:k + words]), "features": [], "starter": False})
    return out

class HybridRetriever:
    def __init__(self, kb_dir="kb_docs"):
        self.chunks = [{"id": i, "source": s, "title": t, "text": x, "features": f, "starter": True} for i, s, t, x, f in STARTER_KB]
        self.chunks += load_user_docs(kb_dir)
        self.by_id = {c["id"]: c for c in self.chunks}
        self.texts = [c["title"] + ". " + c["text"] for c in self.chunks]
        self.bm25 = BM25Okapi([_tok(t) for t in self.texts])
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.M = self.tfidf.fit_transform(self.texts)

    def search(self, query, k=3, min_cos=0.05):
        b = self.bm25.get_scores(_tok(query)); d = cosine_similarity(self.tfidf.transform([query]), self.M)[0]
        rb, rd = np.argsort(-b).argsort(), np.argsort(-d).argsort()
        score = 1 / (60 + rb) + 1 / (60 + rd)
        return [{**self.chunks[j], "score": float(score[j]), "cos": float(d[j])} for j in np.argsort(-score)[:k] if d[j] >= min_cos]
