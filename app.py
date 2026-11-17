import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

load_dotenv()
st.set_page_config(page_title="AI Triage Queue", page_icon="🩺", layout="wide")

DISCLAIMER = ("⚠️ Decision-support only. This tool does not diagnose. "
              "All flags require review by a qualified healthcare professional.")


@st.cache_resource
def train():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = 1 - data.target  # 1 = suspicious (malignant), 0 = benign
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, X_tr, X_te, y_te, auc


model, X_tr, X_te, y_te, auc = train()
FEATURES = list(X_tr.columns)
MEAN, STD = X_tr.mean(), X_tr.std().replace(0, 1)
IMPORTANCE = pd.Series(model.feature_importances_, index=FEATURES)


def tier(p):
    if p >= 0.70:
        return "🔴 High"
    if p >= 0.40:
        return "🟡 Medium"
    return "🟢 Low"


def score(df):
    out = pd.DataFrame(index=df.index)
    out["Risk %"] = (model.predict_proba(df[FEATURES])[:, 1] * 100).round(1)
    out["Priority"] = out["Risk %"].apply(lambda r: tier(r / 100))
    return out.sort_values("Risk %", ascending=False)


def top_factors(row, n=3):
    z = (row[FEATURES] - MEAN) / STD
    contrib = (z * IMPORTANCE).sort_values(ascending=False).head(n)
    return [(f, float(row[f]), float(z[f])) for f in contrib.index]


def get_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return key


def fallback_summary(risk, factors):
    names = ", ".join(f.replace("_", " ") for f, _, _ in factors)
    return (f"Model estimates a {risk:.0f}% likelihood of a suspicious finding. "
            f"Main contributing measurements: {names}. "
            "Recommend clinician review and confirmatory assessment. "
            "This is not a diagnosis.")


def ai_summary(risk, factors):
    key = get_key()
    if not key:
        return fallback_summary(risk, factors), "template (no API key)"
    facts = "; ".join(f"{f}={v:.2f} ({z:+.1f} SD from average)" for f, v, z in factors)
    prompt = (
        "You assist a physician with case triage. A screening model estimated "
        f"a {risk:.0f}% likelihood of a suspicious finding. Top contributing "
        f"measurements: {facts}. Write exactly 3 short sentences: what stands "
        "out, what a clinician should review next, and one sentence on "
        "uncertainty. Do NOT diagnose or prescribe."
    )
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return resp.text.strip(), "Gemini"
    except Exception:
        return fallback_summary(risk, factors), "template (API unavailable)"


st.title("🩺 AI Triage Queue")
st.caption("Clinical decision-support for prioritizing cases for further assessment")
st.warning(DISCLAIMER)

tab1, tab2 = st.tabs(["Triage Queue", "How it works"])

with tab1:
    src = st.radio("Case source", ["Sample held-out cases", "Upload CSV"], horizontal=True)
    if src == "Sample held-out cases":
        n = st.slider("Number of cases", 10, 100, 40)
        cases = X_te.sample(n, random_state=7)
    else:
        up = st.file_uploader("CSV with the 30 feature columns (see How it works)", type="csv")
        cases = None
        if up:
            raw = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in raw.columns]
            if missing:
                st.error(f"Missing columns: {missing[:5]}...")
            else:
                cases = raw
        if cases is None:
            st.info("Upload a CSV, or switch to sample cases.")
            st.stop()

    cases = cases.copy()
    if src == "Upload CSV":
        cases.index = [f"Case-{i:03d}" for i in range(1, len(cases) + 1)]
    else:
        cases.index = [f"Case-{i}" for i in cases.index]
    queue = score(cases)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High priority", int((queue["Priority"] == "🔴 High").sum()))
    c2.metric("🟡 Medium", int((queue["Priority"] == "🟡 Medium").sum()))
    c3.metric("🟢 Low", int((queue["Priority"] == "🟢 Low").sum()))

    st.subheader("Prioritized queue (highest risk first)")
    st.dataframe(
        queue,
        use_container_width=True,
        column_config={"Risk %": st.column_config.ProgressColumn(
            "Risk %", min_value=0, max_value=100, format="%.1f")},
    )

    st.subheader("Case review")
    pick = st.selectbox("Select a case", queue.index)
    risk = float(queue.loc[pick, "Risk %"])
    factors = top_factors(cases.loc[pick])
    st.markdown(f"**{pick}** — {queue.loc[pick, 'Priority']} priority · estimated risk **{risk:.1f}%**")
    st.markdown("**Top factors driving this flag:**")
    for f, v, z in factors:
        st.write(f"- {f.replace('_', ' ')}: {v:.2f} ({z:+.1f} SD vs average)")

    if st.button("Generate clinical summary"):
        with st.spinner("Generating..."):
            text, source = ai_summary(risk, factors)
        st.info(text)
        st.caption(f"Source: {source}. {DISCLAIMER}")

with tab2:
    st.markdown(f"""
**Pipeline:** clinical measurements → Random Forest risk score → triage tier
(High ≥70%, Medium 40–70%, Low <40%) → ranked queue → explainable factors
→ LLM-written clinician summary (with safe template fallback).

**Dataset:** Breast Cancer Wisconsin (Diagnostic), public and de-identified,
via scikit-learn. Queue cases come from a **held-out test set** the model never trained on.

**Held-out ROC-AUC:** `{auc:.3f}`

**Explainability:** each case's top factors are the features that deviate most
from the training average, weighted by model feature importance.

**CSV upload format:** columns must match:
`{", ".join(FEATURES)}`

**Limitations:** demo model; not clinically validated; not for real patient care.
""")
