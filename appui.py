import joblib
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import streamlit as st


import traceback
import sys

try:
    xgb_pipeline = joblib.load("gpa_model.pkl")
    stacking_model = joblib.load("burnout_model.pkl")
    scaler = joblib.load("scaler.pkl")
    cntr = joblib.load("fuzzy_centers.pkl")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()




def recommend(cluster, burnout):

    match (cluster, burnout):

        # Cluster 0 – High AI Dependency Learners
        case (0, "Low"):
            return (
                "You're using AI effectively without feeling overwhelmed — that's a good sign. "
                "To stay on track, start practicing problems independently before turning to AI, "
                "so you build confidence in your own thinking."
            )

        case (0, "Medium"):
            return (
                "Your heavy AI usage may be adding hidden pressure. "
                "Try setting boundaries: use AI to check your work, not replace it. "
                "Dedicate at least 30 minutes daily to solving problems on your own first."
            )

        case (0, "High"):
            return (
                "High AI dependency combined with high stress is a warning sign. "
                "Step back from over-relying on AI for answers — it may be weakening your ability to think under exam pressure. "
                "Focus on rebuilding independent study habits and seek academic support if needed."
            )

        # Cluster 1 – Traditional Learners
        case (1, "Low"):
            return (
                "Your traditional study habits are paying off. "
                "Consider selectively integrating AI tools — not to replace your methods, "
                "but to save time on research and summarizing, so you can focus energy on deeper understanding."
            )

        case (1, "Medium"):
            return (
                "You're putting in strong effort the traditional way, but stress is building. "
                "This might mean you're overloading yourself. "
                "Try using AI tools to speed up routine tasks like note summarizing, "
                "and protect more time for rest and review."
            )

        case (1, "High"):
            return (
                "You're studying hard but burning out. Traditional methods alone can be exhausting "
                "when the workload is heavy. Consider using AI as a study assistant to reduce effort "
                "on low-value tasks, and prioritize sleep and scheduled breaks — consistency matters more than hours."
            )

        # Cluster 2 – Stable Low-Stress Traditional Learners
        case (2, "Low"):
            return (
                "You have one of the healthiest study patterns — balanced, consistent, and low stress. "
                "Keep this rhythm and consider exploring AI tools gradually "
                "to enhance your efficiency without disrupting what's already working."
            )

        case (2, "Medium"):
            return (
                "Your stable habits are solid, but something is starting to add pressure. "
                "Review your recent schedule for overcommitments and make sure "
                "you're maintaining the balance that made your pattern successful."
            )

        case (2, "High"):
            return (
                "Stress is rising despite your stable routine — this could be external pressure "
                "like exams or deadlines piling up. Don't break your good habits under pressure. "
                "Focus on stress management techniques alongside your study routine."
            )

        case _:
            return "No recommendation available for this combination."

st.title("Student Performance & Burnout Prediction System")

def skill_retention_score(genai_hours, study_hours, prompt_skill, anxiety):
    
    skill_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    skill_num = skill_map[prompt_skill]
    
    genai_ratio = genai_hours / (genai_hours + study_hours)
    
    score = (57.58
             + 32.05 * genai_ratio
             + 8.23  * skill_num
             - 2.02  * anxiety)
    
    return round(min(max(score, 0), 100), 2)

major = st.selectbox("Major Category",
    ["STEM","Business","Humanities","Medical","Arts"])

year = st.selectbox("Year of Study",
    ["Freshman","Sophomore","Junior","Senior","Graduate"])

gpa_input = st.number_input("Pre-Semester GPA", 0.0, 4.0,format="%.6f")

genai_level = st.selectbox(
    "Weekly GenAI Usage Level",
    [
        "Very Low (0–5)",
        "Low (5–10)",
        "Moderate (10–20)",
        "High (20–30)",
        "Very High (30–40)"
    ]
)
genai_map = {
    "Very Low (0–5)": 2,
    "Low (5–10)": 7,
    "Moderate (10–20)": 15,
    "High (20–30)": 25,
    "Very High (30–40)": 35
}

genai_hours = genai_map[genai_level]

study_level = st.selectbox(
    "Traditional Study Level",
    [
        "Very Low (1–5)",
        "Low (5–10)",
        "Moderate (10–20)",
        "High (20–30)",
        "Very High (30–36)"
    ]
)
study_map = {
    "Very Low (1–5)": 3,
    "Low (5–10)": 7,
    "Moderate (10–20)": 15,
    "High (20–30)": 25,
    "Very High (30–36)": 33
}

study_hours = study_map[study_level]

ai_dependency = st.slider("AI Dependency (1-10)", 1, 10)

anxiety = st.slider("Exam Anxiety (1-10)", 1, 10)

tool_diversity = st.number_input("Tool Diversity (1–5)", 1, 5)

paid_subscription = st.selectbox("Paid GenAI Subscription",
    [True,False])

Institutional_Policy = st.selectbox("Institutional Policy on GenAI",
    ["Allowed_With_Citation","Strict_Ban","Actively_Encouraged"])

primary_use = st.selectbox("Primary Use Case",
    ["Debugging/Troubleshooting",
     "Copywriting/Drafting",
     "Ideation",
     "Summarizing_Reading",
     "Direct_Answer_Generation"])

skill_level = st.selectbox("Prompt Engineering Skill",
    ["Beginner","Intermediate","Advanced"])


skill_score = skill_retention_score(
    genai_hours,
    study_hours,
    skill_level,
    anxiety
)

cluster_names = {
    0: "AI-Dependent Learners",
    1: "Traditional Balanced Learners",
    2: "Stable Low-Stress Traditional Learners"
}
burnout_labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}
burnout_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Major_Category": major,
        "Year_of_Study": year,
        "Pre_Semester_GPA": gpa_input,
        "Weekly_GenAI_Hours": genai_hours,
        "Primary_Use_Case": primary_use,
        "Prompt_Engineering_Skill": skill_level,
        "Tool_Diversity": tool_diversity,
        "Paid_Subscription": paid_subscription,
        "Traditional_Study_Hours": study_hours,
        "Perceived_AI_Dependency": ai_dependency,
        "Institutional_Policy": Institutional_Policy,
        "Anxiety_Level_During_Exams": anxiety,
        "Skill_Retention_Score": skill_score
    }])

    gpa_pred = xgb_pipeline.predict(input_df)[0]

    burnout_pred = stacking_model.predict(input_df)[0]

    fuzz_features = [
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Perceived_AI_Dependency',
    'Anxiety_Level_During_Exams'
]

    x_scaled = scaler.transform(input_df[fuzz_features])
    x_fuzzy = x_scaled.T

    u, u0, d, jm, p, fpc= fuzz.cluster.cmeans_predict(
        x_fuzzy,
        cntr,
        m=1.3,
        error=0.005,
        maxiter=1000
    )

    cluster = np.argmax(u, axis=0)[0]
    burnout_label = burnout_map[burnout_pred]
    advice = recommend(cluster, burnout_label)
    membership = u[:, 0]


    st.subheader("Results")

    cluster_label = cluster_names[cluster]
    burnout_level = burnout_labels[burnout_pred]



    st.write("Predicted GPA:",gpa_pred)
    st.write("skill retention score:", skill_score)

    st.write("Burnout Level:", burnout_level)
    st.write("Cluster:", cluster_label)
    st.write(f"AI-Dependent Learners: {membership[0]*100:.2f}%")
    st.write(f"Traditional Balanced Learners: {membership[1]*100:.2f}%")
    st.write(f"Stable Low-Stress Traditional Learners: {membership[2]*100:.2f}%")
    st.write("Recommendation:", advice)



