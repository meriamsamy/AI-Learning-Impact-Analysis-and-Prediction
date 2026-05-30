# 🤖 AI Student Impact Analysis

A machine learning project analyzing how generative AI tool usage affects university students' academic performance, burnout risk, and learning behavior. The system predicts a student's expected GPA and burnout risk level, then generates a personalized study recommendation based on their behavioral cluster.

---

## 📊 Dataset

`ai_student_impact_dataset.csv` — student-level records covering:
AI usage habits, study patterns, GPA, anxiety levels, and institutional policies.

---

## 🧠 Project Structure

The notebook covers four main tasks:

---

### 1. Exploratory Data Analysis (EDA)

- Distributions of numeric features (GPA, GenAI hours, anxiety, skill retention)
- Count plots for categorical features (major, year, primary use case, burnout risk)
- Bivariate analysis
- Correlation heatmap

---

### 2. GPA Prediction (Regression)

- **Target:** `Post_Semester_GPA`

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|------|----------|---------|-----------|----------|
| Random Forest | — | — | — | — |
| XGBoost | 92.16% | 91.53% | 0.14 | 0.11 |
| Voting Ensemble (Linear + Poly + RF) | 92.20% | 90.76% | 0.15 | 0.12 |

> Best performer: **XGBoost** with a test R² of 91.53% and lowest error, showing strong generalization and stability.

---

### 3. Burnout Risk Classification

- **Target:** `Burnout_Risk_Level` (Low / Medium / High)

| Model | Train Accuracy | Test Accuracy |
|------|---------------|---------------|
| XGBoost Classifier | 69.55% | 53.09% |
| Stacking Ensemble (SVM + Logistic Regression + Random Forest) | 64.74% | 53.82% |

---

### Per-Class Performance

| Class | XGBoost F1 | Stacking F1 |
|------|-----------|-------------|
| Low | 0.50 | 0.52 |
| Medium | 0.54 | 0.54 |
| High | 0.55 | 0.56 |

> The moderate accuracy (~53–54%) reflects the inherent overlap in burnout behavior, making it a naturally ambiguous classification problem.

---

### 4. Fuzzy C-Means Clustering

- **Features:**
  - Weekly GenAI Hours
  - Traditional Study Hours
  - Perceived AI Dependency
  - Anxiety Level During Exams

- **Clusters Identified:**

| Cluster | Profile |
|--------|---------|
| 0 | High AI Dependency Learners — heavy AI usage, lower traditional study time, higher anxiety |
| 1 | Traditional Learners — low AI usage, structured study habits |
| 2 | Stable Low-Stress Learners — balanced habits, lowest anxiety, most consistent performance |

---

### 5. Personalized Recommendation Engine

A rule-based system that takes:
- Cluster assignment
- Burnout prediction

and returns tailored study advice depending on student behavior patterns.

---

## 🔍 Key Findings

- `Pre_Semester_GPA` is the strongest predictor of final GPA (correlation ≈ 0.93)
- Weekly GenAI usage has near-zero correlation with GPA (≈ -0.02)
- Higher perceived AI dependency correlates with lower skill retention
- Balanced learners (Cluster 2) show the lowest anxiety and most stable performance

---

## 💾 Saved Models

| File | Description |
|------|-------------|
| `gpa_model.pkl` | XGBoost regression pipeline |
| `burnout_model.pkl` | Stacking classification model |
| `scaler.pkl` | MinMaxScaler for fuzzy clustering |
| `fuzzy_centers.pkl` | Fuzzy C-Means cluster centers |

---

## 🚀 Live Demo

You can access the deployed web application here:

👉 [(https://ai-learning-impact-analysis-and-prediction-9dcwumnls2bduzbjjmj.streamlit.app/)](https://github.com/meriamsamy/AI-Learning-Impact-Analysis-and-Prediction/tree/main)
