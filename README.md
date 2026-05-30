# AI-Learning-Impact-Analysis-and-Prediction

A machine learning project analyzing how generative AI tool usage affects university students' academic performance, burnout risk, and learning behavior.

---

## 📊 Dataset

`ai_student_impact_dataset.csv` — student-level records covering:
- AI usage habits  
- Study patterns  
- GPA  
- Anxiety levels  
- Institutional policies  
- Learning behavior indicators  

---

## 🧠 Project Structure

The project is divided into four main components:

---

### 1. Exploratory Data Analysis (EDA)

- Distributions of numeric features (GPA, GenAI hours, anxiety, skill retention)
- Count plots for categorical variables (major, year, use case, burnout risk)
- Bivariate analysis
- Correlation heatmap

---

### 2. GPA Prediction (Regression)

- **Target:** `Post_Semester_GPA`
- **Models used:**
  - Random Forest (GridSearchCV tuned)
  - XGBoost Regressor
  - Voting Ensemble (Linear + Polynomial + Random Forest)

- **Evaluation Metrics:**
  - R² Score
  - RMSE
  - MAE

---

### 3. Burnout Risk Classification

- **Target:** `Burnout_Risk_Level` (Low / Medium / High)

- **Models used:**
  - XGBoost Classifier
  - SVC (GridSearchCV tuned)
  - Stacking Ensemble (SVM + Logistic Regression + Random Forest)

- **Evaluation Metrics:**
  - Accuracy
  - Classification Report
  - Confusion Matrix

---

### 4. Fuzzy C-Means Clustering

- **Features used:**
  - Weekly GenAI Hours
  - Traditional Study Hours
  - Perceived AI Dependency
  - Anxiety Level During Exams

- **Results:**
  - 3 student clusters identified:
    - High AI Dependency Learners
    - Traditional Learners
    - Stable Low-Stress Learners

- Includes a **personalized recommendation system** based on cluster + burnout level

---

## 🔍 Key Findings

- `Pre_Semester_GPA` is the strongest predictor of final GPA (correlation ≈ 0.93)
- Weekly GenAI usage has near-zero correlation with GPA (≈ -0.02)
- Higher AI dependency correlates with lower skill retention
- Balanced learners (Cluster 2) show lowest anxiety and most stable performance

---

## 💾 Saved Models

| File | Description |
|------|------------|
| `gpa_model.pkl` | XGBoost regression pipeline |
| `burnout_model.pkl` | Stacking classification model |
| `scaler.pkl` | MinMaxScaler for fuzzy clustering |
| `fuzzy_centers.pkl` | Fuzzy C-Means cluster centers |

---

## ⚙️ Dependencies

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- scikit-fuzzy  
- joblib  

---

## 🚀 Goal

To understand the impact of generative AI tools on students’ academic performance, mental health, and learning behavior using machine learning and clustering techniques.
