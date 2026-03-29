#  Student Performance Prediction using Machine Learning

##  Overview

This project builds a machine learning model to predict student exam performance based on academic and lifestyle factors. It follows a complete end-to-end ML pipeline including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment using Streamlit.

---

##  Problem Statement

Educational systems lack predictive tools to estimate student performance in advance. This project aims to develop a model that predicts exam scores and identifies key influencing factors such as study habits, attendance, and mental health.

---

##  Machine Learning Pipeline

* Data Collection
* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training
* Model Evaluation
* Model Selection
* Deployment (Streamlit)

---

##  Models Used

* Linear Regression (Best Model ✅)
* Decision Tree Regressor
* Random Forest Regressor

---

##  Model Performance

| Model             | R² Score | RMSE |
| ----------------- | -------- | ---- |
| Linear Regression | 0.799    | 7.33 |
| Random Forest     | 0.770    | 7.83 |
| Decision Tree     | 0.701    | 8.95 |

📌 **Linear Regression was selected as the final model due to highest R² and lowest RMSE.**

---

##  Key Insights

* Study hours and attendance are the most influential features
* Mental health and sleep significantly affect performance
* Excessive social media and entertainment negatively impact scores
* Simpler models like Linear Regression performed better due to linear relationships in data

---

##  Project Structure

```
student-performance-ml/
│
├── app.py
├── best_model.pkl
├── requirements.txt
│
├── data/
│   └── student_habits_performance.csv
│
├── Notebook/
│   └── Notebook.ipynb
│
├── Model/
│   └── model_loader.py
|   └── train.py
│
├── Utils/
│   ├── preprocess.py
│
└── reports/
    └── report.docx
```

---

## ▶ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student-performance-ml.git
cd student-performance-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Train the Model (IMPORTANT)

```bash
python Model/train.py
```

### 4. Run the App

```bash
streamlit run app.py
```

---

##  Input Features

* Study Hours per Day
* Attendance Percentage
* Mental Health Rating
* Sleep Hours
* Part-Time Job

---

##  Output

* Predicted Exam Score (0–100)
* Input Feature Visualization
* Score Progress Indicator

---

##  Important Note

The order of input features must match the training data format to ensure accurate predictions.

---

##  Key Learnings

* End-to-end Machine Learning pipeline
* Model comparison and evaluation
* Hyperparameter tuning using GridSearchCV
* Debugging real-world ML issues (feature order mismatch)
* Building interactive ML apps using Streamlit

---

##  Future Improvements

* Add more features for improved accuracy
* Use advanced models like XGBoost
* Deploy on cloud platforms (Streamlit Cloud / AWS)
* Add explainability (SHAP / LIME)

---
## App Preview
<img width="953" height="902" alt="{251B6478-F795-4C7F-8A39-F3BE650AC407}" src="https://github.com/user-attachments/assets/96926872-3d22-4cc9-9851-ca4ea6bba7a9" />

##  Author

**Adish Jain**
B.Tech AI & ML

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
