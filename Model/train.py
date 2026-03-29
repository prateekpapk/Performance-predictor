import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("data/student_habits_performance.csv")

# Features & target
features = [
    'attendance_percentage',
    'mental_health_rating',
    'study_hours_per_day',
    'sleep_hours',
    'part_time_job'
]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["part_time_job"] = le.fit_transform(df["part_time_job"])

X = df[features]
y = df['exam_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10]
        }
    }
}

best_model = None
best_rmse = float("inf")

for name, config in models.items():
    grid = GridSearchCV(config["model"], config["params"], cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} → RMSE: {rmse:.2f}, R2: {r2:.3f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = grid.best_estimator_

# Save best model
joblib.dump(best_model, "best_model.pkl")

print("Best model saved!")