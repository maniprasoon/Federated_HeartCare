import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------------------------------
# Load merged dataset
# -------------------------------------------------
data = pd.concat([
    pd.read_csv("typical.csv"),
    pd.read_csv("athletic.csv"),
    pd.read_csv("diver.csv")
], ignore_index=True)

data.columns = data.columns.str.strip().str.lower()

# -------------------------------------------------
# Features & Target
# -------------------------------------------------
X = data.drop(columns=["num"])
y = (data["num"] > 0).astype(int)

# -------------------------------------------------
# Column types
# -------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# -------------------------------------------------
# Preprocessing pipelines
# -------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# -------------------------------------------------
# Model pipeline
# -------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -------------------------------------------------
# Train/Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# Train & Evaluate
# -------------------------------------------------
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Module 2: Centralized Model Accuracy:", accuracy)

import json

with open("metrics_centralized.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)
