import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------------------------------
# Load client datasets
# -------------------------------------------------
clients = {
    "Typical": pd.read_csv("typical.csv"),
    "Athletic": pd.read_csv("athletic.csv"),
    "Diver": pd.read_csv("diver.csv")
}

# Normalize columns
for k in clients:
    clients[k].columns = clients[k].columns.str.strip().str.lower()

# -------------------------------------------------
# Shared preprocessing (IDENTICAL across clients)
# -------------------------------------------------
sample_df = next(iter(clients.values()))
X_sample = sample_df.drop(columns=["num"])

categorical_cols = X_sample.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_sample.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
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
# Local training + model persistence
# -------------------------------------------------
def train_local_model(client_df, client_name):
    X = client_df.drop(columns=["num"])
    y = (client_df["num"] > 0).astype(int)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    # SAVE model for Module 5
    model_path = f"model_{client_name.lower()}.pkl"
    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

    clf = model.named_steps["classifier"]
    return clf.coef_, clf.intercept_

# -------------------------------------------------
# Federated Averaging
# -------------------------------------------------
ROUNDS = 3
global_weights = None
global_bias = None

for rnd in range(ROUNDS):
    print(f"\nFederated Round {rnd + 1}")

    weights, biases = [], []

    for name, df in clients.items():
        w, b = train_local_model(df, name)
        weights.append(w)
        biases.append(b)
        print(f"Client {name} trained locally")

    global_weights = np.mean(weights, axis=0)
    global_bias = np.mean(biases, axis=0)

print("\nFederated Learning Completed")
print("Global weight shape:", global_weights.shape)
