import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Load available models safely
# -------------------------------------------------
model_files = {
    "Typical": "model_typical.pkl",
    "Athletic": "model_athletic.pkl",
    "Diver": "model_diver.pkl"
}

models = {}

for name, path in model_files.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        print(f"Warning: Model file not found for {name}")

# -------------------------------------------------
# Load evaluation data (same schema as training)
# -------------------------------------------------
# -------------------------------------------------
# Load evaluation data
# -------------------------------------------------
eval_data = pd.concat([
    pd.read_csv("typical.csv"),
    pd.read_csv("athletic.csv"),
    pd.read_csv("diver.csv")
], ignore_index=True)

eval_data.columns = eval_data.columns.str.strip().str.lower()

# -------------------------------------------------
# SCHEMA ALIGNMENT FIX (critical)
# -------------------------------------------------
# Some saved models were trained with 'user_type'
# Ensure evaluation data contains it if required
if "user_type" not in eval_data.columns:
    eval_data["user_type"] = "unknown"

X_eval = eval_data.drop(columns=["num"])
y_eval = (eval_data["num"] > 0).astype(int)


# -------------------------------------------------
# Current active model
# -------------------------------------------------
current_state = "Typical"

# -------------------------------------------------
# Model switching + evaluation
# -------------------------------------------------
import json

def swap_model(new_state):
    global current_state

    if new_state not in models:
        raise ValueError(f"Model for state '{new_state}' is not available")

    current_state = new_state
    model = models[new_state]

    # Evaluate swapped model
    y_pred = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)

    print(f"Model successfully switched to {new_state}")
    print(f"Post-swap accuracy: {accuracy:.3f}")

    # ✅ WRITE METRICS HERE (CRITICAL)
    with open("metrics_post_drift.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

    return accuracy


# -------------------------------------------------
# Example drift response
# -------------------------------------------------
post_drift_accuracy = swap_model("Athletic")


