import pandas as pd
import numpy as np

# Load dataset (UCI Heart or similar CSV)
data = pd.read_csv(
    r"D:\OneDrive_2025-12-31\HCL problem statements projects\Federated_HeartCare\heart_disease_uci.csv"
)


# Create user categories
def simulate_users(data):
    typical = data.copy()
    athletic = data.copy()
    diver = data.copy()

    # Typical users – baseline
    typical["user_type"] = "Typical"

    # Athletic users – lower resting heart rate
    athletic["thalch"] = athletic["thalch"] - np.random.randint(5, 15, size=len(athletic))
    athletic["user_type"] = "Athletic"

    # Divers – oxygen & heart variations
    diver["thalch"] = diver["thalch"] - np.random.randint(10, 20, size=len(diver))
    diver["user_type"] = "Diver"

    return typical, athletic, diver

typical, athletic, diver = simulate_users(data)

# Save datasets
typical.to_csv("typical.csv", index=False)
athletic.to_csv("athletic.csv", index=False)
diver.to_csv("diver.csv", index=False)

print("Module 1 Completed: User datasets created")

