import json
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load metrics from previous modules
# -------------------------------------------------
with open("metrics_centralized.json") as f:
    centralized_metrics = json.load(f)

with open("metrics_post_drift.json") as f:
    post_drift_metrics = json.load(f)

# -------------------------------------------------
# Extract accuracies (THIS WAS MISSING)
# -------------------------------------------------
accuracy_before = centralized_metrics["accuracy"]
accuracy_after = post_drift_metrics["accuracy"]

# -------------------------------------------------
# Console summary
# -------------------------------------------------
print("Evaluation Summary")
print(f"Centralized Model Accuracy (Module 2): {accuracy_before:.3f}")
print(f"Post-Drift Swapped Model Accuracy (Module 5): {accuracy_after:.3f}")
print(f"Accuracy Change: {accuracy_after - accuracy_before:+.3f}")

# -------------------------------------------------
# Bar chart comparison (color-enhanced)
# -------------------------------------------------
labels = [
    "Before Drift\n(Centralized Model)",
    "After Drift\n(Swapped Federated Model)"
]
accuracies = [accuracy_before, accuracy_after]

colors = ["#4C72B0", "#55A868"]  # blue = baseline, green = improvement

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, accuracies, color=colors)

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Federated HeartCare: Centralized vs Drift-Aware Performance")

# Annotate bars with values
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
