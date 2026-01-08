from river.drift import ADWIN
import numpy as np

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
np.random.seed(42)

# -------------------------------------------------
# Initialize drift detector
# -------------------------------------------------
drift_detector = ADWIN()

# -------------------------------------------------
# Simulated heart-rate stream
# -------------------------------------------------
heart_rate_stream = np.random.normal(loc=70, scale=2, size=100)

# Introduce concept drift
heart_rate_stream[50:] += 20

# -------------------------------------------------
# Stream processing
# -------------------------------------------------
for i, rate in enumerate(heart_rate_stream):
    drift_detector.update(rate)

    if drift_detector.drift_detected:
        print(f"Drift detected at index {i}")
        break

print("Module 4: Drift detection completed")
