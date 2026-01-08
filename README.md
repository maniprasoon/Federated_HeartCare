# Federated HeartCare: Privacy-Preserving & Drift-Aware Heart Disease Prediction

An end-to-end **adaptive healthcare intelligence system** that predicts heart disease while preserving patient privacy.  
The system combines **Federated Learning, Concept Drift Detection, and real-time visualization** to deliver **personalized, reliable, and privacy-safe predictions**.

Unlike traditional models that assume static patient behavior, Federated HeartCare focuses on **continuous adaptation** as human physiology and lifestyle evolve.

---

## 🚀 Project Overview

Conventional heart disease prediction systems:
- Centralize sensitive patient data
- Assume stable data distributions
- Degrade in performance when user behavior changes

**Federated HeartCare solves these limitations** by training models collaboratively across distributed clients without sharing raw data and dynamically adapting to physiological changes using concept drift detection.

**Key outcomes:**
- Privacy-preserving model training
- Continuous physiological monitoring
- Automatic detection of behavioral drift
- Adaptive model switching for sustained accuracy
- Executive-ready visual analytics via Streamlit

---

## 🧠 Architecture

**Adaptive Federated Healthcare Intelligence Pipeline**

Wearable / Client Health Data (CSV)  
→ Local Model Training (Client Devices)  
→ Federated Aggregation (Server – FedAvg)  
→ Continuous Monitoring & Drift Detection  
→ Adaptive Model Switching  
→ Streamlit Dashboard (Real-Time Insights)

This architecture ensures **data privacy, adaptability, and real-world reliability**, making it suitable for modern digital healthcare systems.

---

## 🛠️ Tech Stack

- **Python**
- **Federated Learning** (Simulated FedAvg)
- **Scikit-learn** – Predictive modeling
- **River** – Concept drift detection
- **Pandas & NumPy** – Data processing
- **Matplotlib** – Performance visualization
- **Joblib** – Model persistence
- **Streamlit** – Interactive web dashboard

---

## 📊 Data Sources

The project uses the **UCI Heart Disease Dataset**, extended with **synthetic physiological variations** to simulate real-world user categories:

- **Typical users** – baseline physiological patterns  
- **Athletic users** – lower resting heart rate, higher activity  
- **Diver users** – altered heart and oxygen dynamics  

These datasets simulate **continuous monitoring scenarios** encountered in wearable-based healthcare systems.

---

## 🔍 Key Features

- **Privacy-Preserving Federated Learning**  
  Trains models locally on user data and shares only model parameters, not raw data.

- **Multi-Profile Personalization**  
  Maintains specialized models for typical, athletic, and diver user profiles.

- **Concept Drift Detection**  
  Continuously monitors physiological signals to detect significant distributional changes.

- **Adaptive Model Switching**  
  Automatically swaps predictive models when a drift event is detected.

- **Real-Time Streamlit Dashboard**  
  Visualizes monitoring, drift alerts, model adaptation, and performance evaluation.

- **Dual Performance Visualization**  
  Uses both **trend line charts** and **comparative bar charts** for clear evaluation.

---

## 📈 Streamlit Dashboard Pages

1. **System Overview**
   - Privacy status
   - Federated learning mode
   - System health indicators

2. **User Profile**
   - User category selection
   - Active model visualization

3. **Live Monitoring**
   - Heart rate trends
   - Activity level streams

4. **Concept Drift Detection**
   - Drift alerts
   - Explanation of detected changes

5. **Model Adaptation**
   - Previous vs current model
   - Adaptation confirmation

6. **Performance Evaluation**
   - Accuracy trends (before vs after adaptation)
   - Centralized vs federated bar comparison

---

## 📂 Project Structure

```text
FEDERATED_HEARTCARE/
│
├── app.py                          # Streamlit frontend application
├── requirements.txt                # Project dependencies
│
├── athletic.csv                    # Athletic user dataset
├── diver.csv                       # Diver user dataset
├── typical.csv                     # Typical user dataset
├── heart_disease_uci.csv           # Base heart disease dataset
│
├── model_typical.pkl               # Trained model for typical users
├── model_athletic.pkl              # Trained model for athletic users
├── model_diver.pkl                 # Trained model for diver users
│
├── metrics_centralized.json        # Centralized model evaluation metrics
├── metrics_post_drift.json         # Post-drift federated metrics
│
├── module1_data_preparation.py     # Data preparation & user simulation
├── module2_centralized_model.py    # Centralized learning baseline
├── module3_federated_learning.py   # Federated learning logic
├── module4_drift_detection.py      # Concept drift detection
├── module5_model_swapping.py       # Adaptive model switching
└── module6_evaluation.py           # Performance evaluation
```
## ▶️ Running the Project Locally

To run the application on your local machine:

```bash
pip install -r requirements.txt
streamlit run app.py
```
## 🌐 Deployment

The application is deployed using **Streamlit Cloud**, enabling public access without local setup while maintaining full reproducibility.

### Deployment Steps:
1. Push the complete project repository to GitHub  
2. Log in to Streamlit Cloud  
3. Connect the GitHub repository  
4. Select `app.py` as the entry point  
5. Deploy and share the generated public URL  

---

## 💡 Why This Project Matters

- Demonstrates **privacy-first AI** in healthcare applications  
- Addresses **real-world concept drift**, a challenge often ignored in academic models  
- Combines **federated learning with adaptive intelligence** for sustained accuracy  
- Bridges **research concepts with deployable, real-time systems**  
- Mirrors how **AI-driven healthcare monitoring** works in practical environments  

---
