# 🔋 BatteryBuddy: Predictive Health Management (PHM) for Li-ion Cells

**BatteryBuddy** is an end-to-end machine learning solution designed to tackle the "silent failure" problem in Lithium-ion batteries. By analyzing telemetry data- voltage, current, and temperature, logged across hundreds of charge-discharge cycles, it predicts a battery's **State of Health (SoH)** and **Remaining Useful Life (RUL)**.

### 🚀 Quick Links
*   **Live Dashboard:** [batterybuddy.streamlit.app](https://batterybuddy.streamlit.app/)
*   **Training Notebook:** [Google Colab](https://colab.research.google.com/drive/1TJUg__qUKIUHvr_cHj6hYhltaiJqNhny?usp=sharing)
*   **Primary Data Source:** [NASA Li-ion Battery Aging Dataset (Kaggle)](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset/data)

---

## 🧠 The Problem
Industrial cells, EV batteries, and backup power systems degrade slowly and silently. By the time a drop in performance is noticeable to a human operator, the window for proactive, cost-effective replacement has already closed. BatteryBuddy provides a "Business Impact" layer that translates technical degradation into risk-assessment costs.

## 🛠️ Technical Architecture
*   **Data Engineering:** A robust pipeline built to handle real-world sensor issues: simulated noise, random dropouts, calibration drift, and voltage spikes.
*   **Hybrid Modeling:**
    *   **XGBoost:** Used for sharp, feature-based cycle-by-cycle regressions.
    *   **LSTM (PyTorch):** Captures the temporal "memory" of the battery to estimate uncertainty and long-term trends.
*   **Explainability:** Integrated **SHAP** (SHapley Additive exPlanations) to demystify "black box" predictions, showing exactly how temperature or voltage sag influenced the health score.
*   **Frontend:** A Streamlit-based dashboard featuring a business impact layer (calculating "Replace Now vs. Risk It" costs in INR).

---

## 📊 The "Honest Finding" (OOD Performance)
Most ML projects only report their best numbers. During development, I found that while the model hits an **$R^2$ of 0.93** on batteries within the training distribution, the performance drops to **-3.9** on completely different battery architectures. 

This gap is a key takeaway of the project: it highlights the critical challenge of **Out-of-Distribution (OOD)** generalization in Battery Management Systems and the need for domain-specific fine-tuning.

---

## 🧪 Demonstration Guide
The NASA dataset contains various experiments run under different conditions. For the best experience with the live demo, use these specific Battery IDs:

### ✅ Recommended for Demos
*   **B0005 (Best Showcase):** 168 cycles with smooth, clean degradation from 100% down to ~70%. Perfect for showing clear RUL trends.
*   **B0034:** Shows deep degradation (down to 40-60%), ideal for triggering the "Critical Replacement" alerts in the dashboard.
*   **B0043:** 65 cycles of clean, moderate degradation data.

### ❌ Avoid for Demos
*   **B0025 - B0028:** Only 28 cycles; not enough for the LSTM window to generate meaningful history.
*   **B0031 / B0032:** These batteries barely degrade (staying above 90%), making for a "flat" and uninteresting visualization.
*   **B0052:** Only 4 cycles—completely statistically insignificant.

---

## 📁 Repository Structure
```text
DASHBOARD/
├── models/
│   ├── xgb_model.pkl          # Trained XGBoost weights
│   ├── lstm_model.pt          # PyTorch LSTM state dict
│   ├── scaler.pkl             # Normalization parameters
│   |── cycle_features_clean.csv
|   └── model_summary.json     # Evaluation metrics
├── app.py                     # Streamlit entry point
├── requirements.txt           # Dependency list

```

## ⚙️ Local Setup
1. **Clone the repo:**
   ```bash
   git clone https://github.com/[your-username]/batterybuddy.git
   cd batterybuddy/dashboard
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

**Developed by Yuvraj Singh Hajari**
*3rd Year Computer Science & Engineering Student | Focus on Full-Stack & ML*
```