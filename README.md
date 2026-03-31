# Fundamentals-in-AIML-Project-VITyarthi-Winter-Sem-25BAI10012
# 📈 Retail Sales Forecasting and Inventory Optimization

## 🎯 Core Objective
I Built a high-performance machine learning pipeline utilizing a LightGBM regressor to predict daily retail store sales volume, assisting supply chain managers in preventing stockouts and optimizing logistics.

## ⚙️ Key Features & Methodology
* **Temporal Feature Engineering:** I Deconstructed raw calendar dates into granular numeric features (year, month, week, day of the week) and mathematically engineered a custom binary flag to explicitly capture weekend purchasing behaviors.
* **Algorithmic Optimization:** I Converted categorical columns (Store ID, Location Type, Region Code) into native Pandas category types. This completely bypassed memory-heavy one-hot encoding and drastically reduced model training time.

## 🚀 Predictive Performance
* **Accuracy:** Achieved exceptional mathematical accuracy with an R² score over **0.90** (explaining the vast majority of purchasing variance).
* **Error Rate:** Reached a Mean Absolute Error (MAE) of roughly **6.7**, meaning daily sales predictions are generally within seven items of the true volume.

## 💡 Business Insights
I leaned on gain-based metrics to sort out which features mattered most, and it became clear that the specific Store ID stood out first, with the Day of the Week and the Location Type coming next as the strongest baseline signals for predicting daily sales.

## 💻 Installation and Setup
To reproduce this project locally, follow these steps:

1. Ensure you have a Python environment set up with the following libraries installed:
   ```bash
   pip install pandas numpy scikit-learn lightgbm matplotlib
