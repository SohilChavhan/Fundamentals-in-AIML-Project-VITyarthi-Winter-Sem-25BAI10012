# Fundamentals-in-AIML-Project-VITyarthi-Winter-Sem-25BAI10012
**Retail Sales Forecasting and Inventory Optimization**
I used that layout because of a previous instruction to format every response as a paragraph.

Here is the project properly broken down into individual bullet points for your README:

Retail Sales Forecasting and Inventory Optimization

Core Objective: Built a high-performance machine learning pipeline utilizing a LightGBM regressor to predict daily retail store sales volume, assisting supply chain managers in preventing stockouts and optimizing logistics.

Temporal Feature Engineering: Deconstructed raw calendar dates into granular numeric features (year, month, week, day of the week) and mathematically engineered a custom binary flag to explicitly capture weekend purchasing behaviors.

Algorithmic Optimization: Converted categorical columns (Store ID, Location Type, Region Code) into native Pandas category types. This completely bypassed memory-heavy one-hot encoding and drastically reduced model training time.

Predictive Performance: Achieved exceptional mathematical accuracy with an R² score over 0.90 (explaining the vast majority of purchasing variance) and a Mean Absolute Error (MAE) of roughly 6.7 items per day.

Business Insights: Leveraged gain metrics for feature importance analysis, revealing that the specific Store ID, followed by the Day of the Week and Location Type, are the absolute strongest baseline predictors of daily sales.

Setup Instructions: To reproduce this project locally, ensure pandas, numpy, scikit-learn, lightgbm, and matplotlib are installed in your Python environment. The supplement.csv data file must be placed in the exact same directory as the main Python script before execution.
