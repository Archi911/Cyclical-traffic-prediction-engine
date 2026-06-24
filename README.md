# Cyclical-traffic-prediction
Machine learning pipeline that predicts interstate traffic volume

## Highlights
* **Time-Series Anomaly Handling:** Implemented forward-fill (`ffill`) imputation to correct physical sensor failures (e.g., 0 Kelvin readings) without introducing data leakage.
* **Cyclical Temporal Encoding:** Transformed linear timestamps into 2D mathematical circles using Sine/Cosine trigonometry, preventing the algorithm from misinterpreting the proximity of midnight and 11 PM.
* **Algorithm Benchmarking:** Evaluated a gauntlet of 6 models (Linear, Ridge, Trees, XGBoost, LightGBM, CatBoost), mathematically proving the necessity of tree-based architectures for multimodal data.
* **Bayesian Optimization:** Utilized `Optuna` to navigate the hyperparameter space of the winning XGBoost model, restricting `max_depth` and `subsample` rates to strictly prevent overfitting.
* **MLOps Serialization:** Exported the final optimized pipeline via `joblib` for low-latency production deployment.

## 📊 Performance Metrics
* **Baseline XGBoost $R^2$:** 0.9592
* **Optuna Tuned XGBoost $R^2$:** 0.9698
* **Final RMSE:** ~346 vehicles 

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-150458?style=for-the-badge&logo=xgboost&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-2A2A2A?style=for-the-badge&logo=python&logoColor=white)

## 🚀 Quick Start
To run this pipeline locally and make predictions:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Load the pre-trained model in any Python script:
```python
import joblib
import pandas as pd

# Load the optimized model
model = joblib.load('models/traffic_xgboost_model.joblib')

# Pass engineered cyclical data for an instant prediction
# prediction = model.predict(new_data)
