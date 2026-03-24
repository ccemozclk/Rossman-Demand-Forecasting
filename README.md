# 🛒 Retail Demand Forecasting: Rossmann Store Sales

## 📌 Project Overview
This project aims to forecast 6 weeks of daily sales for 1,115 Rossmann stores located across Europe. Accurate sales forecasting enables retail managers to optimize store staffing, increase productivity, and minimize overstocking/out-of-stock costs.

By applying **Time-Series Analysis** and **Machine Learning**, this project successfully lowered the Root Mean Square Percentage Error (RMSPE) down to **12.67%**, providing a highly reliable model for business decision-makers.

## 🛠️ Tech Stack & Methodologies
* **Language:** Python
* **Data Manipulation & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning Algorithms:** LightGBM, XGBoost
* **Hyperparameter Optimization:** Optuna (Bayesian Optimization)
* **Feature Engineering:** Time-based Lag features, Rolling Windows (Moving Averages), One-Hot & Ordinal Encoding.
* **Validation Strategy:** Time-Based Train/Validation Split (to prevent data leakage).
* **Dataset:** Rossmann Store Sales (e.g https://www.kaggle.com/competitions/rossmann-store-sales/data)

## 🚀 Key Steps & Achievements
1. **Data Preprocessing:** Handled missing values logically (e.g., assigning max distance for missing competitor distance) and merged static store data with historical daily sales.
2. **Business Rule Implementation:** Filtered out days when stores were closed to prevent the model from learning zero-sales as poor performance.
3. **Statistical Testing:** Conducted ANOVA and Welch's T-Test to statistically prove the significant impact of 'Day of Week' and 'Promotions' on sales.
4. **Feature Engineering:** Created crucial time-series features such as `Sales_Lag_1`, `Sales_Lag_7`, and `Sales_Roll_Mean_30` to help the model understand historical trends.
5. **Model Selection & Tuning:** Compared XGBoost and LightGBM. Selected LightGBM for its superior speed and accuracy. Tuned hyperparameters using **Optuna** to achieve the final model.

## 📊 Business Value (For Stakeholders)
* **Financial Precision:** A 12.67% RMSPE means highly accurate revenue predictions, allowing for tighter financial planning.
* **Supply Chain Optimization:** Knowing exactly how much a store will sell 6 weeks in advance drastically reduces warehouse holding costs and prevents missed sales due to stockouts.
* **Workforce Management:** The model's sensitivity to promotional days and day-of-week trends allows for precise staff shift scheduling.

## 🔮 Next Steps (Future Work)
- [ ] Develop a REST API using **FastAPI** to serve the `rossmann_model.pkl`.
- [ ] Build an interactive web interface using **Streamlit** for business users.
- [ ] **Dockerize** the entire application for scalable and isolated deployment.

---
*Created by Cem OZCELIK*
