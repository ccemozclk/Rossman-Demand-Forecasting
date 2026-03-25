# 🛒 Retail Demand Forecasting: End-to-End MLOps Pipeline

# 📌 Project Overview

This project is a complete, production-ready Machine Learning system designed to forecast 6 weeks of daily sales for 1,115 Rossmann stores across Europe. By accurately predicting future sales, retail managers can optimize store staffing, minimize warehouse holding costs, and prevent out-of-stock scenarios.

Going beyond a standard Jupyter Notebook analysis, this project implements a full MLOps architecture. It features an automated object-oriented training pipeline, a REST API for real-time inference, and an interactive web dashboard for business stakeholders—all fully containerized using Docker.

# 🏗️ Architecture & Microservices

The system is built on a modern decoupled architecture:

- **Machine Learning Pipeline (Backend Engine):** Object-Oriented pipelines (src/components) that automate data ingestion, transformation (handling missing values, feature engineering, encoding), and model training using LightGBM.

- **FastAPI Service (Inference API):** A lightning-fast REST API that loads the pre-trained .pkl model and serves predictions over HTTP.

- **Streamlit Dashboard (Frontend UI):** A business-facing interactive web application divided into two core modules:

- **EDA & Business Insights:** Visualizes historical data and provides an Executive Summary on the impact of promotions and store types.

- **AI Simulation:** Allows managers to input future dates/promo scenarios and get instant sales predictions along with a dynamically calculated Confidence Interval (± Error Margin) based on the model's live RMSPE score.

- **Docker Compose:** Orchestrates the API and UI containers within an isolated network for a single-command deployment.

# 🛠️ Tech Stack

- **Language:** Python 3.10

- **ML & Data:** Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, Optuna

- **MLOps & API:** FastAPI, Uvicorn, Pydantic

- **Frontend:** Streamlit, Seaborn, Matplotlib

- **Deployment:** Docker, Docker Compose

- **Dataset:** Rossmann Store Sales (e.g https://www.kaggle.com/competitions/rossmann-store-sales/data)

# 🚀 Key Achievements

- **Automated Data Pipelines:** Replaced static notebook code with reusable OOP classes (DataIngestion, DataTransformation, ModelTrainer).

- **Time-Based Validation:** Implemented strict time-based train/validation splits to prevent data leakage, bringing the real-world RMSPE down to ~12.83%.

- **Dynamic Metric Tracking:** The training pipeline automatically calculates the validation RMSPE and saves it as a metrics.json artifact, allowing the frontend to display dynamic, honest confidence intervals to the end-user.

- **Resilient Inference:** The DataTransformation module safely handles unseen categorical inputs during inference, preventing API crashes in production.

# 📁 Project Structure

Rossman-Demand-Forecasting/
├── artifacts/                  # Stores generated model (.pkl) and metrics (.json)
├── dataset/
│   ├── raw/                    # Original raw CSV files
│   └── processed/              # Cleaned and transformed data for training
├── src/
│   ├── api/                    # FastAPI application (app.py)
│   ├── components/             # OOP Data & Model modules (ingestion, transformation, trainer)
│   ├── pipelines/              # Training and Inference execution scripts
│   ├── ui/                     # Streamlit dashboard (dashboard.py)
│   └── utils/                  # Helper functions (logger, common object loaders)
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # Environment definition
├── setup.py                    # Local package installer
└── requirements.txt            # Python dependencies

# ⚙️ How to Run the Project

- **Method 1:** The Docker Way (Recommended)

- Make sure Docker Desktop is running on your machine, then execute a single command from the root directory:

    - ``docker-compose up --build``

- Frontend (UI): Open http://localhost:8501 in your browser.

- Backend (API Docs): Open http://localhost:8000/docs to test the API via Swagger UI.

- **Method 2:** Local Python Environment

- If you prefer running it locally without Docker:

- 1. Install dependencies and the local package:

    - ``pip install -r requirements.txt``
        ``pip install -e .``

- 2. (Optional) Run the training pipeline to generate a fresh model:

    - ``python src/pipelines/training_pipeline.py``

- 3. Start the FastAPI Server:

    - ``uvicorn src.api.app:app --host 0.0.0.0 --port 8000``

- 4. Start the Streamlit App (in a new terminal):

    - ``streamlit run src/ui/dashboard.py``

# 📊 Business Value Delivered

- **Financial Precision:** A ~12% RMSPE allows for tighter financial planning and realistic revenue expectations.

- **Actionable Insights:** Discovered that promotions in B-type stores yield lower ROI compared to A and C-type stores, allowing the marketing team to reallocate their budget efficiently.

- **Stakeholder Transparency:** Translates complex ML metrics into business terms (e.g., displaying "± €850 margin of error" instead of abstract statistical jargon).

# Created by Cem OZCELIK | Data Scientist & Industrial Engineeras



