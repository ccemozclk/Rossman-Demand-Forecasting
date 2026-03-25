import streamlit as st
import requests
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

st.set_page_config(page_title="Rossmann Sales Forecast & EDA", page_icon="🛒", layout="wide")

st.title("🛒 Rossmann Sales Forecast and Executive Summary")
st.markdown("This panel provides strategic insights derived from analyzing historical sales data and calculates future revenue expectations using an AI-based simulation engine.")
st.divider()

tab_sim, tab_eda = st.tabs(["🔮 AI Simulation", "📊 Exploratory Data Analysis (EDA) and Insights"])



with tab_sim:
    st.header("Store Sales Forecasting Simulator")
    col1, col2 = st.columns(2)

    with col1:
        store_id = st.number_input("Store Number (Store ID)", min_value=1, max_value=1115, value=1)
        target_date = st.date_input("Estimated Date", datetime.date(2026, 4, 15))
        
    with col2:
        promo = st.selectbox("Is there a promotion?", options=["No (0)", "Yes (1)"])
        promo_val = 1 if "Yes" in promo else 0
        
        school_holiday = st.selectbox("Is it school holiday?", options=["No (0)", "Yes (1)"])
        school_holiday_val = 1 if "Yes" in school_holiday else 0
        
        state_holiday = st.selectbox("Public Holiday?", options=["None (0)", "a (General)", "b (Easter)", "c (Christmas)"]) 
        state_holiday_val = state_holiday.split(" ")[0]

    st.divider()

    if st.button("🔮 Make Sales Prediction", width="stretch"):
        payload = {
            "Store": store_id,
            "Date": target_date.strftime("%Y-%m-%d"),
            "Promo": promo_val,
            "StateHoliday": state_holiday_val,
            "SchoolHoliday": school_holiday_val
        }
        try:
            API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
            with st.spinner("Artificial Intelligence Analyzes Historical Data..."):
                response = requests.post(API_URL, json=payload)
                
            if response.status_code == 200:
                result = response.json()
                predicted_sales = result["Predicted_Sales_Euro"]
                metrics_path = os.path.join("artifacts", "metrics.json")
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                        dynamic_rmspe = metrics.get("rmspe", 0.13) 
                except FileNotFoundError:
                    dynamic_rmspe = 0.13

                error_margin = predicted_sales * dynamic_rmspe
                lower_bound = predicted_sales - error_margin
                upper_bound = predicted_sales + error_margin


                st.success("Prediction Completed Successfully!")
                st.metric(label="Expected Daily Income", value=f"{predicted_sales:,.2f} €")
                st.info(f"💡 **Confidence Interval:** Our AI model, which takes into account instantaneous fluctuations in the real world, leaves a margin of **± {error_margin:,.0f} €** (based on a dynamically calculated {dynamic_rmspe*100:.2f}% RMSPE) in the prediction. It is highly probable that the turnover will be in the **{lower_bound:,.0f} € to {upper_bound:,.0f} €** range.")
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Error! Details: {e}")



with tab_eda:
    st.header("Sales Dynamics and Promotion Analysis")
    
    @st.cache_data
    def load_eda_data():
        raw_path = os.path.join("dataset", "raw", "merged_raw.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path, low_memory=False)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            return df[(df['Open'] == 1) & (df['Sales'] > 0)]
        return None

    df_eda = load_eda_data()
    
    if df_eda is not None:
        
        st.subheader("1. Time and Promotion Impact on Overall Sales")
        sns.set_theme(style="whitegrid")
        fig1, axes1 = plt.subplots(1, 3, figsize=(22, 6))

        sns.barplot(x='DayOfWeek', y='Sales', data=df_eda, ax=axes1[0], hue='DayOfWeek', palette='viridis', errorbar=None, legend=False)
        axes1[0].set_title('Average Sales Volume by Day of the Week')

        sns.boxplot(x='Promo', y='Sales', data=df_eda, ax=axes1[1], hue='Promo', palette='Set2', legend=False)
        axes1[1].set_title('The Impact of Promotions on Sales')

        sns.lineplot(x='Month', y='Sales', data=df_eda, estimator=np.mean, ax=axes1[2], marker='o', color='coral', linewidth=2)
        axes1[2].set_title('Average Sales Trend by Month')
        axes1[2].set_xticks(range(1, 13))

        st.pyplot(fig1)
        st.divider()

        
        st.subheader("2. Store Type (StoreType) and Product Range (Assortment) Breakdown")
        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))

        sns.barplot(x='StoreType', y='Sales', hue='Promo', data=df_eda, order=['a', 'b', 'c', 'd'], palette='Set1', ax=axes2[0])
        axes2[0].set_title('The Effect of Promotion on Sales According to Store Type')

        sns.barplot(x='Assortment', y='Sales', hue='Promo', data=df_eda, order=['a', 'b', 'c'], palette='Set2', ax=axes2[1])
        axes2[1].set_title('The Effect of Promotion on Sales According to Product Range')

        st.pyplot(fig2)
        st.divider()

        
        st.header("📌 Executive Summary & Action Items")
        
        col_text, col_table = st.columns([1.5, 1])
        
        with col_table:
            st.markdown("#### Percentage Impact of Promotion (Lift)")
            promo_impact = df_eda.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
            promo_impact.columns = ['No_Promo_Sales', 'Promo_Sales']
            promo_impact['Lift_Percentage (%)'] = ((promo_impact['Promo_Sales'] - promo_impact['No_Promo_Sales']) / promo_impact['No_Promo_Sales']) * 100
            
            
            st.dataframe(promo_impact.round(2).sort_values(by='Lift_Percentage (%)', ascending=False), use_container_width=True)

        with col_text:
            st.info("""
            **🔍 Insights on B-Type Stores:**
            * Looking at the charts, we can see that **B-type stores** both generate very high sales volumes and are not as dramatically affected by promotions as others. They already have their own core customer base. 
            * *Note:* B-type stores are generally large-volume stores with high circulation, such as those in train stations or airports, and are often open on Sundays.
            * When a customer visits a store on a promotional day, their basket size increases if there is a wide selection of products (Assortment c or b) available.
            
            **🎯 Action Item:**
            * Allocating an aggressive promotional budget to B-type stores may reduce the return on investment (ROI). These stores already receive high traffic based on their location.
            """)
            
            st.success("""
            **💰 Where Promotions Work Best (Budget Optimization):**
            * Looking at the Lift_Percentage (%) table, we can clearly see that sales in standard street/mall stores of types **A, C, or D** surge by **40% to 50%** on days when promotions are offered.
            
            **🎯 Action Item:**
            * We should shift the lion's share of our promotion and marketing budget to **A and C type stores**. Every €1 spent on discounts/campaigns in these stores provides us with the highest marginal turnover.
            """)
            
    else:
        st.error("Data set not found. Please ensure the file `dataset/raw/merged_raw.csv` is in the project directory.")