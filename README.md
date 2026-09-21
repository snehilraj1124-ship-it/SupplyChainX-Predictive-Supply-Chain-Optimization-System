# 📦 SupplyChainX

## Predictive Supply Chain Optimization & Demand Intelligence System

SupplyChainX is an end-to-end machine learning platform designed to forecast product demand, analyze inventory levels, identify stock-out risks, and support data-driven supply chain planning.

The project combines **Machine Learning, Time-Series Feature Engineering, Data Analytics, Inventory Intelligence, and Interactive Business Intelligence** into a single portfolio-ready solution.

---

## 🚀 Key Features

- 📈 Product-level demand forecasting
- 📦 Inventory risk analysis
- 🔮 Machine learning-based demand prediction
- ⏱️ Lead-time-aware inventory analysis
- 📊 Historical demand trend analysis
- 🚨 Stock-out risk identification
- 📉 Forecast performance evaluation
- 🖥️ Interactive Streamlit dashboard
- 🧠 Lag and rolling-window feature engineering
- 📋 Business-oriented supply chain KPIs
- 🌐 Flask REST API for forecasting

---

## 🧠 Machine Learning Approach

The primary forecasting model is a **Random Forest Regressor**.

The model uses historical and operational features including:

- Previous-day demand
- 7-day demand lag
- 14-day demand lag
- 28-day demand lag
- 7-day rolling demand
- 28-day rolling demand
- Current inventory
- Supplier lead time
- Product category
- Unit price
- Promotion activity
- Day of week
- Month
- Day of year
- Estimated stock coverage

A chronological holdout period is used instead of random splitting to better simulate real-world forecasting.

---

## 📊 Model Performance

The current synthetic-data experiment achieved:

| Metric | Result |
|---|---:|
| MAE | 4.12 |
| RMSE | 7.05 |
| R² | 0.973 |

> These results are based on synthetic portfolio data and are intended for educational and demonstration purposes.

---

## 🏗️ System Architecture

```text
Historical Sales & Inventory Data
              │
              ▼
       Data Preprocessing
              │
              ▼
       Feature Engineering
              │
              ▼
     Lag & Rolling Features
              │
              ▼
      Demand Forecasting
              │
        ┌─────┴─────┐
        ▼           ▼
 Demand Forecast  Inventory Risk
        │           │
        ▼           ▼
 Stock Planning  Stock-out Alerts
        │           │
        └─────┬─────┘
              ▼
      Business Dashboard
              │
       ┌──────┴──────┐
       ▼             ▼
   Streamlit      REST API
