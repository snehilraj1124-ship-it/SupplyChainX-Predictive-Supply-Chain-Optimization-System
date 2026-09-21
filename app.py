import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="SupplyChainX",
    page_icon="📦",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "data/supply_chain_data.csv",
        parse_dates=["date"]
    )

df = load_data()

with open("reports/metrics.json", "r") as f:
    metrics = json.load(f)

# -----------------------------
# Header
# -----------------------------
st.title("📦 SupplyChainX")
st.subheader("Predictive Supply Chain Optimization & Demand Intelligence")

st.markdown(
    """
    SupplyChainX uses machine learning and operational analytics
    to forecast demand, analyze inventory coverage and identify
    potential stock-out risks.
    """
)

st.divider()

# -----------------------------
# KPI Section
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total SKUs",
    f"{df['sku'].nunique()}"
)

col2.metric(
    "Historical Records",
    f"{len(df):,}"
)

col3.metric(
    "Forecast MAE",
    f"{metrics['MAE']:.2f}"
)

col4.metric(
    "Model R²",
    f"{metrics['R2']:.3f}"
)

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Demand Analytics",
        "🚨 Inventory Risk",
        "🤖 Model Performance"
    ]
)

# =========================================================
# TAB 1 — DEMAND ANALYTICS
# =========================================================

with tab1:

    st.header("📈 Demand Analytics")

    selected_sku = st.selectbox(
        "Select SKU",
        sorted(df["sku"].unique())
    )

    sku_data = df[
        df["sku"] == selected_sku
    ].copy()

    # Demand trend
    fig_demand = px.line(
        sku_data,
        x="date",
        y="demand",
        title=f"Demand Trend — {selected_sku}",
        labels={
            "date": "Date",
            "demand": "Demand"
        }
    )

    fig_demand.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_demand,
        use_container_width=True
    )

    # Two-column analytics
    left, right = st.columns(2)

    with left:

        category_demand = (
            df.groupby("category", as_index=False)
            ["demand"]
            .mean()
            .sort_values("demand", ascending=False)
        )

        fig_category = px.bar(
            category_demand,
            x="category",
            y="demand",
            title="Average Demand by Category",
            labels={
                "category": "Category",
                "demand": "Average Demand"
            }
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    with right:

        monthly = (
            df.groupby(
                df["date"].dt.month,
                as_index=False
            )["demand"]
            .mean()
        )

        monthly.columns = [
            "month",
            "average_demand"
        ]

        fig_month = px.line(
            monthly,
            x="month",
            y="average_demand",
            markers=True,
            title="Average Demand by Month"
        )

        st.plotly_chart(
            fig_month,
            use_container_width=True
        )


# =========================================================
# TAB 2 — INVENTORY RISK
# =========================================================

with tab2:

    st.header("🚨 Inventory Risk Analysis")

    inventory = (
        df.groupby("sku")
        .agg(
            average_demand=("demand", "mean"),
            average_inventory=("inventory", "mean"),
            lead_time_days=("lead_time_days", "mean")
        )
        .reset_index()
    )

    inventory["stock_cover_days"] = (
        inventory["average_inventory"]
        / (inventory["average_demand"] + 1)
    )

    inventory["risk"] = (
        inventory["stock_cover_days"]
        < inventory["lead_time_days"]
    )

    inventory["risk_status"] = inventory["risk"].map(
        {
            True: "HIGH RISK",
            False: "NORMAL"
        }
    )

    high_risk_count = int(
        inventory["risk"].sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "SKUs Analyzed",
        len(inventory)
    )

    c2.metric(
        "High-Risk SKUs",
        high_risk_count
    )

    c3.metric(
        "Average Stock Coverage",
        f"{inventory['stock_cover_days'].mean():.1f} days"
    )

    st.divider()

    st.subheader("SKU Inventory Risk Table")

    display_data = inventory.sort_values(
        "stock_cover_days"
    ).copy()

    st.dataframe(
        display_data[
            [
                "sku",
                "average_demand",
                "average_inventory",
                "lead_time_days",
                "stock_cover_days",
                "risk_status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Inventory vs Demand")

    fig_inventory = px.scatter(
        inventory,
        x="average_demand",
        y="average_inventory",
        color="risk_status",
        hover_name="sku",
        size="lead_time_days",
        title="Inventory Level vs Average Demand",
        labels={
            "average_demand": "Average Daily Demand",
            "average_inventory": "Average Inventory",
            "risk_status": "Risk Status"
        }
    )

    st.plotly_chart(
        fig_inventory,
        use_container_width=True
    )


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================

with tab3:

    st.header("🤖 Forecasting Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{metrics['MAE']:.2f}"
    )

    c2.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}"
    )

    c3.metric(
        "R² Score",
        f"{metrics['R2']:.3f}"
    )

    st.divider()

    st.subheader("Evaluation Metrics")

    metric_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Absolute Error",
                "Root Mean Squared Error",
                "R² Score"
            ],
            "Value": [
                metrics["MAE"],
                metrics["RMSE"],
                metrics["R2"]
            ]
        }
    )

    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        The model uses a chronological holdout period rather than
        random splitting. This better represents how a real demand
        forecasting system would be evaluated on future observations.
        """
    )

    st.subheader("Machine Learning Pipeline")

    st.code(
        """
Historical Demand
        ↓
Data Cleaning
        ↓
Lag Features
        ↓
Rolling Features
        ↓
Calendar Features
        ↓
Inventory Features
        ↓
Random Forest Regressor
        ↓
Demand Forecast
        ↓
Inventory Risk Analysis
        """,
        language="text"
    )

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "SupplyChainX • Machine Learning + Supply Chain Analytics • "
    "Synthetic dataset for portfolio demonstration"
)
