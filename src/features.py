import pandas as pd


def engineer_features(df):
    """
    Create time-series and inventory features
    for SupplyChainX demand forecasting.
    """

    out = df.copy()

    # Ensure correct date format
    out["date"] = pd.to_datetime(out["date"])

    # Sort chronologically for each SKU
    out = out.sort_values(
        ["sku", "date"]
    ).reset_index(drop=True)

    # Group demand by SKU
    demand_group = out.groupby("sku")["demand"]

    # -----------------------------
    # Lag Features
    # -----------------------------

    out["lag_1"] = demand_group.shift(1)
    out["lag_7"] = demand_group.shift(7)
    out["lag_14"] = demand_group.shift(14)
    out["lag_28"] = demand_group.shift(28)

    # -----------------------------
    # Rolling Demand Features
    # -----------------------------

    out["rolling_7"] = (
        demand_group
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["rolling_28"] = (
        demand_group
        .shift(1)
        .rolling(28)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # Calendar Features
    # -----------------------------

    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear

    # Weekend indicator
    out["is_weekend"] = (
        out["dayofweek"] >= 5
    ).astype(int)

    # -----------------------------
    # Inventory Features
    # -----------------------------

    out["stock_cover_days"] = (
        out["inventory"] /
        (out["rolling_7"] + 1)
    )

    # Potential stock-out risk
    out["stockout_risk"] = (
        out["inventory"] <
        out["rolling_7"] *
        out["lead_time_days"]
    ).astype(int)

    # Inventory-to-demand ratio
    out["inventory_demand_ratio"] = (
        out["inventory"] /
        (out["rolling_28"] + 1)
    )

    return out
