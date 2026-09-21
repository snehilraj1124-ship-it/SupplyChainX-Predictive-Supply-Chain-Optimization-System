import joblib
import pandas as pd

from .features import engineer_features


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "models/demand_forecasting_model.joblib"

FEATURES = [
    "sku",
    "category",
    "unit_price",
    "lead_time_days",
    "promotion",
    "inventory",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_28",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "stock_cover_days",
    "inventory_demand_ratio"
]


# =========================================================
# Load Model
# =========================================================

def load_model():
    """
    Load the trained SupplyChainX forecasting model.
    """

    return joblib.load(
        MODEL_PATH
    )


# =========================================================
# Forecast Function
# =========================================================

def forecast_next_row(
    historical_data
):
    """
    Generate a demand forecast using
    historical SKU observations.

    Parameters
    ----------
    historical_data : list[dict]
        Historical observations for one or
        more SKUs.

    Returns
    -------
    float
        Forecasted demand for the latest
        observation.
    """

    if not historical_data:
        raise ValueError(
            "Historical data cannot be empty."
        )

    df = pd.DataFrame(
        historical_data
    )

    required_columns = [
        "date",
        "sku",
        "category",
        "unit_price",
        "lead_time_days",
        "promotion",
        "inventory",
        "demand"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Feature engineering
    df = engineer_features(
        df
    )

    # Remove rows without sufficient
    # historical information.
    df = df.dropna().reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError(
            "Insufficient historical observations. "
            "At least 29 observations are recommended "
            "for lag and rolling features."
        )

    # Load trained model
    model = load_model()

    # Generate forecast
    prediction = model.predict(
        df[FEATURES]
    )

    prediction = max(
        0,
        float(prediction[-1])
    )

    return prediction


# =========================================================
# Example Usage
# =========================================================

if __name__ == "__main__":

    data = pd.read_csv(
        "data/supply_chain_data.csv"
    )

    # Select one SKU
    sample = (
        data[
            data["sku"] == "SKU-001"
        ]
        .sort_values("date")
        .tail(40)
    )

    forecast = forecast_next_row(
        sample.to_dict(
            orient="records"
        )
    )

    print(
        f"Forecasted demand: "
        f"{forecast:.2f}"
    )
