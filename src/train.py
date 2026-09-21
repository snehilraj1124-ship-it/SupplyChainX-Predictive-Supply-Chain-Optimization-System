import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from features import engineer_features


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "data/supply_chain_data.csv"
MODEL_PATH = "models/demand_forecasting_model.joblib"
METRICS_PATH = "reports/metrics.json"


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
# Load & Prepare Data
# =========================================================

print("\nLoading supply-chain dataset...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

print(
    f"Loaded {len(df):,} records "
    f"across {df['sku'].nunique()} SKUs."
)


# =========================================================
# Feature Engineering
# =========================================================

print("\nEngineering forecasting features...")

df = engineer_features(df)

# Remove rows where lag/rolling values
# are unavailable.
df = df.dropna().reset_index(drop=True)

print(
    f"Usable records after feature engineering: "
    f"{len(df):,}"
)


# =========================================================
# Chronological Train/Test Split
# =========================================================

# Last 60 days are kept as the test period.
cutoff_date = (
    df["date"].max()
    - pd.Timedelta(days=60)
)

train = df[
    df["date"] < cutoff_date
].copy()

test = df[
    df["date"] >= cutoff_date
].copy()


print("\nChronological split:")
print(f"Training records : {len(train):,}")
print(f"Testing records  : {len(test):,}")


X_train = train[FEATURES]
y_train = train["demand"]

X_test = test[FEATURES]
y_test = test["demand"]


# =========================================================
# Preprocessing
# =========================================================

categorical_features = [
    "sku",
    "category"
]

numeric_features = [
    feature
    for feature in FEATURES
    if feature not in categorical_features
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# Machine Learning Model
# =========================================================

print("\nTraining Random Forest forecasting model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# =========================================================
# Model Training
# =========================================================

pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# Forecast
# =========================================================

print("\nGenerating demand forecasts...")

predictions = pipeline.predict(
    X_test
)


# Prevent negative demand predictions
predictions = predictions.clip(
    min=0
)


# =========================================================
# Model Evaluation
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


metrics = {
    "MAE": round(float(mae), 4),
    "RMSE": round(float(rmse), 4),
    "R2": round(float(r2), 4),
    "training_records": int(len(train)),
    "testing_records": int(len(test)),
    "number_of_skus": int(df["sku"].nunique()),
    "test_period_days": 60
}


# =========================================================
# Save Model
# =========================================================

print("\nSaving trained model...")

joblib.dump(
    pipeline,
    MODEL_PATH
)


# =========================================================
# Save Metrics
# =========================================================

with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# =========================================================
# Results
# =========================================================

print("\n" + "=" * 50)
print("SUPPLYCHAINX MODEL RESULTS")
print("=" * 50)

print(
    f"MAE  : {metrics['MAE']}"
)

print(
    f"RMSE : {metrics['RMSE']}"
)

print(
    f"R²   : {metrics['R2']}"
)

print("=" * 50)

print(
    "\nModel saved to:",
    MODEL_PATH
)

print(
    "Metrics saved to:",
    METRICS_PATH
)

print("\nTraining completed successfully.")
