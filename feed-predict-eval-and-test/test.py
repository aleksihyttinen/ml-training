import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
)

from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------- load
raw = pd.read_csv(
    "data.csv",
    usecols=["Type", "Start", "End", "Duration"]
)

raw["Start"] = pd.to_datetime(raw["Start"])
raw["End"] = pd.to_datetime(raw["End"])


# ---------------------------------------------------------------- feeds
df = raw[raw["Type"] == "Feed"].dropna(subset=["Start"]).copy()

df = (
    df.sort_values(["Start", "End"], na_position="last")
    .drop_duplicates(subset="Start", keep="first")
    .reset_index(drop=True)
)


# Target: minutes until next feed
df["minutes_until_next_feed"] = (
    df["Start"].shift(-1) - df["Start"]
).dt.total_seconds() / 60


# Feeding history
df["previous_interval"] = (
    df["Start"] - df["Start"].shift(1)
).dt.total_seconds() / 60

df["interval_2_back"] = df["previous_interval"].shift(1)
df["interval_3_back"] = df["previous_interval"].shift(2)


# ---------------------------------------------------------------- time features
df["hour"] = df["Start"].dt.hour
df["minute"] = df["Start"].dt.minute
df["day_of_week"] = df["Start"].dt.dayofweek

df["time_minutes"] = (
    df["Start"].dt.hour * 60 +
    df["Start"].dt.minute
)

# Cyclical representation of time of day
df["time_sin"] = np.sin(
    2 * np.pi * df["time_minutes"] / 1440
)

df["time_cos"] = np.cos(
    2 * np.pi * df["time_minutes"] / 1440
)



# ---------------------------------------------------------------- model data

FEATURES = [
    "previous_interval",
    "interval_2_back",
    "interval_3_back",
    "time_sin",
    "time_cos",
    "day_of_week",
]


model_df = df.dropna(
    subset=FEATURES + ["minutes_until_next_feed"]
).reset_index(drop=True)




model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=3,
    min_samples_leaf=10,
    loss="absolute_error",
    random_state=42,
)

 

n = len(model_df)

split = int(n * 0.80)

train_df = model_df.iloc[:split].copy()
holdout_df = model_df.iloc[split:].copy()

print("total rows:", n)
print("training rows:", len(train_df))
print("holdout rows:", len(holdout_df))

print("\nTraining period:")
print(train_df["Start"].iloc[0])
print("to")
print(train_df["Start"].iloc[-1])

print("\nHoldout period:")
print(holdout_df["Start"].iloc[0])
print("to")
print(holdout_df["Start"].iloc[-1])

# ---------------------------------------------------------------- final holdout test

X_train = train_df[FEATURES]
y_train = train_df["minutes_until_next_feed"]

X_holdout = holdout_df[FEATURES]
y_holdout = holdout_df["minutes_until_next_feed"]


# Train the selected model on ALL of the first 80%
model.fit(X_train, y_train)


# Predict the final 20%
predictions = model.predict(X_holdout)


# Calculate MAE
holdout_mae = mean_absolute_error(
    y_holdout,
    predictions
)


print("\n" + "=" * 60)
print("FINAL FUTURE HOLDOUT")
print("=" * 60)
print(f"MAE: {holdout_mae:.1f} minutes")


# ---------------------------------------------------------------- inspect predictions

results = holdout_df[
    ["Start", "minutes_until_next_feed"]
].copy()

results["prediction"] = predictions

results["absolute_error"] = (
    results["minutes_until_next_feed"]
    - results["prediction"]
).abs()

print("\nHoldout predictions:")
print(results.to_string(index=False))

results["actual_bin"] = pd.cut(
    results["minutes_until_next_feed"],
    bins=[0, 60, 120, 180, 240, 300, np.inf],
    labels=[
        "<1h",
        "1–2h",
        "2–3h",
        "3–4h",
        "4–5h",
        "5h+",
    ],
)

print(
    results.groupby("actual_bin", observed=False)["absolute_error"]
    .agg(["count", "mean", "median"])
)

