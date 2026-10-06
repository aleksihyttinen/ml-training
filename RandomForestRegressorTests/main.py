import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../data.csv')
df.head()
# Start and End to dates
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])

# Sort from oldest to newest
df = df.sort_values("Start").reset_index(drop=True)

# Convert XX:XX formatted Duration to minutes
df["Duration_minutes"] = (
    pd.to_numeric(df["Duration"].str.split(":").str[0], errors="coerce") * 60
    + pd.to_numeric(df["Duration"].str.split(":").str[1], errors="coerce")
)

# Sort by Start and End, drop duplicates with missing ends (so miss inputs to app) and reset index
df = df.sort_values(
    ["Start", "End"],
    na_position="last"
)

df = df.drop_duplicates(
    subset="Start",
    keep="first"
)

df = df.reset_index(drop=True)

# Calculate minutes until the next feed
df["minutes_until_next_feed"] = (
    df["Start"].shift(-1) - df["Start"]
).dt.total_seconds() / 60


# Calcuate mean
average_interval = df["minutes_until_next_feed"].mean()

# Split data to training and testing
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# Calculate baseline prediction from training data
train_target = train["minutes_until_next_feed"].dropna()

baseline_prediction = train_target.mean()

# Baseline prediction: 159.75161987041037
print("Baseline prediction:", baseline_prediction)

# Evaluate baseline prediction on the test set
test = test.dropna(subset=["minutes_until_next_feed"]).copy()

test["baseline_prediction"] = baseline_prediction

test["baseline_error"] = abs(
    test["minutes_until_next_feed"] - test["baseline_prediction"]
)

mae = test["baseline_error"].mean()

# Baseline MAE: 63.96687012865056
print("Baseline MAE:", mae)


# Calculate minutes since the previous feed
df["previous_interval"] = (
    df["Start"] - df["Start"].shift(1)
).dt.total_seconds() / 60


# Add time of day info
df["hour"] = df["Start"].dt.hour
df["minute"] = df["Start"].dt.minute
df["day_of_week"] = df["Start"].dt.dayofweek

# Create df for modeling

model_df = df.dropna(
    subset=[
        "previous_interval",
        "minutes_until_next_feed"
    ]
).copy()


X = model_df[
    [
        "previous_interval",
        "hour",
        "minute",
        "day_of_week"
    ]
]

y = model_df["minutes_until_next_feed"]


# Split data into training and testing sets for modeling
split_index = int(len(model_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the model

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

# Random Forest MAE: 56.367370689655175
print("Random Forest MAE:", mae)

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)


# Create "around the clock" time features
import numpy as np

minutes_since_midnight = (
    df["Start"].dt.hour * 60
    + df["Start"].dt.minute
)


df["time_sin"] = np.sin(
    2 * np.pi * minutes_since_midnight / 1440
)
df["time_cos"] = np.cos(
    2 * np.pi * minutes_since_midnight / 1440
)

model_df = df.dropna(
    subset=[
        "previous_interval",
        "minutes_until_next_feed"
    ]
).copy()

# Create X with the new time features
X = model_df[
    [
        "previous_interval",
        "time_sin",
        "time_cos",
        "day_of_week"
    ]
]

y = model_df["minutes_until_next_feed"]

split_index = int(len(model_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the model with the new features
model_sin_cos = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_sin_cos.fit(X_train, y_train)

y_pred = model_sin_cos.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

# Random Forest with Time Features MAE: 56.425646551724135
print("Random Forest with Time Features MAE:", mae)

feature_importance = pd.Series(
    model_sin_cos.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

# Add recent feeding history feature

df["interval_2_back"] = df["previous_interval"].shift(1)
df["interval_3_back"] = df["previous_interval"].shift(2)

model_df = df.dropna(
    subset=[
        "previous_interval",
        "interval_2_back",
        "interval_3_back",
        "minutes_until_next_feed"
    ]
).copy()

# Create X with the recent feeding history features
X = model_df[
    [
        "previous_interval",
        "interval_2_back",
        "interval_3_back",
        "hour",
        "minute",
        "day_of_week"
    ]
]

y = model_df["minutes_until_next_feed"]

split_index = int(len(model_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the model with the recent feeding history features
model_history = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_history.fit(X_train, y_train)

y_pred_history = model_history.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_history)

# Random Forest with Recent Feeding History MAE: 48.85921739130435
print("Random Forest with Recent Feeding History MAE:", mae)

feature_importance = pd.Series(
    model_history.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

# Evaluate the model with recent feeding history features
results = X_test.copy()

results["actual_minutes"] = y_test
results["predicted_minutes"] = y_pred_history

results["error_minutes"] = (
    results["predicted_minutes"]
    - results["actual_minutes"]
)

results["absolute_error"] = abs(
    results["error_minutes"]
)


within_15 = (results["absolute_error"] <= 15).mean() * 100
within_30 = (results["absolute_error"] <= 30).mean() * 100
within_60 = (results["absolute_error"] <= 60).mean() * 100
within_90 = (results["absolute_error"] <= 90).mean() * 100


# Try predicting from feed end -> next ones start. Old model was from Start -> Start

df["minutes_until_next_feed_from_end"] = (
    df["Start"].shift(-1) - df["End"]
).dt.total_seconds() / 60


df["previous_gap"] = (
    df["Start"] - df["End"].shift(1)
).dt.total_seconds() / 60

df["gap_2_back"] = df["previous_gap"].shift(1)

df["gap_3_back"] = df["previous_gap"].shift(2)


model_end_df = df.dropna(
    subset=[
        "Duration_minutes",
        "previous_gap",
        "gap_2_back",
        "gap_3_back",
        "minutes_until_next_feed_from_end"
    ]
).copy()


X_end = model_end_df[
    [
        "Duration_minutes",
        "previous_gap",
        "gap_2_back",
        "gap_3_back",
        "hour",
        "minute",
        "day_of_week"
    ]
]


y_end = model_end_df[
    "minutes_until_next_feed_from_end"
]


split_index = int(len(model_end_df) * 0.8)

X_end_train = X_end.iloc[:split_index]
X_end_test = X_end.iloc[split_index:]

y_end_train = y_end.iloc[:split_index]
y_end_test = y_end.iloc[split_index:]


model_end = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


model_end.fit(
    X_end_train,
    y_end_train
)


y_pred_end = model_end.predict(X_end_test)


mae_end = mean_absolute_error(
    y_end_test,
    y_pred_end
)

# Random Forest predicting from feed end MAE: 50.172168674698796
print("Random Forest predicting from feed end MAE:", mae_end)

feature_importance = pd.Series(
    model_end.feature_importances_,
    index=X_end.columns
).sort_values(ascending=False)

print(feature_importance)





# Try the same but without duration


df["minutes_until_next_feed_from_end"] = (
    df["Start"].shift(-1) - df["End"]
).dt.total_seconds() / 60


df["previous_gap"] = (
    df["Start"] - df["End"].shift(1)
).dt.total_seconds() / 60

df["gap_2_back"] = df["previous_gap"].shift(1)

df["gap_3_back"] = df["previous_gap"].shift(2)

model_end_no_duration_df = df.dropna(
    subset=[
        "previous_gap",
        "gap_2_back",
        "gap_3_back",
        "minutes_until_next_feed_from_end"
    ]
).copy()


X_end_no_duration = model_end_no_duration_df[
    [
        "previous_gap",
        "gap_2_back",
        "gap_3_back",
        "hour",
        "minute",
        "day_of_week"
    ]
]


y_end_no_duration = model_end_no_duration_df[
    "minutes_until_next_feed_from_end"
]


split_index = int(len(model_end_no_duration_df) * 0.8)

X_end_train = X_end_no_duration.iloc[:split_index]
X_end_test = X_end_no_duration.iloc[split_index:]

y_end_train = y_end_no_duration.iloc[:split_index]
y_end_test = y_end_no_duration.iloc[split_index:]


model_end_no_duration = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


model_end_no_duration.fit(
    X_end_train,
    y_end_train
)


y_pred_end = model_end_no_duration.predict(X_end_test)


mae_end = mean_absolute_error(
    y_end_test,
    y_pred_end
)

# Random Forest predicting from feed end without duration MAE: 49.11698795180723
print("Random Forest predicting from feed end without duration MAE:", mae_end)

feature_importance = pd.Series(
    model_end_no_duration.feature_importances_,
    index=X_end_no_duration.columns
).sort_values(ascending=False)

print(feature_importance)


# # Inspect the worst predictions for the model
# results_end = X_end_no_duration.iloc[split_index:].copy()

# results_end["actual_minutes"] = y_end_test
# results_end["predicted_minutes"] = y_pred_end

# results_end["absolute_error"] = abs(
#     results_end["predicted_minutes"]
#     - results_end["actual_minutes"]
# )

# # Inspect the absolute error grouped by hour
# print(
#     results_end.groupby("hour")["absolute_error"].agg(
#         ["count", "mean"]
#     )
# )

# # Inspect the 10 worst predictions for the model by absolute error
# print(
#     results_end[
#         [
#             "actual_minutes",
#             "predicted_minutes",
#             "absolute_error"
#         ]
#     ]
#     .sort_values("absolute_error", ascending=False)
#     .head(10)
# )


# bad_predictions = (
#     results_end
#     .sort_values("absolute_error", ascending=False)
#     .head(10)
# )

# # Print the top 10 worst predictions with detailed feature information
# print(
#     bad_predictions[
#         [
#             "previous_gap",
#             "gap_2_back",
#             "gap_3_back",
#             "hour",
#             "minute",
#             "actual_minutes",
#             "predicted_minutes",
#             "absolute_error"
#         ]
#     ]
# )

# Try mean and median of recent gaps as features
df["recent_gap_mean"] = (
    df[
        ["previous_gap", "gap_2_back", "gap_3_back"]
    ].mean(axis=1)
)

df["recent_gap_median"] = (
    df[
        ["previous_gap", "gap_2_back", "gap_3_back"]
    ].median(axis=1)
)

model_end_med_median_df = df.dropna(
    subset=[
        "previous_gap",
        "gap_2_back",
        "gap_3_back",
        "recent_gap_mean",
        "recent_gap_median",
        "minutes_until_next_feed_from_end"
    ]
).copy()

X_end = model_end_med_median_df[
    [
        "previous_gap",
        "gap_2_back",
        "gap_3_back",
        "recent_gap_mean",
        "recent_gap_median",
        "hour",
        "minute",
        "day_of_week"
    ]
]

y_end = model_end_med_median_df["minutes_until_next_feed_from_end"]

split_index = int(len(model_end_med_median_df) * 0.8)

X_end_train = X_end.iloc[:split_index]
X_end_test = X_end.iloc[split_index:]

y_end_train = y_end.iloc[:split_index]
y_end_test = y_end.iloc[split_index:]

model_end_features = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model_end_features.fit(
    X_end_train,
    y_end_train
)

y_pred_end_features = model_end_features.predict(
    X_end_test
)

mae_end_features = mean_absolute_error(
    y_end_test,
    y_pred_end_features
)

# Random Forest with recent gap summaries MAE: 48.401204819277105
print(
    "Random Forest with recent gap summaries MAE:",
    mae_end_features
)

feature_importance = pd.Series(
    model_end_features.feature_importances_,
    index=X_end.columns
).sort_values(ascending=False)

print(feature_importance)

errors = abs(
    y_pred_end_features - y_end_test
)

# Within 15 min: 0.2891566265060241
# Within 30 min: 0.40963855421686746
# Within 60 min: 0.7228915662650602
# Within 90 min: 0.8313253012048193
print("Within 15 min:", (errors <= 15).mean())
print("Within 30 min:", (errors <= 30).mean())
print("Within 60 min:", (errors <= 60).mean())
print("Within 90 min:", (errors <= 90).mean())