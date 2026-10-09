import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# EVALUATING THE BEST MODEL + FEATURES

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


# ---------------------------------------------------------------- sleeps
sleeps = raw[
    raw["Type"] == "Sleep"
].dropna(subset=["Start", "End"]).copy()

sleeps = sleeps.sort_values("End").reset_index(drop=True)

sleeps["sleep_duration"] = (
    (sleeps["End"] - sleeps["Start"])
    .dt.total_seconds() / 60
)


# Last sleep that ended before this feed started
df = pd.merge_asof(
    df.sort_values("Start"),
    sleeps[
        ["End", "sleep_duration"]
    ].rename(
        columns={
            "End": "last_sleep_end",
            "sleep_duration": "last_sleep_duration",
        }
    ),
    left_on="Start",
    right_on="last_sleep_end",
    direction="backward",
    tolerance=pd.Timedelta(hours=24),
)

df["minutes_since_wake"] = (
    df["Start"] - df["last_sleep_end"]
).dt.total_seconds() / 60


# ---------------------------------------------------------------- sleep in previous 6 hours
first_sleep = sleeps["Start"].min()


def sleep_in_window(t, hours=6):
    """
    Calculate total sleep occurring in the previous `hours`.

    Returns NaN if there isn't a complete history window yet.
    """

    if t < first_sleep + pd.Timedelta(hours=hours):
        return np.nan

    w0 = t - pd.Timedelta(hours=hours)

    s = sleeps["Start"].clip(
        lower=w0,
        upper=t
    )

    e = sleeps["End"].clip(
        lower=w0,
        upper=t
    )

    return (
        (e - s)
        .dt.total_seconds()
        .sum() / 60
    )


df["sleep_last_6h"] = df["Start"].apply(
    sleep_in_window
)


# ---------------------------------------------------------------- model data

BASE = [
    "previous_interval",
    "interval_2_back",
    "interval_3_back",
    "hour",
    "minute",
    "day_of_week",
]

SLEEP = [
    "minutes_since_wake",
    "last_sleep_duration",
    "sleep_last_6h",
]


# Only base features and target must exist.
# Sleep features are allowed to be missing.
model_df = df.dropna(
    subset=BASE + ["minutes_until_next_feed"]
).reset_index(drop=True)


# Missing sleep information -> -1
# This lets all rows be used by sklearn models.
model_df[SLEEP] = model_df[SLEEP].fillna(-1)


y = model_df["minutes_until_next_feed"].to_numpy()

# Rows with actual 6-hour sleep information
covered = (
    model_df["sleep_last_6h"] >= 0
).to_numpy()


print(
    f"rows: {len(model_df)}, "
    f"with sleep info: {covered.sum()}"
)


# ---------------------------------------------------------------- feature sets

feature_sets = {

    "history": [
        "previous_interval",
        "interval_2_back",
        "interval_3_back",
        "day_of_week",
    ],

    "history + time sin cos": [
        "previous_interval",
        "interval_2_back",
        "interval_3_back",
        "time_sin",
        "time_cos",
        "day_of_week",
    ],

    "history + time sin cos + sleep": [
        "previous_interval",
        "interval_2_back",
        "interval_3_back",
        "time_sin",
        "time_cos",
        "day_of_week",
        "minutes_since_wake",
        "last_sleep_duration",
        "sleep_last_6h",
    ],
}


# ---------------------------------------------------------------- models

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        loss="absolute_error",
        random_state=42,
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),
}

gb_models = {
    "GB 1": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=5,
        loss="absolute_error",
        random_state=42,
    ),

    "GB 2": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        loss="absolute_error",
        random_state=42,
    ),

    "GB 3": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        loss="absolute_error",
        random_state=42,
    ),

    "GB 4": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        loss="absolute_error",
        random_state=42,
    ),

    "GB 5": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        loss="absolute_error",
        random_state=42,
    ),

    "GB 6": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),
}

gb_models = {
    "A": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),

    "B": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),

    "C": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),

    "D": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),

    "E": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),

    "F": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),

    "G": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
    ),
}
# ---------------------------------------------------------------- evaluation

tscv = TimeSeriesSplit(n_splits=5)


def report(name, maes):

    if maes:
        print(
            f"{name}: "
            f"{np.mean(maes):.1f} ± {np.std(maes):.1f}  "
            f"folds={np.round(maes, 1)}"
        )

    else:
        print(
            f"{name}: "
            f"no folds with sleep-covered test rows"
        )


# ---------------------------------------------------------------- baseline

base_maes = []

for tr, te in tscv.split(model_df):

    # Only evaluate on rows with sleep information
    keep = covered[te]

    if keep.sum() == 0:
        continue

    # Mean of training targets
    pred = np.full(
        keep.sum(),
        y[tr].mean()
    )

    base_maes.append(
        mean_absolute_error(
            y[te][keep],
            pred
        )
    )


report("baseline", base_maes)


# ---------------------------------------------------------------- model comparison

results = []


for model_name, model_template in gb_models.items():

    for feature_name, feats in feature_sets.items():

        maes = []

        for tr, te in tscv.split(model_df):

            # Only evaluate on sleep-covered test rows
            keep = covered[te]

            if keep.sum() == 0:
                continue

            # Create a fresh model for every fold
            model = model_template.__class__(
                **model_template.get_params()
            )

            # Train only on past data
            model.fit(
                model_df.iloc[tr][feats],
                y[tr]
            )

            # Predict future data
            pred = model.predict(
                model_df.iloc[te][feats]
            )

            # Score only sleep-covered test rows
            mae = mean_absolute_error(
                y[te][keep],
                pred[keep]
            )

            maes.append(mae)

        if maes:

            mean_mae = np.mean(maes)
            std_mae = np.std(maes)

            results.append({
                "model": model_name,
                "features": feature_name,
                "mae": mean_mae,
                "std": std_mae,
                "folds": np.round(maes, 1),
            })

            print(
                f"{model_name:22s} | "
                f"{feature_name:30s} | "
                f"{mean_mae:.1f} ± {std_mae:.1f} | "
                f"folds={np.round(maes, 1)}"
            )


# ---------------------------------------------------------------- results table

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mae"
).reset_index(drop=True)


print("\n")
print("=" * 80)
print("RANKED RESULTS")
print("=" * 80)

for i, row in results_df.iterrows():

    print(
        f"{i + 1}. "
        f"{row['model']} + {row['features']}: "
        f"{row['mae']:.1f} ± {row['std']:.1f}"
    )


# ---------------------------------------------------------------- best model

best = results_df.iloc[0]

print("\n")
print("=" * 80)
print("BEST MODEL")
print("=" * 80)

print(f"Model:    {best['model']}")
print(f"Features: {best['features']}")
print(f"MAE:      {best['mae']:.1f} ± {best['std']:.1f}")
print(f"Folds:    {best['folds']}")

