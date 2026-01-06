import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

df = pd.read_csv("data/data_train.csv")
df.columns = df.columns.str.strip()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE")

# 2. Feature engineering
df["dayofweek"] = df["DATE"].dt.dayofweek
df["week"] = df["DATE"].dt.isocalendar().week.astype(int)
df["month"] = df["DATE"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

df["cpm_log"] = np.log1p(df["CPM"])
df["cpm_sq"] = df["CPM"] ** 2

df["channel_history_count"] = df.groupby("CHANNEL_NAME").cumcount().clip(lower=1)

global_mean_views = df["VIEWS"].mean()
global_mean_cpm = df["CPM"].mean()

df["channel_mean_views"] = (
    df.groupby("CHANNEL_NAME")["VIEWS"]
      .expanding().mean().shift(1)
      .reset_index(level=0, drop=True)
      .fillna(global_mean_views)
)

df["channel_median_views"] = (
    df.groupby("CHANNEL_NAME")["VIEWS"]
      .expanding().median().shift(1)
      .reset_index(level=0, drop=True)
      .fillna(global_mean_views)
)

df["cpm_vs_channel_mean"] = df["CPM"] / global_mean_cpm

# 3. Train / valid split
split_date = df["DATE"].quantile(0.8)
train = df[df["DATE"] <= split_date]
valid = df[df["DATE"] > split_date]

features = [
    "CPM", "cpm_log", "cpm_sq",
    "dayofweek", "week", "month", "is_weekend",
    "channel_mean_views", "channel_median_views",
    "channel_history_count", "cpm_vs_channel_mean",
    "CHANNEL_NAME"
]

X_train = train[features]
y_train = np.log1p(train["VIEWS"])

# 4. Model
model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.05,
    depth=8,
    loss_function="MAE",
    random_seed=42,
    verbose=200
)

model.fit(X_train, y_train, cat_features=["CHANNEL_NAME"])



# 6. Save channel stats
channel_stats = {
    "__global__": {
        "mean": global_mean_views,
        "median": df["VIEWS"].median(),
        "count": 1,
        "mean_cpm": global_mean_cpm
    }
}

for ch, g in df.groupby("CHANNEL_NAME"):
    channel_stats[ch] = {
        "mean": g["VIEWS"].mean(),
        "median": g["VIEWS"].median(),
        "count": len(g),
        "mean_cpm": g["CPM"].mean()
    }
