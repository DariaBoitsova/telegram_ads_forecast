import numpy as np
import pandas as pd

def build_features(CPM, CHANNEL_NAME, DATE, channel_stats):
    DATE = pd.to_datetime(DATE)
    stats = channel_stats.get(CHANNEL_NAME, channel_stats["__global__"])

    features = {
        "CPM": CPM,
        "cpm_log": np.log1p(CPM),
        "cpm_sq": CPM ** 2,
        "dayofweek": DATE.dayofweek,
        "week": DATE.isocalendar().week,
        "month": DATE.month,
        "is_weekend": int(DATE.dayofweek in [5, 6]),
        "channel_mean_views": stats["mean"],
        "channel_median_views": stats["median"],
        "channel_history_count": stats["count"],
        "cpm_vs_channel_mean": CPM / stats["mean_cpm"],
        "CHANNEL_NAME": CHANNEL_NAME
    }

    return pd.DataFrame([features])
