import numpy as np
import pandas as pd

def build_features(cpm, channel, date, channel_stats):
    date = pd.to_datetime(date)
    stats = channel_stats.get(channel, channel_stats["__global__"])

    features = {
        "CPM": cpm,
        "cpm_log": np.log1p(cpm),
        "cpm_sq": cpm ** 2,
        "dayofweek": date.dayofweek,
        "week": date.isocalendar().week,
        "month": date.month,
        "is_weekend": int(date.dayofweek in [5, 6]),
        "channel_mean_views": stats["mean"],
        "channel_median_views": stats["median"],
        "channel_history_count": stats["count"],
        "cpm_vs_channel_mean": cpm / stats["mean_cpm"],
        "CHANNEL_NAME": channel
    }

    return pd.DataFrame([features])
