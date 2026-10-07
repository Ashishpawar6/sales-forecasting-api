LAGS = (0, 1, 6, 13, 27)      # lag k = sales k days before the forecast origin (0 = origin day itself)
WINDOWS = (7, 28)             # rolling-mean windows ending at the forecast origin
MAX_HISTORY = max(max(LAGS) + 1, max(WINDOWS))  # days of history a prediction needs
DEFAULT_HORIZON = 14
KEEP_HISTORY_DAYS = 60        # per-series history stored with the model for serving
STRATEGIES = ("recursive", "direct", "combined")

CALENDAR_FEATURES = ["dow", "month", "day", "dayofyear", "weekofyear"]
LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLL_FEATURES = [f"roll_mean_{w}" for w in WINDOWS]
FEATURES = LAG_FEATURES + ROLL_FEATURES + CALENDAR_FEATURES + ["promo", "store", "item"]
CATEGORICAL = ["store", "item"]
