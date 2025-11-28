import pandas as pd
import numpy as np


def compute_htf_bias(df: pd.DataFrame) -> str:
    """
    Calcule le biais EMA HTF en 4h : "long", "short" ou "neutral".
    """
    close_4h = df["close"].resample("4h").last().dropna()

    if isinstance(close_4h, pd.DataFrame):
        if close_4h.shape[1] == 0:
            return "neutral"
        close_4h = close_4h.iloc[:, 0].dropna()

    if close_4h.empty:
        return "neutral"

    ema9 = close_4h.ewm(span=9, adjust=False).mean()
    ema21 = close_4h.ewm(span=21, adjust=False).mean()

    if ema9.empty or ema21.empty:
        return "neutral"

    if ema9.iloc[-1] > ema21.iloc[-1]:
        return "long"
    elif ema9.iloc[-1] < ema21.iloc[-1]:
        return "short"
    return "neutral"


def is_in_killzone(ts: pd.Timestamp) -> bool:
    """
    Détermine si l'heure est dans les killzones (NY time).
    - Matin : 9:50–10:10
    - Après-midi : 13:50–14:10
    """
    ny_time = ts.tz_convert("America/New_York")
    t = ny_time.time()
    return (
        (t >= ny_time.replace(hour=9,  minute=50).time() and t <= ny_time.replace(hour=10, minute=10).time()) or
        (t >= ny_time.replace(hour=13, minute=50).time() and t <= ny_time.replace(hour=14, minute=10).time())
    )


def detect_swings(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    """
    Détecte les swing highs/lows et retourne aussi leurs valeurs.
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("Missing required 'high' and 'low' columns in DataFrame.")

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    swing_high_val = np.full(n, np.nan)
    swing_low_val = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        hi_window = highs[i - lookback:i + lookback + 1]
        lo_window = lows[i - lookback:i + lookback + 1]

        if highs[i] == np.max(hi_window):
            swing_high[i] = True
            swing_high_val[i] = highs[i]
        if lows[i] == np.min(lo_window):
            swing_low[i] = True
            swing_low_val[i] = lows[i]

    return pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_high_val": swing_high_val,
        "swing_low_val": swing_low_val,
    }, index=df.index.copy())


def is_structure_clean(swing_df: pd.DataFrame, min_distance: float = 0.002) -> bool:
    """
    Évalue si la structure est "propre" (mouvements clairs entre swings).
    - min_distance: distance minimale moyenne en % entre swings pour valider.
    """
    highs = swing_df[swing_df["swing_high"]]["swing_high_val"].dropna()
    lows = swing_df[swing_df["swing_low"]]["swing_low_val"].dropna()
    if len(highs) < 2 or len(lows) < 2:
        return False

    # Calcul de l'amplitude moyenne entre swings
    swing_points = pd.concat([highs, lows]).sort_index()
    distances = swing_points.diff().dropna().abs()

    if distances.empty:
        return False

    # Moyenne relative en pourcentage
    avg_dist_pct = (distances / swing_points.shift(1)).mean()
    return avg_dist_pct > min_distance
