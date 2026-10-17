import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class TechnicalAnalyzer:
    """
    Computes key technical indicators for Indian stocks from historical OHLCV data.
    Includes RSI, EMA (20/50/200), MACD, Bollinger Bands, and overall technical bias.
    """

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the OHLCV dataframe and returns latest values, signals,
        and overall market trend recommendation.
        """
        if df is None or df.empty or len(df) < 15:
            return {
                "error": "Insufficient historical candle data to calculate technical indicators."
            }

        df = df.copy()

        # Ensure numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["Close"]

        # EMAs
        df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA_200"] = close.ewm(span=200, adjust=False).mean() if len(df) >= 50 else pd.Series(index=df.index, dtype=float)

        # RSI (14)
        df["RSI_14"] = TechnicalAnalyzer.calculate_rsi(close, 14)

        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Bollinger Bands (20, 2)
        df["SMA_20"] = close.rolling(window=20).mean()
        df["BB_STD"] = close.rolling(window=20).std()
        df["BB_Upper"] = df["SMA_20"] + (df["BB_STD"] * 2)
        df["BB_Lower"] = df["SMA_20"] - (df["BB_STD"] * 2)

        # Latest values
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row

        current_price = float(last_row["Close"])
        rsi_val = round(float(last_row["RSI_14"]), 2)
        ema_20 = round(float(last_row["EMA_20"]), 2) if not pd.isna(last_row["EMA_20"]) else None
        ema_50 = round(float(last_row["EMA_50"]), 2) if not pd.isna(last_row["EMA_50"]) else None
        ema_200 = round(float(last_row["EMA_200"]), 2) if not pd.isna(last_row["EMA_200"]) else None

        macd_val = round(float(last_row["MACD"]), 2)
        macd_signal = round(float(last_row["MACD_Signal"]), 2)
        macd_hist = round(float(last_row["MACD_Hist"]), 2)

        bb_upper = round(float(last_row["BB_Upper"]), 2) if not pd.isna(last_row["BB_Upper"]) else None
        bb_lower = round(float(last_row["BB_Lower"]), 2) if not pd.isna(last_row["BB_Lower"]) else None

        # Key Support and Resistance from 20-period High/Low
        recent_window = df.tail(min(30, len(df)))
        resistance = round(float(recent_window["High"].max()), 2)
        support = round(float(recent_window["Low"].min()), 2)

        # Interpretations
        rsi_condition = "Neutral"
        if rsi_val >= 70:
            rsi_condition = "Overbought (Potential pullback risk)"
        elif rsi_val <= 30:
            rsi_condition = "Oversold (Potential rebound opportunity)"
        elif rsi_val > 55:
            rsi_condition = "Bullish Momentum"
        elif rsi_val < 45:
            rsi_condition = "Bearish Momentum"

        # Trend bias calculation
        bullish_points = 0
        bearish_points = 0

        # Price vs EMAs
        if ema_20:
            if current_price > ema_20:
                bullish_points += 1
            else:
                bearish_points += 1

        if ema_50:
            if current_price > ema_50:
                bullish_points += 1
            else:
                bearish_points += 1

        if ema_200:
            if current_price > ema_200:
                bullish_points += 2  # Strong secular bull sign
            else:
                bearish_points += 2

        # MACD
        if macd_val > macd_signal:
            bullish_points += 1
        else:
            bearish_points += 1

        # Overall stance
        total_pts = bullish_points + bearish_points
        if total_pts > 0:
            bull_pct = bullish_points / total_pts
            if bull_pct >= 0.75:
                overall_bias = "Strongly Bullish"
            elif bull_pct >= 0.55:
                overall_bias = "Bullish"
            elif bull_pct <= 0.25:
                overall_bias = "Strongly Bearish"
            elif bull_pct <= 0.45:
                overall_bias = "Bearish"
            else:
                overall_bias = "Consolidation / Neutral"
        else:
            overall_bias = "Neutral"

        return {
            "current_price": round(current_price, 2),
            "rsi_14": rsi_val,
            "rsi_condition": rsi_condition,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "macd": macd_val,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "macd_crossover": "Bullish Crossover" if macd_val > macd_signal else "Bearish Crossover",
            "bollinger_upper": bb_upper,
            "bollinger_lower": bb_lower,
            "support_level": support,
            "resistance_level": resistance,
            "overall_bias": overall_bias,
            "price_above_ema200": (current_price > ema_200) if ema_200 else None,
            "price_above_ema50": (current_price > ema_50) if ema_50 else None
        }
