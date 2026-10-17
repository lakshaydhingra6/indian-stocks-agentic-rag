import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class TechnicalAnalyzer:
    """
    Computes advanced technical indicators for Indian stocks from historical OHLCV data.
    Includes RSI, EMA (20/50/200), Golden/Death Cross detection, MACD, Bollinger Bands,
    Pivot Points (S1/S2/R1/R2), ATR volatility, and actionable trade setups.
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
        Analyzes the OHLCV dataframe and returns comprehensive technicals,
        moving average crossovers, pivot points, and trend bias.
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
        high = df["High"]
        low = df["Low"]

        # EMAs
        df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA_200"] = close.ewm(span=200, adjust=False).mean() if len(df) >= 40 else pd.Series(index=df.index, dtype=float)

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

        # ATR (Average True Range, 14) for volatility & stop loss
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = round(float(true_range.rolling(14).mean().iloc[-1] or (close.iloc[-1] * 0.02)), 2)

        # Latest values
        last_row = df.iloc[-1]
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

        # Golden Cross / Death Cross status
        ma_cross = "Neutral"
        if ema_50 and ema_200:
            if ema_50 > ema_200:
                ma_cross = "Golden Cross (Bullish Long-Term Trend)"
            else:
                ma_cross = "Death Cross (Bearish Long-Term Trend)"

        # Classic Pivot Points from latest daily high, low, close
        h = float(last_row["High"])
        l = float(last_row["Low"])
        c = float(last_row["Close"])
        pivot = round((h + l + c) / 3, 2)
        r1 = round((2 * pivot) - l, 2)
        s1 = round((2 * pivot) - h, 2)
        r2 = round(pivot + (h - l), 2)
        s2 = round(pivot - (h - l), 2)

        # Suggested Stop Loss and Target using ATR
        suggested_stop_loss = round(current_price - (1.5 * atr_14), 2)
        suggested_target = round(current_price + (2.5 * atr_14), 2)

        # Interpretations
        rsi_condition = "Neutral"
        if rsi_val >= 70:
            rsi_condition = "Overbought (Potential pullback risk)"
        elif rsi_val <= 30:
            rsi_condition = "Oversold (Rebound / Bargain hunting opportunity)"
        elif rsi_val > 55:
            rsi_condition = "Bullish Momentum"
        elif rsi_val < 45:
            rsi_condition = "Bearish Momentum"

        # Overall stance
        bullish_points = 0
        bearish_points = 0
        if ema_20:
            bullish_points += 1 if current_price > ema_20 else 0
            bearish_points += 1 if current_price <= ema_20 else 0
        if ema_50:
            bullish_points += 1 if current_price > ema_50 else 0
            bearish_points += 1 if current_price <= ema_50 else 0
        if ema_200:
            bullish_points += 2 if current_price > ema_200 else 0
            bearish_points += 2 if current_price <= ema_200 else 0
        if macd_val > macd_signal:
            bullish_points += 1
        else:
            bearish_points += 1

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
            "ma_crossover_status": ma_cross,
            "macd": macd_val,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "macd_crossover": "Bullish Crossover" if macd_val > macd_signal else "Bearish Crossover",
            "bollinger_upper": bb_upper,
            "bollinger_lower": bb_lower,
            "pivot_point": pivot,
            "resistance_1": r1,
            "resistance_2": r2,
            "support_1": s1,
            "support_2": s2,
            "atr_14": atr_14,
            "suggested_stop_loss": suggested_stop_loss,
            "suggested_target": suggested_target,
            "overall_bias": overall_bias,
            "price_above_ema200": (current_price > ema_200) if ema_200 else None,
            "price_above_ema50": (current_price > ema_50) if ema_50 else None
        }
