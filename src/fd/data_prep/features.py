import math
import numpy as np
import pandas as pd
from dataclasses import dataclass


SECONDS_PER_DAY = 86400.0
TWO_PI = 2.0 * math.pi


@dataclass
class Scaler:
    mu_time: float
    sigma_time: float
    mu_amount: float  # log
    sigma_amount: float  # log

    def _transform_days(self, t_seconds: np.ndarray) -> np.ndarray:
        days = t_seconds / SECONDS_PER_DAY
        # protect against sigma=0 on tiny subsets
        denom = self.sigma_time if self.sigma_time > 0 else 1.0
        return (days - self.mu_time) / denom
    
    def transform_time_features(self, df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
        """Return a DataFrame with 3 columns: time_days_z, tod_sin, tod_cos."""
        t = df[time_col].to_numpy(dtype=np.float64)

        # 1) z-score of Time/86400 (day scale)
        time_days_z = self._transform_days(t)

        # 2) time-of-day in [0,1) then sine/cosine
        tod = np.mod(t, SECONDS_PER_DAY) / SECONDS_PER_DAY
        tod_sin = np.sin(TWO_PI * tod)
        tod_cos = np.cos(TWO_PI * tod)

        return pd.DataFrame(
            {
                "time_days_z": time_days_z,
                "tod_sin": tod_sin,
                "tod_cos": tod_cos,
            },
            index=df.index,
        )
        
    def transform_amount(self, df: pd.DataFrame, amount_col: str = "Amount") -> np.ndarray:
        a = df[amount_col].to_numpy(dtype=np.float64)
        log_amount = np.log1p(a)
        denom = self.sigma_amount if self.sigma_amount > 0 else 1.0
        return ((log_amount - self.mu_amount) / denom).astype(np.float32)
    
    def get_normalized_features(self, df: pd.DataFrame, amount_col = "Amount", time_col = "Time") -> pd.DataFrame:
        """only those features that needed to be normalized, add the rest manually"""
        out = self.transform_time_features(df, time_col=time_col)
        if amount_col in df.columns:
            out[amount_col] = self.transform_amount(df, amount_col=amount_col)
        return out



def fit_scaler(train_df: pd.DataFrame, time_col: str = "Time", amount_col: str = "Amount") -> Scaler:
    """Normalize for TRAIN ONLY"""
    # time
    days = train_df[time_col].to_numpy(dtype=np.float64) / SECONDS_PER_DAY
    mu_time = float(days.mean())
    sigma_time = float(days.std(ddof=0))
    if sigma_time == 0.0: sigma_time = 1.0
    
    # amount
    amount = train_df[amount_col].to_numpy(dtype=np.float64)
    log_amount = np.log1p(amount)
    mu_amount = float(log_amount.mean())
    sigma_amount = float(log_amount.std(ddof=0))
    if sigma_amount == 0.0:
        sigma_amount = 1.0
    return Scaler(mu_time=mu_time, 
                  sigma_time=sigma_time, 
                  mu_amount=mu_amount, 
                  sigma_amount=sigma_amount)
    
    