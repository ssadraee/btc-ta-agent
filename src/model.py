"""
XGBoost-based classifier for BTC signal generation.

One BTCModel instance per timeframe (1h, 4h, 1d).
Handles training, prediction, persistence, and incremental retraining.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from indicators import FEATURE_COLUMNS, generate_labels

logger = logging.getLogger(__name__)

# Map raw labels (-1, 0, 1) to XGBoost class indices (0, 1, 2)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_REVERSE_MAP = {0: -1, 1: 0, 2: 1}


class BTCModel:
    """XGBoost classifier for one timeframe."""

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.model: XGBClassifier | None = None
        self._feature_columns = FEATURE_COLUMNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self.model is not None

    def train(
        self,
        df: pd.DataFrame,
        outcome_weights: dict[str, float] | None = None,
    ) -> dict:
        """
        Train the model on a feature DataFrame.

        Labels are generated internally from the 'close' column using
        the timeframe's configured lookahead window.

        Args:
            df: DataFrame with FEATURE_COLUMNS + 'close' column
            outcome_weights: optional {iso_timestamp: weight} from evaluated
                signals. Rows matching error-period timestamps get boosted
                sample weights so the model learns harder from its mistakes.

        Returns:
            dict with accuracy, classification_report string
        """
        labels = generate_labels(df, self.timeframe)
        mask = labels.notna()
        X = df.loc[mask, self._feature_columns].astype(float)
        y_raw = labels[mask].astype(int)
        y = y_raw.map(LABEL_MAP)

        if len(X) < 100:
            raise ValueError(f"Not enough training data: {len(X)} rows")

        # Time-ordered train/test split (no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Handle class imbalance by computing weights
        class_counts = y_train.value_counts()
        max_count = class_counts.max()
        sample_weights = y_train.map(lambda c: max_count / class_counts.get(c, 1)).values

        # Apply outcome-based weight multipliers from evaluated signals
        if outcome_weights:
            sample_weights = self._apply_outcome_weights(
                df.loc[mask].iloc[:split_idx], sample_weights, outcome_weights
            )

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        accuracy = (y_pred == y_test.values).mean()

        logger.info(
            "Model [%s] trained on %d samples. Test accuracy: %.3f",
            self.timeframe, len(X_train), accuracy
        )
        logger.debug("Classification report:\n%s", report)

        return {"accuracy": accuracy, "report": report, "n_train": len(X_train)}

    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Predict signal for the latest candle in df.

        Returns:
            (signal, confidence)
            signal: 1=BUY, 0=HOLD, -1=SELL
            confidence: probability of the predicted class [0, 1]
        """
        if self.model is None:
            raise RuntimeError(f"Model [{self.timeframe}] not trained yet")

        X = df[self._feature_columns].astype(float).iloc[[-1]]
        proba = self.model.predict_proba(X)[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        signal = LABEL_REVERSE_MAP[class_idx]

        return signal, confidence

    def retrain_incremental(
        self,
        df: pd.DataFrame,
        outcome_weights: dict[str, float] | None = None,
    ) -> dict:
        """
        Retrain the model from scratch on an augmented dataset.

        XGBoost does not support true online learning, but full retraining
        on 2+ years of data still runs in ~10-30 seconds — acceptable for
        a GitHub Actions workflow.

        Args:
            df: full training DataFrame
            outcome_weights: optional {iso_timestamp: weight} to boost
                sample weights for time periods where the model was wrong
        """
        logger.info("Retraining model [%s] with %d rows", self.timeframe, len(df))
        return self.train(df, outcome_weights=outcome_weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_outcome_weights(
        self,
        df_train: pd.DataFrame,
        sample_weights: np.ndarray,
        outcome_weights: dict[str, float],
    ) -> np.ndarray:
        """
        Multiply sample_weights by outcome-based multipliers for matching rows.

        Matches signal timestamps to training data rows within one candle
        duration window (1h for 1h timeframe, 4h for 4h, etc.).
        """
        candle_hours = {"1h": 1, "4h": 4, "1d": 24}
        window = timedelta(hours=candle_hours.get(self.timeframe, 1))

        # Parse outcome timestamps once
        parsed_outcomes = []
        for ts_str, weight in outcome_weights.items():
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                parsed_outcomes.append((ts, weight))
            except (ValueError, TypeError):
                continue

        if not parsed_outcomes:
            return sample_weights

        sample_weights = sample_weights.copy()
        boosted = 0

        if "timestamp" in df_train.columns:
            for i, row_ts in enumerate(df_train["timestamp"]):
                if pd.isna(row_ts):
                    continue
                if isinstance(row_ts, str):
                    try:
                        row_ts = datetime.fromisoformat(row_ts)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(row_ts, pd.Timestamp):
                    row_ts = row_ts.to_pydatetime()
                if row_ts.tzinfo is None:
                    row_ts = row_ts.replace(tzinfo=timezone.utc)

                for outcome_ts, weight in parsed_outcomes:
                    if abs((row_ts - outcome_ts).total_seconds()) <= window.total_seconds():
                        sample_weights[i] *= weight
                        if weight > 1.0:
                            boosted += 1
                        break

        if boosted:
            logger.info(
                "Outcome weights applied: %d training samples boosted for [%s]",
                boosted, self.timeframe,
            )
        return sample_weights

    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model [%s] saved to %s", self.timeframe, path)

    def load(self, path: str) -> bool:
        """
        Load a previously saved model from disk.

        Returns True if loaded successfully, False if file not found.
        """
        if not os.path.exists(path):
            logger.info("No saved model found at %s", path)
            return False
        self.model = joblib.load(path)
        logger.info("Model [%s] loaded from %s", self.timeframe, path)
        return True

    def get_feature_importance(self) -> dict:
        """Return top feature importances for explanation generation."""
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(sorted(
            zip(self._feature_columns, importance),
            key=lambda x: x[1],
            reverse=True
        ))
