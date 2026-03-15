"""
XGBoost-based classifier for BTC signal generation.

One BTCModel instance per timeframe (1h, 4h, 1d).
Handles training, prediction, persistence, and incremental retraining.
"""

import logging
import os
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

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the model on a feature DataFrame.

        Labels are generated internally from the 'close' column using
        the timeframe's configured lookahead window.

        Args:
            df: DataFrame with FEATURE_COLUMNS + 'close' column

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

    def retrain_incremental(self, df: pd.DataFrame) -> dict:
        """
        Retrain the model from scratch on an augmented dataset.

        XGBoost does not support true online learning, but full retraining
        on 2+ years of data still runs in ~10-30 seconds — acceptable for
        a GitHub Actions workflow.
        """
        logger.info("Retraining model [%s] with %d rows", self.timeframe, len(df))
        return self.train(df)

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
