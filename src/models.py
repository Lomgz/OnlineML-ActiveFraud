from __future__ import annotations

import inspect
from typing import Dict, List, Optional

import numpy as np
from river.anomaly import HalfSpaceTrees
from river.forest import ARFClassifier
from river.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


class OnlineFraudModel:
    """Wrapper around river ARFClassifier with safe signature handling."""

    def __init__(self, n_models: int = 10, seed: int = 42) -> None:
        self.model = ARFClassifier(n_models=n_models, seed=seed)
        sig = inspect.signature(self.model.learn_one)
        self.learn_one_params = tuple(sig.parameters.keys())
        self.supports_sample_weight = "w" in self.learn_one_params or "sample_weight" in self.learn_one_params
        self.n_updates = 0

    def predict_proba_one(self, x: Dict[str, float]) -> float:
        proba = self.model.predict_proba_one(x)
        if not proba:
            return 0.0
        return float(proba.get(1, proba.get(True, 0.0)))

    def learn_one(self, x: Dict[str, float], y: int) -> None:
        self.model.learn_one(x, int(y))
        self.n_updates += 1


class SemiSupervisedHST:
    """
    MinMaxScaler + HalfSpaceTrees updated ONLY on confirmed normal (y == 0) labels.
    Read-only transform during scoring so anomalies never pollute scaler or HST mass.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 250,
        min_warmup_normals: int = 50,
        seed: int = 42,
    ) -> None:
        self.scaler = MinMaxScaler()
        self.hst = HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=seed,
        )
        self.min_warmup_normals = int(min_warmup_normals)
        self.n_normal_seen = 0

    def _transform_clipped(self, x: Dict[str, float]) -> Dict[str, float]:
        scaled = self.scaler.transform_one(x)
        return {k: min(1.0, max(0.0, float(v))) for k, v in scaled.items()}

    def score_one(self, x: Dict[str, float]) -> float:
        if self.n_normal_seen < self.min_warmup_normals:
            return 0.0
        x_scaled = self._transform_clipped(x)
        raw_score = float(self.hst.score_one(x_scaled))
        return min(1.0, max(0.0, raw_score))

    def learn_one(self, x: Dict[str, float], y: int) -> bool:
        if int(y) != 0:
            return False
        self.scaler.learn_one(x)
        x_scaled = self._transform_clipped(x)
        self.hst.learn_one(x_scaled)
        self.n_normal_seen += 1
        return True


class BatchXGBoostWrapper:
    """
    Unified XGBoost baseline supporting:
      - 'static': fit once on matured warm-up, frozen forever.
      - 'periodic': retrain at chunk boundaries using standard matured labels.
      - 'periodic_al': retrain at chunk boundaries using standard + Fast-Track matured labels.
    """

    VALID_MODES = {"static", "periodic", "periodic_al"}

    def __init__(
        self,
        feature_cols: List[str],
        mode: str = "static",
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from {self.VALID_MODES}.")
        self.feature_cols = list(feature_cols)
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed

        self.model: Optional[XGBClassifier] = None
        self._constant_proba: Optional[float] = None
        self.fit_count = 0

    def _dicts_to_matrix(self, x_list: List[Dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[float(row[c]) for c in self.feature_cols] for row in x_list],
            dtype=np.float32,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        unique_classes = np.unique(y_arr)
        self.fit_count += 1

        if len(unique_classes) < 2:
            self.model = None
            self._constant_proba = float(unique_classes[0]) if len(unique_classes) == 1 else 0.0
            return

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        scale_pos_weight = float(n_neg) / max(1.0, float(n_pos))

        clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=-1,
        )
        clf.fit(X_arr, y_arr)
        self.model = clf
        self._constant_proba = None

    def update_if_allowed(self, X: np.ndarray, y: np.ndarray) -> bool:
        if self.mode == "static" and self.fit_count >= 1:
            return False
        self.fit(X, y)
        return True

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if self.model is None:
            val = 0.0 if self._constant_proba is None else self._constant_proba
            return np.full(len(X_arr), val, dtype=np.float64)
        return self.model.predict_proba(X_arr)[:, 1].astype(np.float64)

    def predict_proba_one(self, x: Dict[str, float]) -> float:
        if self.model is None:
            return 0.0 if self._constant_proba is None else self._constant_proba
        vec = np.asarray([[float(x[c]) for c in self.feature_cols]], dtype=np.float32)
        return float(self.model.predict_proba(vec)[0, 1])