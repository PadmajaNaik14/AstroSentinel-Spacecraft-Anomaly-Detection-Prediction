"""SHAP explainability: TreeExplainer on the Isolation Forest's engineered
rolling-error features. This is the only thing labeled "SHAP" anywhere in
AstroSentinel -- SHAP on the raw LSTM sequence model is impractical/unreliable
at this scope, so LSTM timestep attribution is a separate, explicitly-labeled
reconstruction-error heatmap instead (see ml.evaluation / backend inference).
"""

import numpy as np
import shap
from sklearn.ensemble import IsolationForest

from ml.models.isolation_forest import ENGINEERED_FEATURE_NAMES


class ShapExplainer:
    def __init__(self, isolation_forest_model: IsolationForest):
        self._explainer = shap.TreeExplainer(isolation_forest_model)

    def explain(self, feature_vector: np.ndarray) -> dict:
        """feature_vector: shape (6,) engineered features for one timestep."""
        shap_values = self._explainer.shap_values(feature_vector.reshape(1, -1))
        values = np.asarray(shap_values).reshape(-1)
        return {
            "feature_names": ENGINEERED_FEATURE_NAMES,
            "shap_values": [float(v) for v in values],
            "feature_values": [float(v) for v in feature_vector],
        }
