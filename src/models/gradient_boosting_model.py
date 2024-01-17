import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from .base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model wrapper class.

    This class is a wrapper for the GradientBoostingClassifier from sklearn, providing
    train and predict functionalities.
    """

    def __init__(self) -> None:
        """Initializes the GradientBoostingModel with a GradientBoostingClassifier."""
        self.model = GradientBoostingClassifier(random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the GradientBoosting model on the provided dataset.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained GradientBoosting model.

        Args:
            X (np.ndarray): Feature data for prediction.

        Returns:
            np.ndarray: The predicted values.
        """
        return self.model.predict(X)
