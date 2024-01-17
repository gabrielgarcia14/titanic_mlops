import numpy as np

from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model wrapper class.

    This class is a wrapper for the RandomForestClassifier from sklearn, providing
    train and predict functionalities.
    """

    def __init__(self) -> None:
        """Initializes the RandomForestModel with a RandomForestClassifier."""
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the RandomForest model on the provided dataset.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained RandomForest model.

        Args:
            X (np.ndarray): Feature data for prediction.

        Returns:
            np.ndarray: The predicted values.
        """
        return self.model.predict(X)
