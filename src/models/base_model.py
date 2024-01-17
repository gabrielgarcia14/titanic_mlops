from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract base class for machine learning models.

    This class defines the basic structure and interface for machine learning models,
    with abstract methods for training and prediction.
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the model on the provided dataset.

        Args:
            X_train (np.ndarray): Training feature data, where rows are samples and columns are features.
            y_train (np.ndarray): Training target data, which is an array of target variable values.

        Note:This method needs to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts outputs using the trained model on the given input data.

        Args:
            X (np.ndarray): Feature data for prediction, where rows are samples and columns are features.

        Returns:
            np.ndarray: The predicted values, typically as a 1D array of the same length as the input samples.

        Note:
            This method needs to be implemented by subclasses.
        """
        pass
