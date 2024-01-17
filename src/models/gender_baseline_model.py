import numpy as np

from .base_model import BaseModel


class GenderBaselineModel(BaseModel):
    """Gender baseline model for predicting survival.

    This model predicts survival based solely on gender Female. It's a simple baseline
    model without any actual training process.
    """

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Performs training for the GenderBaselineModel.

        Since this is a baseline model based on gender, no actual training is performed.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.

        Note:
            This method does not perform any operation as the model is a simple baseline.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts survival based on gender.

        The prediction is made based on the gender of the passenger: predicts `1` (survived) if female,
        and `0` (not survived) if male. Assumes that gender is represented in the 9th column (index 8) of the input array.

        Args:
            X (np.ndarray): Feature data for prediction.

        Returns:
            np.ndarray: Predicted survival, with `1` for female and `0` for male.
        """
        return (X[:, 8].astype("int")).astype(int)
