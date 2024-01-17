import logging

from abc import ABC, abstractmethod

import numpy as np


class Evaluation(ABC):
    """This class represents an evaluation object."""

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """This method evaluates the performance of a model."""
        pass


class Accuracy(Evaluation):
    """This class represents an accuracy object.
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
    """

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """This method evaluates the performance of a model."""

        try:
            correct_predictions = np.sum(y_true == y_pred)
            total_predictions = len(y_true)
            score = correct_predictions / total_predictions
            return score
        except Exception as e:
            logging.error("Error while evaluating accuracy of the model: {}".format(e))
            raise e


class Precision(Evaluation):
    """This class represents a Precision object."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            predicted_positives = np.sum(y_pred == 1)
            score = true_positives / predicted_positives if predicted_positives != 0 else 0
            return score
        except Exception as e:
            logging.error(f"Error while evaluating Precision Score: {e}")
            raise


class Recall(Evaluation):
    """This class represents a Recall object."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            actual_positives = np.sum(y_true == 1)
            score = true_positives / actual_positives if actual_positives != 0 else 0
            return score
        except Exception as e:
            logging.error(f"Error while evaluating Recall Score: {e}")
            raise


class F1Score(Evaluation):
    """This class represents an F1 Score object."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            precision_evaluator = Precision()
            recall_evaluator = Recall()
            precision = precision_evaluator.evaluate(y_true, y_pred)
            recall = recall_evaluator.evaluate(y_true, y_pred)
            score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            return score
        except Exception as e:
            logging.error(f"Error while evaluating F1 Score: {e}")
            raise


class FullReport(Evaluation):
    """This class represents a Full Report object."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        try:
            accuracy_evaluator = Accuracy()
            precision_evaluator = Precision()
            recall_evaluator = Recall()
            f1_evaluator = F1Score()

            accuracy = accuracy_evaluator.evaluate(y_true, y_pred)
            precision = precision_evaluator.evaluate(y_true, y_pred)
            recall = recall_evaluator.evaluate(y_true, y_pred)
            f1_evaluator = f1_evaluator.evaluate(y_true, y_pred)

            return (accuracy, precision, recall, f1_evaluator)
        except Exception as e:
            logging.error(f"Error while evaluating F1 Score: {e}")
            raise
