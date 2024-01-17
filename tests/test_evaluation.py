import numpy as np
from src.evaluation import Accuracy

def test_accuracy_evaluation():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    accuracy_evaluator = Accuracy()
    accuracy = accuracy_evaluator.evaluate(y_true, y_pred)
    expected_accuracy = 0.8  # 4 out of 5 predictions are correct
    assert accuracy == expected_accuracy, f"Expected accuracy is {expected_accuracy}, but got {accuracy}"