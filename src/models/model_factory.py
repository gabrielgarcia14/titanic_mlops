from .base_model import BaseModel
from .gender_baseline_model import GenderBaselineModel
from .gradient_boosting_model import GradientBoostingModel
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel


class ModelFactory:
    """Factory class for creating instances of different model types.

    This class provides a static method to instantiate model objects based on a given model name.
    """

    @staticmethod
    def create_model(model_name: str) -> BaseModel:
        """Creates and returns an instance of the specified model.

        Based on the provided model name, this method returns an instance of the corresponding model class.
        Supported models include 'gender_baseline', 'random_forest', 'gradient_boosting', and 'logistic_regression'.

        Args:
            model_name (str): The name of the model to create.

        Returns:
            An instance of the specified model class.

        Raises:
            ValueError: If an unknown model type is specified.
        """
        if model_name == "gender_baseline":
            return GenderBaselineModel()
        elif model_name == "random_forest":
            return RandomForestModel()
        elif model_name == "gradient_boosting":
            return GradientBoostingModel()
        elif model_name == "logistic_regression":
            return LogisticRegressionModel()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
