import logging
import os
import sys
import click

from sklearn.model_selection import train_test_split

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_manager import DataManager
from src.evaluation import Accuracy, F1Score, FullReport, Precision, Recall
from src.models.model_factory import ModelFactory

evaluation_metrics = {"accuracy": Accuracy,
                      "precision": Precision,
                      "recall": Recall, 
                      "f1": F1Score,
                      "all": FullReport}

WELCOME_MESSAGE = """
****************************Welcome to the Titanic MLOps CLI!***************************

You can train the titanic model and evaluate it using the following command:
    poetry run python src/cli.py FILE_PATH [OPTIONS] 

Options:
  --model    Choose a model: gender_baseline, random_forest, gradient_boosting, logistic_regression
  --metric   Choose an evaluation metric: accuracy, precision, recall, f1, or all to display a full report

Example:
    poetry run python src/cli.py ./data/train.csv --model logistic_regression --metric all
  
"""

@click.command()
@click.argument("file_path", required=False)
@click.option(
    "--model",
    default="logistic_regression",
    help="Choose a model: gender_baseline, random_forest, gradient_boosting, logistic_regression",
)
@click.option(
    "--metric",
    default="accuracy",
    type=click.Choice(["accuracy", "precision", "recall", "f1", "all"], case_sensitive=False),
    help="Evaluation metric to use",
)
def main(file_path: str, model: str, metric: str) -> None:
    """Command-line interface for training and evaluating machine learning models using the titanic dataset.

    This script trains a specified machine learning model on a dataset and evaluates it using a chosen metric.

    Args:
        file_path (str): The path to the dataset file.
        model (str): The name of the model to train.
        metric (str): The metric to use for evaluation.
    """
    # Display the welcome message
    if not file_path:
        click.echo(WELCOME_MESSAGE)
        click.echo("Use **poetry run python src/cli.py --help** to see all options.")
        return

    logging.basicConfig(level=logging.INFO)

    try:
        data_manager = DataManager(file_path)
        click.echo("Loading data...")
        data_manager.load_data()
        click.echo("Preprocessing data...")
        data_manager.preprocess()
        processed_data = data_manager.get_processed_data()
        target = data_manager.get_target()

        click.echo("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(processed_data, target, test_size=0.2, random_state=42)

        click.echo(f"Training the {model} model...")
        model_instance = ModelFactory.create_model(model)
        model_instance.train(X_train, y_train)

        click.echo("Evaluating model accuracy...")
        y_pred = model_instance.predict(X_test)
        # accuracy_evaluator = Accuracy()
        if metric == "all":
            full_report_evaluator = FullReport()
            accuracy, precision, recall, f1_score = full_report_evaluator.evaluate(y_test, y_pred)
            click.echo(
                f"\nFull Report: \n\tAccuracy: {accuracy:.2f}, \n\tPrecision: {precision:.2f}, \n\tRecall: {recall:.2f}, \n\tF1 Score: {f1_score:.2f}"
            )

        else:
            # Select the evaluation metric
            evaluator_class = evaluation_metrics[metric]
            evaluator = evaluator_class()
            score = evaluator.evaluate(y_test, y_pred)
            click.echo(f"Model trained with {metric}: {score:.2f}")

    except Exception as e:
        click.echo(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
