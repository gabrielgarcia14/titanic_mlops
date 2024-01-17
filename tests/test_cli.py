import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.cli import main

# Mocks for the dependencies
mock_data_manager = MagicMock()
mock_model_factory = MagicMock()
mock_model_instance = MagicMock()
mock_evaluation_metric = MagicMock()

@pytest.fixture
def runner():
    return CliRunner()

@patch('src.cli.DataManager', return_value=mock_data_manager)
@patch('src.cli.ModelFactory.create_model', return_value=mock_model_instance)
@patch('src.cli.evaluation_metrics', {'accuracy': MagicMock(return_value=mock_evaluation_metric)})
def test_cli_valid_input(mock_data_manager, mock_model_factory, mock_evaluation_metric, runner):
    mock_data_manager.get_processed_data.return_value = 'mock_data'
    mock_data_manager.get_target.return_value = 'mock_target'
    mock_model_instance.predict.return_value = 'mock_predictions'
    mock_evaluation_metric.evaluate.return_value = 0.9

    result = runner.invoke(main, ['data/train.csv', '--model', 'logistic_regression', '--metric', 'accuracy'])
    assert result.exit_code == 0
    assert "Model trained with accuracy: 0.90" in result.output

def test_cli_invalid_model(runner):
    result = runner.invoke(main, ['data/train.csv', '--model', 'invalid_model', '--metric', 'accuracy'])
    assert result.exit_code != 0
    assert "Unknown model type: invalid_model" in result.output

def test_cli_invalid_metric(runner):
    result = runner.invoke(main, ['data/train.csv', '--model', 'logistic_regression', '--metric', 'invalid_metric'])
    assert result.exit_code != 0
    assert "Invalid value for '--metric'" in result.output

def test_cli_missing_data_file(runner):
    result = runner.invoke(main, ['invalid_path/train.csv', '--model', 'logistic_regression', '--metric', 'accuracy'])
    assert result.exit_code != 0
    assert "Error loading data" in result.output