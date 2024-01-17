# Titanic MLOps

## Overview
Titanic MLOps is a machine learning project aimed at providing solutions for the Titanic survival prediction problem. The project leverages modern software engineering practices and includes a command-line interface (CLI) for easy interaction, alongside a robust development environment managed with Poetry.

## Getting Started

### Prerequisites

* Python 3.11 or higher
* Poetry for dependency management (You have to [install it](https://python-poetry.org/docs/#installing-with-the-official-installer) depending on your OS)

### Installation

1. Clone the Repository:

git clone https://github.com/gabrielgarcia14/titanic_mlops.git
cd titanic_mlops

2. Install dependencies

make setup

### Using the Makefile

The project includes a Makefile for simplifying common tasks:

* Install dependencies:
make setup
* Run the Application:
make run
* Run the tests:
make tests
* Run the coverage report:
make coverage_report 
* Run Linters and Formatters:
make lint     # Run flake8
make format   # Format code with black
make sort     # Sort imports with isort
make type-check # Run type checking with mypy
quality_checks # Run all of the above

### Using the CLI

The CLI provides a user-friendly way to interact with the model. You can train the model and evaluate its performance using various metrics.

* Basic Command Structure:

poetry run python src/cli.py FILE_PATH [OPTIONS] 

* Options 

--model MODEL_NAME: Choose a model to train. Options are gender_baseline, random_forest, gradient_boosting, logistic_regression.

--metric METRIC: Choose an evaluation metric. Options are accuracy, precision, recall, f1, and all for a full report.

### Example Usage:

poetry run python src/cli.py ./data/train.csv --model logistic_regression --metric all 

This command will run the logistic regression model on the data provided in ./data/train.csv and evaluate it using all available metrics.

## Development

* Jupyter Notebooks: 
If you're using Jupyter notebooks for development, you can
start Jupyter Lab using the Makefile command make jupyter. This is useful for exploratory data analysis and trying out code snippets.

* Code Formatting: 
To maintain code quality and consistency, the project uses flake8 for linting, black for code formatting, isort for import sorting, and mypy for type checking. You can run these tools via the Makefile.

## Testing

Tests are an integral part of the project. To run all tests, use the Makefile command make test. Make sure to write tests when adding new features or fixing bugs.

## Contributing
Contributions to the project are welcome. Please adhere to the project's code style guidelines and write tests for new features or bug fixes. For major changes, please open an issue first to discuss what you would like to change.
