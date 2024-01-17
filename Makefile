# Define variables for directories
SRC_DIR := src

.PHONY: setup run tests coverage_report lint format type_check

# Install project dependencies using Poetry
setup:
	@echo "Installing dependencies..."
	poetry install

# Run the project 
run:
	@echo "Running the application..."
	poetry run python src/cli.py

# Test the project 
tests:
	@echo "Running the test..."
	poetry run coverage run --source src -m pytest tests

# Coverage test report
coverage_report:
	@echo "Running the test..."
	poetry run coverage report

# Example target to run Jupyter Lab
jupyter:
	@echo "Starting Jupyter Lab..."
	poetry run jupyter lab

# Run linters and code checkers
lint:
	@echo "Running Flake8..."
	@poetry run flake8 $(SRC_DIR)

format:
	@poetry run black $(SRC_DIR)
	@echo "Running isort..."
	@poetry run isort $(SRC_DIR)

type_check:
	@echo "Running mypy..."
	@poetry run mypy $(SRC_DIR)

quality_checks: lint format type_check

help:
	@echo "Available commands:"
	@echo " make setup - Install all the project dependencies"
	@echo " make run - Run CLI"
	@echo " make tests - Run tests"
	@echo " make coverage_report - Generate Coverage Report"
	@echo " make jupyter - Run jupyter lab"
	@echo " make lint - Run Flake8 to check code style"
	@echo " make format - Format code with Black and isort"
	@echo " make type_check - Run mypy for type checking"
	@echo " make quality_checks - Run all the above quality checks"