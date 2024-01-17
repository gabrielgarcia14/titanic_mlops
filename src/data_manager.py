import logging
import re

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataManager:
    """Manages data loading and preprocessing for Titanic dataset.

    This class handles operations like data loading from a file, preprocessing,
    feature transformation, and keeping track of the target variable.
    """

    def __init__(self, filepath: str) -> None:
        """Initializes the DataManager with the specified file path.

        Args:
            filepath (str): Path to the data file.
        """
        self.filepath = filepath
        self._data = None
        self._processed_data = None
        self._target = None

    @property
    def data(self) -> pd.DataFrame:
        """Returns the loaded data.

        Raises:
            ValueError: If data is not yet loaded.
        """
        if self._data is not None:
            return self._data
        else:
            raise ValueError("Data not loaded. Please run load_data() method first.")

    @property
    def processed_data(self) -> np.ndarray:
        """Returns the processed data.

        Raises:
            ValueError: If data is not yet processed.
        """
        if self._processed_data is not None:
            return self._processed_data
        else:
            raise ValueError("Data not processed. Please run preprocess() method first.")

    def load_data(self) -> None:
        """Loads data from the specified file path.

        The method tries to load the dataset into a pandas DataFrame. It logs
        and raises an exception if an error occurs during file loading.

        Raises:
            Exception: If an error occurs during data loading.
        """
        try:
            self._data = pd.read_csv(self.filepath)
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess(self) -> None:
        """Performs data preprocessing steps.

        This includes dropping unnecessary columns, extracting titles, creating family
        features, imputing age, and transforming features. The processed data is stored
        internally and can be accessed using the processed_data property.

        Raises:
            ValueError: If data has not been loaded prior to preprocessing.
            Exception: If an error occurs during preprocessing.
        """
        if self._data is None:
            raise ValueError("Data not loaded. Please run load_data() method first.")

        try:
            self._drop_unnecessary_columns()
            self._extract_titles()
            self._create_family_features()
            self._impute_age()
            self._target = self._data["Survived"]
            self._data.drop(columns=["Survived"], inplace=True)
            self._processed_data = self._transform_features(self._data)
            logging.info("Data preprocessing completed successfully.")
        except Exception as e:
            logging.error(f"Error in preprocessing: {e}")
            raise

    def _drop_unnecessary_columns(self) -> None:
        self._data.drop(columns=["Ticket", "Cabin", "PassengerId"], inplace=True)

    def _extract_titles(self) -> None:
        self._data["Title"] = self._data["Name"].apply(
            lambda x: re.findall(r"\b\w+\.", x)[0] if re.findall(r"\b\w+\.", x) else "Unknown"
        )
        title_mappings = {"Mlle.": "Miss.", "Ms.": "Miss.", "Mme.": "Mrs.", "Lady.": "Mrs."}
        self._data["Title"] = self._data["Title"].replace(title_mappings)
        self._data["Title"] = self._data["Title"].apply(
            lambda x: x if x in (["Mr.", "Miss.", "Mrs.", "Master."]) else "Other"
        )
        self._data.drop(columns=["Name"], inplace=True)

    def _impute_age(self) -> None:
        """Impute Age based on mean age by Title and Pclass."""
        mean_ages = self._data.groupby(["Title", "Pclass"])["Age"].transform("mean")
        self._data["Age"].fillna(mean_ages, inplace=True)

    def _create_family_features(self) -> None:
        self._data["FamilySize"] = self._data["SibSp"] + self._data["Parch"] + 1
        self._data["IsAlone"] = 1
        self._data.loc[self._data["FamilySize"] > 1, "IsAlone"] = 0

    def _transform_features(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms the features using a predefined pipeline.

        Args:
            data (pd.DataFrame): The data to be transformed.

        Returns:
            np.ndarray: The transformed feature data.
        """
        numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
        categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return preprocessor.fit_transform(data)

    def get_target(self) -> np.ndarray:
        """Returns the target variable data.

        Raises:
            ValueError: If the target variable is not set.
        """
        if self._target is not None:
            return self._target
        else:
            raise ValueError("Target variable not set. Please run preprocess() method first.")

    def get_processed_data(self) -> np.ndarray:
        """Returns the processed feature data.

        Raises:
            ValueError: If the data has not been processed yet.
        """
        if self._processed_data is not None:
            return self._processed_data
        else:
            raise ValueError("Data has not been processed. Call preprocess() method first.")
