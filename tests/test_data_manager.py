import pytest
from src.data_manager import DataManager

def test_load_data_success():
    filepath = "./data/train.csv" 
    dm = DataManager(filepath)
    dm.load_data()
    assert dm.data is not None, "Data should be loaded"

def test_load_data_fail_with_invalid_path():
    with pytest.raises(Exception):
        filepath = "no_exist_folder/data.csv"
        dm = DataManager(filepath)
        dm.load_data()