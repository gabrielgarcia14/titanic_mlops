from src.models.model_factory import ModelFactory
from src.models.random_forest_model import RandomForestModel

def test_create_random_forest_model():
    model = ModelFactory.create_model("random_forest")
    assert isinstance(model, RandomForestModel), "The created model should be an instance of RandomForestModel"