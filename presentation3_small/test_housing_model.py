# filename: test_housing_model.py

import pytest
import pickle
try:
    from sklearn.datasets import fetch_california_housing
    from housing_model import model
except ImportError as e:
    print(f"An error occurred during import: {e}")
    raise


@pytest.fixture
def load_data():
    # Load California housing dataset
    data = fetch_california_housing()
    return data.data, data.target


def test_model_training(load_data):
    X, y = load_data

    # Make predictions using the trained model
    predictions = model.predict(X)

    # Check if the model is making predictions
    assert predictions is not None


def test_model_saving():
    try:
        # Load the saved model
        with open('housing_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Check if the model was saved and loaded correctly
        assert loaded_model is not None
    except FileNotFoundError:
        pytest.fail("Model file not found.")
