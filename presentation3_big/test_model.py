import numpy as np
import model


def test_train_model():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    model = model.train_model(X_train, y_train)
    assert model is not None
    predictions = model.predict(X_train)
    assert predictions.shape == y_train.shape
