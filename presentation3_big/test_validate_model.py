import numpy as np
import model
import validate_model


def test_validate_model():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    model = model.train_model(X_train, y_train)
    validation_report = validate_model.validate_model(model, X_train, y_train)
    assert isinstance(validation_report, dict)
    assert 'mse' in validation_report
    assert 'r2' in validation_report
    assert 'mae' in validation_report
    # R-squared should be between 0 and 1
    assert 0 <= validation_report['r2'] <= 1
