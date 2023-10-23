import numpy as np
import transform_data


def test_transform_data():
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[5, 6], [7, 8]])
    transformed_data = transform_data.transform_data(X_train, X_test)
    assert isinstance(transformed_data, dict)
    assert 'scaler' in transformed_data
    assert 'X_train_transformed' in transformed_data
    assert 'X_test_transformed' in transformed_data
    assert transformed_data['X_train_transformed'].shape == X_train.shape
    assert np.isclose(transformed_data['X_train_transformed'].mean(), 0)
