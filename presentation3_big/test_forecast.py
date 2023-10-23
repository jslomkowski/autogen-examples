import numpy as np
import model
import forecast
import transform_data


def test_generate_forecast():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    model = model.train_model(X_train, y_train)
    new_data = np.array([[5, 6]])
    transformed_data = transform_data.transform_data(X_train, new_data)
    forecast = forecast.generate_forecast(
        model, transformed_data['X_test_transformed'], transformed_data['scaler'])
    assert forecast is not None
    assert forecast.shape == (1,)  # We are forecasting for 1 sample
    assert isinstance(forecast[0], float)
