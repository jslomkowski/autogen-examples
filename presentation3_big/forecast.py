import numpy as np


def generate_forecast(model, new_data, scaler):
    if model is None or new_data is None:
        return None
    if not isinstance(new_data, (list, np.ndarray)):
        raise ValueError("new_data must be a list or numpy array")
    new_data_transformed = scaler.transform(new_data)
    forecast = model.predict(new_data_transformed)
    return forecast
