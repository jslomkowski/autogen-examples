from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def validate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(
        f'Mean Squared Error: {mse}, R-squared: {r2}, Mean Absolute Error: {mae}')
    return {'mse': mse, 'r2': r2, 'mae': mae}
