from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
