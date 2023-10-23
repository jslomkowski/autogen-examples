from sklearn.preprocessing import StandardScaler


def transform_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    return {'scaler': scaler, 'X_train_transformed': X_train_transformed, 'X_test_transformed': X_test_transformed}
