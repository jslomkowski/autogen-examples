from sklearn.ensemble import RandomForestRegressor


def train_model(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2):
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    return model
