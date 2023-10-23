import load_data


def test_load_data():
    data = load_data.load_data()
    assert isinstance(data, dict)
    assert 'X_train' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert 'y_test' in data
    # California housing data has 8 features
    assert data['X_train'].shape[1] == 8
    assert data['X_train'].dtype == float
