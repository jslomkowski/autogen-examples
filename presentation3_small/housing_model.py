# filename: housing_model.py

try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import pickle
except ImportError as e:
    print(f"An error occurred during import: {e}")
    raise

try:
    # Load California housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model to disk
    with open('housing_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
