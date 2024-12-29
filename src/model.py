import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_logistic_model(features, target):
    """Train a Logistic Regression model and save it."""
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

    # Create and train the LogisticRegression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(x_train, y_train)

    # Evaluate model performance
    score = model.score(x_test, y_test)
    print(f"Test Accuracy Score: {round(score * 100, 2)}%")

    # Save the trained model
    joblib.dump(model, '../src/trained_model.pkl')
    return model

def save_feature_importance(model, feature_names):
    """Save feature importance to a file and plot it."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0])  # Taking the absolute value of coefficients
    }).sort_values('importance', ascending=False)

    # Save feature importance
    feature_importance.to_csv('../src/feature_importance.csv', index=False)

    # Generate colors using the Viridis colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    explode = [0.1 if i == 0 else 0 for i in range(len(feature_importance))]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.pie(
        feature_importance['importance'],
        labels=feature_importance['feature'],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode
    )
    plt.title('Top Features Influencing the Logistic Regression Model')
    plt.savefig('../src/feature_importance.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Load preprocessed data
    processed_data = pd.read_csv('../src/preprocessed_data.csv')

    # Extract features and target
    feature_names = [
        'Temperature', 'Humidity', 'Fine particulate matter',
        'Coarse particulate matter', 'NO2', 'SO2', 'CO',
        'Nearest Industrial Areas', 'Population_Density'
    ]
    features = processed_data[feature_names]
    target = processed_data['Air Quality']

    # Train model and save results
    model = train_logistic_model(features, target)
    save_feature_importance(model, feature_names)
