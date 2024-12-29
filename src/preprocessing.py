import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Define preprocessing steps as functions
def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def rename_columns(df):
    """Rename columns for better readability."""
    return df.rename(columns={
        'PM2.5': 'Fine particulate matter',
        'PM10': 'Coarse particulate matter',
        'Proximity_to_Industrial_Areas': 'Nearest Industrial Areas'
    })

def handle_missing_values(df):
    """Handle missing values by imputing with the median."""
    imputer = SimpleImputer(strategy='median')
    df[df.columns] = imputer.fit_transform(df)
    return df

def scale_numeric_data(df, numeric_cols):
    """Scale numerical data using RobustScaler."""
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, '../src/scaler.pkl')  # Save the scaler for later use
    return df

def encode_categorical_data(df, categorical_col):
    """Encode categorical data using LabelEncoder."""
    encoder = LabelEncoder()
    df[categorical_col] = encoder.fit_transform(df[categorical_col])
    joblib.dump(encoder, '../src/encoder.pkl')  # Save the encoder for later use
    return df

def remove_outliers(df, numeric_cols):
    """Remove outliers from numerical data based on the IQR method."""
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[numeric_cols] >= lower_bound) & (df[numeric_cols] <= upper_bound)]

def save_feature_names(numeric_cols):
    """Save feature names to a text file."""
    with open('../src/feature_names.txt', 'w') as f:
        f.write('\n'.join(numeric_cols))

# Define the preprocessing pipeline
def preprocess_data(file_path):
    """Complete preprocessing pipeline."""
    df = load_data(file_path)
    df = rename_columns(df)

    numeric_cols = [
        'Temperature', 'Humidity', 'Fine particulate matter',
        'Coarse particulate matter', 'NO2', 'SO2', 'CO',
        'Nearest Industrial Areas', 'Population_Density'
    ]

    df = handle_missing_values(df)
    df = scale_numeric_data(df, numeric_cols)
    df = remove_outliers(df, numeric_cols)
    df['Air Quality'] = encode_categorical_data(df, 'Air Quality')['Air Quality']

    save_feature_names(numeric_cols)
    return df

# Example usage
if __name__ == "__main__":
    file_path = r"D:\\Data Analysis\\GDG-CoreTeam\\Air pollution\\Dataset\\updated_pollution_dataset.csv"
    processed_data = preprocess_data(file_path)
    print("Preprocessing completed.")
