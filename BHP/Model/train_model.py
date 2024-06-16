import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib
import json

# Load your data
csv_file_path = 'C:/Users/Meghana setty/Downloads/Bengaluru_House_Data.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Preprocess 'total_sqft' column to handle ranges and convert to numeric
def preprocess_total_sqft(x):
    if '-' in x:
        try:
            return (float(x.split('-')[0]) + float(x.split('-')[1])) / 2
        except ValueError:
            return None
    else:
        try:
            return float(x)
        except ValueError:
            return None

data['total_sqft'] = data['total_sqft'].apply(preprocess_total_sqft)

# Drop rows with NaN values in 'total_sqft' column
data.dropna(subset=['total_sqft'], inplace=True)

# Define features and target variable
X = data[['total_sqft', 'bath']]  # Replace with your actual feature columns
y = data['price']  # Replace with your actual target column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in the training data
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Save the model
joblib_file = 'bengaluru_house_data_model.pickle'
joblib.dump(model, joblib_file)

print("Model trained and saved successfully!")

# Save data columns to a JSON file
columns_info = {
    "data_columns": X.columns.tolist()
}
with open('columns.json', 'w') as f:
    json.dump(columns_info, f)
