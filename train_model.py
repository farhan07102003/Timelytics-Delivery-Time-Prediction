import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load synthetic data
df = pd.read_csv('delivery_data.csv')

# Define features (X) and target (y)
X = df[['product_category', 'customer_location', 'shipping_method']]
y = df['delivery_time_hours']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['product_category', 'customer_location', 'shipping_method'])
    ])

# Create a pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'delivery_time_model.pkl')
print("Model trained and saved!")