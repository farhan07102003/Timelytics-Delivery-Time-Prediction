import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
num_samples = 1000

# Features
product_categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books']
customer_locations = ['North America', 'Europe', 'Asia', 'Other']
shipping_methods = ['Standard', 'Express', 'Overnight']

# Generate random data
data = {
    'product_category': np.random.choice(product_categories, num_samples),
    'customer_location': np.random.choice(customer_locations, num_samples),
    'shipping_method': np.random.choice(shipping_methods, num_samples),
    'delivery_time_hours': np.random.randint(12, 120, num_samples)  # Target variable
}

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('delivery_data.csv', index=False)
print("Dataset created!")