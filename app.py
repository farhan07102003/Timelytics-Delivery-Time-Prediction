import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_time_model.pkl')

# App title
st.title('Timelytics Delivery Time Prediction 🚚⏱️')

# Input widgets
st.header('Order Details')
product_category = st.selectbox(
    'Product Category',
    ('Electronics', 'Clothing', 'Home & Kitchen', 'Books')
)

customer_location = st.selectbox(
    'Customer Location',
    ('North America', 'Europe', 'Asia', 'Other')
)

shipping_method = st.selectbox(
    'Shipping Method',
    ('Standard', 'Express', 'Overnight')
)

# Create input DataFrame
input_data = pd.DataFrame([[product_category, customer_location, shipping_method]],
                          columns=['product_category', 'customer_location', 'shipping_method'])

# Predict button
if st.button('Predict Delivery Time'):
    prediction = model.predict(input_data)[0]
    st.success(f'Estimated Delivery Time: **{prediction:.1f} hours**')