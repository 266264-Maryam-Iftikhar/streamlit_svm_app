import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the pre-trained SVM model and scaler
model_path = r"C:\Users\Maryam Iftikhar\OneDrive\Desktop\streamlit_app\svm_trained_model.pkl"
scaler_path = r"C:\Users\Maryam Iftikhar\OneDrive\Desktop\streamlit_app\scaler.pkl"
svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to make predictions
def predict_purchase(gender, age, estimated_salary):
    gender_encoded = 1 if gender == 'Male' else 0
    # Use only age and estimated salary for scaling
    features = np.array([[age, estimated_salary]])  # Don't scale gender
    scaled_features = scaler.transform(features)  # Apply scaling
    # Add the gender feature to the scaled data
    final_features = np.concatenate((scaled_features, [[gender_encoded]]), axis=1)
    prediction = svm_model.predict(final_features)
    return 'Will Purchase' if prediction[0] == 1 else 'Will Not Purchase'

# Streamlit App Interface
st.set_page_config(page_title="SVM Purchase Prediction App", page_icon=":guardsman:", layout="wide")

# Add custom CSS for styling the app
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Helvetica', sans-serif;
        }
        .title {
            font-size: 80px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
        }
        .description {
            font-size: 50px;
            color: #333;
            text-align: center;
        }
        .prediction-result {
            font-size: 22px;
            font-weight: bold;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .positive {
            background-color: #4CAF50;
        }
        .negative {
            background-color: #f44336;
        }
        .input-box {
            margin: 20px;
            padding: 15px;
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with user inputs
st.sidebar.header("User Inputs")
gender = st.sidebar.selectbox("Select Gender:", ["Male", "Female"])
age = st.sidebar.slider("Enter Age:", 18, 60, 25)
estimated_salary = st.sidebar.number_input("Enter Estimated Salary:", min_value=1000, max_value=150000, value=50000, step=1000)

# Add model information in sidebar
st.sidebar.header("Model Information")
st.sidebar.write("""
    **Model Type:** Support Vector Machine (SVM)
    **Trained on:** User Purchase Data
    **Prediction Objective:** Predict if a user will purchase a product based on gender, age, and salary.
    """)

# Display the title
st.markdown('<p class="title">SVM Purchase Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Predict whether a user will purchase a product based on their details using a trained SVM model.</p>', unsafe_allow_html=True)

# Organize input form into columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    
with col2:
    st.subheader("Financial Details")
    st.write(f"**Estimated Salary:** {estimated_salary}")

# Prediction button with custom styling
if st.button("Predict", key="predict", help="Click to predict whether the user will purchase the product"):
    result = predict_purchase(gender, age, estimated_salary)
    
    # Style the prediction result
    if result == 'Will Purchase':
        st.markdown(f'<p class="prediction-result positive">Prediction: {result}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="prediction-result negative">Prediction: {result}</p>', unsafe_allow_html=True)

# Example of adding a chart (optional)
# You can create some simple insights for users to visualize, like a bar chart of predicted vs actual purchase data.
# For now, adding a dummy chart as an example:
st.subheader("Prediction Accuracy Chart")
accuracy_data = [93, 7]
labels = ['Correct Predictions', 'Incorrect Predictions']
fig, ax = plt.subplots()
ax.bar(labels, accuracy_data, color=['#4CAF50', '#f44336'])
ax.set_ylabel('Percentage')
ax.set_title('Prediction Accuracy')
st.pyplot(fig)
