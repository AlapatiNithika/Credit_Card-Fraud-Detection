import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the credit card transaction dataset from an external source
@st.cache
def load_data():
    data = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
    return data

data = load_data()

# Filter and balance the dataset by separating legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample the legitimate transactions to create a balanced dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split the dataset into feature matrix (X) and target labels (y)
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split the data into training and testing sets, ensuring stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Initialize and train a logistic regression model for fraud detection
model = LogisticRegression()
model.fit(X_train, y_train)

# Create a Streamlit web application interface
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features separated by commas to check if the transaction is legitimate or fraudulent:")
st.write("For example, enter '0.1,0.2,0.3,...'")

# Create a text input field for users to input all feature values
input_features = st.text_input('Input All features')

# Create a button for users to submit their input and obtain a prediction
submit = st.button("Submit")

# Make a prediction based on user input when the submit button is clicked
if submit:
    # Split the input string into a list of feature values
    input_features_list = input_features.split(',')
    # Convert the list of strings to a NumPy array of floats
    features = np.array(input_features_list, dtype=np.float64).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Display the prediction result along with the model's confidence
    if prediction[0] == 0:
        st.write("Legitimate transaction with confidence:", probability[0][0])
    else:
        st.write("Fraudulent transaction with confidence:", probability[0][1])

# Display the training and testing accuracy of the logistic regression model
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Testing Accuracy: {test_acc:.2f}")
