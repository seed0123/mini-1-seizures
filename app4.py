import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler

# Function to preprocess the target variable to binary
def toBinary(x):
    if x != 1:
        return 0
    else:
        return 1

# Custom CSS styling
def set_custom_style():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;  /* Set font family for the entire app */
            background-color: #f7f7f7;  /* Set a light gray background color */
            color: #333;  /* Set the default text color to dark gray */
            margin: 0;  /* Remove default margin */
            padding: 0; /* Remove default padding */
        }
        .title {
            font-size: 3rem;  /* Larger font size for the title */
            font-weight: bold;
            text-align: center;
            color: #007bff;  /* Set the title color to blue */
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        .file-upload {
            padding: 2rem;
            background-color: #ffffff;  /* White background for file uploader */
            border: 2px dashed #007bff;  /* Dashed border with blue color */
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction {
            font-size: 1.5rem;  /* Larger font size for prediction outcome */
            text-align: center;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .centered-image-container {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .centered-image {
            max-width: 100%;
            height: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function to run the Streamlit app
def main():
    set_custom_style()  # Apply custom CSS styles

    st.markdown('<h1 class="title">Epileptic Seizure Recognition App</h1>', unsafe_allow_html=True)

    st.markdown(
        """
        Welcome to the Epileptic Seizure Recognition App! This app allows you to analyze seizure status based on EEG data.
        Upload your CSV file containing EEG data, and the app will predict whether the sample indicates a pre-ictal seizure or not.

        <p>Understanding and predicting seizure events can be crucial for medical diagnosis and treatment.</p>
        """,
        unsafe_allow_html=True
    )

    # # Centered image container
    # st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
    # st.image("download.jpeg", caption="Image source: Your Source Here", use_column_width=False, 
    #          output_format='auto')  # Centered image with caption
    # st.markdown('</div>', unsafe_allow_html=True)

    # File uploader to upload the CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", 
                                      help="Please upload a CSV file with your data.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Preprocessing
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        y = y.apply(toBinary)

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Standardize features
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Reshape data for LSTM input (assuming each sample is a sequence of features)
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        # Model training - LSTM
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, X.shape[1]), activation="relu", return_sequences=True))
        model.add(LSTM(32, activation="sigmoid"))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=10, batch_size=32)

        # Make prediction on a single data point (first data point in test set)
        single_data_point = x_test[0:1]  # Get the first data point from test set
        prediction = model.predict(single_data_point)

        # Display prediction outcome
        st.markdown('<div class="prediction">', unsafe_allow_html=True)
        if prediction[0][0] >= 0.5:
            st.write("<p>Prediction: <strong>Pre-Ictal (Chances for attacking seizures is High)</strong></p>", unsafe_allow_html=True)
        else:
            st.write("<p>Prediction: <strong>Non Pre-Ictal (Chances for attacking seizures is less)</strong></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
