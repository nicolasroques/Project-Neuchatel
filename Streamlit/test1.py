


import streamlit as st
from joblib import load

import streamlit as st
#from sklearn.pipeline import Pipeline
# Other necessary imports for your model

# Load your trained model (the best model from GridSearchCV)
# You should save your trained model to a file after training and then load it here.
best_model = load('model_lr.pkl')

# Define a function to make predictions
def predict_language_level(text, model):
    # Preprocess the text if necessary
    # Make a prediction
    return model.predict([text])[0]

# Streamlit UI
def main():
    st.title("Language Level Predictor")
    st.write("Enter a text to determine its language proficiency level.")

    # Text input
    text_input = st.text_area("Input your text here:")

    if st.button("Predict"):
        if text_input:
            # Predict the language level
            result = predict_language_level(text_input, best_model)
            st.write(f"The predicted language level of the text is: {result}")
        else:
            st.write("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
