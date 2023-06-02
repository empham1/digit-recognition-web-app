import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from joblib import load

class DigitRecognizer:
    def __init__(self, model_path):
        # load model
        self.model = load(model_path)

    def preprocess_image(self, image):
        # Resize image to required input size of model
        resized_image = image.resize((8, 8))
        # Convert image to grayscale
        grayscale_image = ImageOps.invert(resized_image.convert('L'))
        # Convert image to numpy array
        numpy_data = np.array(grayscale_image)
        # Flatten image into a 1D array
        flattened_image = numpy_data.flatten()
        # Reshape flattened image so model can accept single input
        reshaped_image = flattened_image.reshape(1, -1)
        # Return reshaped image
        return reshaped_image

    def predict_digit(self, image):
        # preprocess the image
        preprocessed_image = self.preprocess_image(image)
        # use object's model to predict digit
        prediction = self.model.predict(preprocessed_image)
        # return predicted digit
        return prediction[0]

    
def main():
    # write title
    st.title('Handwritten Digit Recognition')
    # Create an instance of DigitRecognizer class
    digit_recognizer = DigitRecognizer('model.joblib')
    # Upload image file
    uploaded_file = st.file_uploader('Please upload an image of a digit:', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # Load uploaded image
        image = Image.open(uploaded_file)
        # Display uploaded image
        st.image(image, use_column_width=False)
        # Make prediction using DigitRecognizer object
        digit_prediction = digit_recognizer.predict_digit(image)
        # Display the predicted digit
        st.write('Predicted Digit:', digit_prediction)


if __name__ == '__main__':
    main()
