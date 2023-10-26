#import required modules
import streamlit as app_interface
import numpy as np_matrixlib
import tensorflow as tf_framework
from PIL import Image

# Initialize the tflite model
model_directory = "primary.tflite"
model_interpreter = tf_framework.lite.Interpreter(model_path=model_directory)
model_interpreter.allocate_tensors()

# Retrieve input-output information
input_info = model_interpreter.get_input_details()
output_info = model_interpreter.get_output_details()
input_dimensions = input_info[0]['shape']
output_dimensions = output_info[0]['shape']
input_type = input_info[0]['dtype']
output_type = output_info[0]['dtype']

# Specify labels
labels = ['Covid', 'Viral Pneumonia', 'Normal']

app_interface.set_page_config(page_title="Chest X-Ray Diagnostic Tool", layout="wide")
section1, section2 = app_interface.columns([1, 1])  # Arrange the interface in two parts

# Section for uploading images
with section1:
    app_interface.title('Chest X-Ray Diagnostic Tool')
    app_interface.markdown('<h3 style="font-weight:normal;">Utilize this tool to classify Chest X-ray images into categories: COVID-19, Viral Pneumonia, or Normal. Clear images yield better outcomes.</h3>', unsafe_allow_html=True)

    # Upload mechanism
    file_to_upload = app_interface.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

    # Image analysis
    if file_to_upload:
        # Adjust and preprocess image
        loaded_image = Image.open(file_to_upload)
        if loaded_image.mode != "RGB":
            loaded_image = loaded_image.convert("RGB")  # Adjust to RGB mode if different
        loaded_image = loaded_image.resize((input_dimensions[1], input_dimensions[2]))
        processed_image = np_matrixlib.array(loaded_image, dtype=np_matrixlib.float32)
        processed_image /= 255.0
        processed_image = np_matrixlib.expand_dims(processed_image, axis=0)

        # Function to categorize image
        def categorize(input_img):
            model_interpreter.set_tensor(input_info[0]['index'], input_img.astype(input_type))
            model_interpreter.invoke()
            results = model_interpreter.get_tensor(output_info[0]['index'])
            category_index = np_matrixlib.argmax(results, axis=1)
            category_name = labels[category_index[0]]
            return category_name
        
        # Display results instantly
        category_result = categorize(processed_image)
        app_interface.markdown(f"<h3>Result: <span style='font-style: italic; font-weight: bold;'>{category_result}</span></h3>", unsafe_allow_html=True)

# Section to show uploaded image
with section2:
    if file_to_upload:
        app_interface.image(loaded_image, caption="Provided Image", use_column_width=True)
