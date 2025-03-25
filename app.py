import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
TFLITE_MODEL_PATH = "model/pneumonia_model.tflite"

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input & output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    """Preprocess image for TFLite model (ResNet50 input format)."""
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array.astype(np.float32)  # Ensure correct data type for TFLite

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Validate file type
    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

    image_path = "temp_image.png"
    file.save(image_path)

    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor (prediction)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]  # Extracting prediction value

    confidence = max(prediction, 1 - prediction)  # Get confidence score
    os.remove(image_path)  # Clean up temp file

    result = "Pneumonia Detected" if prediction > 0.7 else "Normal"
    return jsonify({'prediction': result, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)

