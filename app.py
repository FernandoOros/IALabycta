from flask import Flask, render_template, request, jsonify, session, send_from_directory
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import numpy as np
import json
import os
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current working directory
cwd = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for the model and class indices
model_path = os.path.join(cwd, 'model', 'imageModel.h5')
class_indices_path = os.path.join(cwd, 'classes', 'classIndices.json')
images_path = os.path.join(cwd, 'images')

# Ensure the images directory exists
os.makedirs(images_path, exist_ok=True)

# Load the model
try:
    model = load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load class indices from the JSON file
try:
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
except FileNotFoundError as e:
    logger.error(f"Class indices file not found: {e}")
    class_indices = {}

def predict_image(image_path, model, class_indices):
    if model is None:
        return {'error': 'Model not loaded'}, 500

    # Load and preprocess the image using Pillow
    try:
        img = Image.open(image_path).convert('RGB').resize((256, 256))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale the image

        # Predict
        predictions = model.predict(img_array)[0]

        # Get the indices of the top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_probabilities = predictions[top_3_indices]
        top_3_classes = [list(class_indices.keys())[i] for i in top_3_indices]

        # Prepare prediction result
        result = [
            {'class': top_3_classes[i], 'probability': float(top_3_probabilities[i])}
            for i in range(3)
        ]

        return result
    except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        return {'error': 'Prediction failed'}, 500

@app.route('/', methods=["GET"])
def hello():
    return render_template('index.html')

@app.route('/', methods=["POST"])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    imageFile = request.files.get('imageFile')
    if not imageFile:
        return jsonify({'error': 'No image file provided'}), 400

    # Delete the previous image if exists
    previous_image_path = session.pop('image_path', None)
    if previous_image_path and os.path.exists(previous_image_path):
        try:
            os.remove(previous_image_path)
            logger.info(f"Previous image {previous_image_path} removed successfully")
        except Exception as e:
            logger.error(f"Error removing previous image: {e}")

    image_path = os.path.join(images_path, imageFile.filename)
    session['image_path'] = image_path

    try:
        imageFile.save(image_path)
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        return jsonify({'error': f'Permission denied: {e}'}), 500
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return jsonify({'error': 'Failed to save image'}), 500

    # Get predictions
    predictions = predict_image(image_path, model, class_indices)

    if 'error' in predictions:
        return jsonify(predictions), 500

    image_url = f'/images/{imageFile.filename}'
    return render_template('index.html', predictions=predictions, image_url=image_url)

@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(images_path, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'False') == 'True')
