import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# --- Configuration ---
# Define the path where your trained model is located.
# MAKE SURE 'best_waste_model.h5' IS IN THE SAME FOLDER AS app.py
MODEL_PATH = 'best_waste_model.h5'
UPLOAD_FOLDER = 'static/uploads' # Folder to temporarily save uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained Keras model
# It's good practice to load the model once when the app starts
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Define your class names in the correct order (alphabetical, as per ImageDataGenerator)
# !!! IMPORTANT: YOU MUST REPLACE THIS LIST WITH YOUR ACTUAL 12 CLASS NAMES IN ALPHABETICAL ORDER !!!
# Use the output from `print(train_generator.class_indices)` you got in Colab to get the exact order.
CLASS_NAMES =  ['batteries', 'biological', 'brown glass', 'cardboard', 'clothes', 'green glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'vegetable waste', 'white glass'] #
    # Example - REPLACE THESE WITH YOUR ACTUAL 12 CLASS NAMES
    


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# New Welcome Page Route (this will be your main page)
@app.route('/')
def welcome():
    return render_template('welcome.html') # Render the new welcome.html

# Existing Classification Page Route (changed from '/' to '/classify')
# ... (your existing code above, including @app.route('/') and @app.route('/classify')) ...
@app.route('/classify')
def classify_page(): # I've changed the function name to clarify, but 'index' works too
    return render_template('index.html') # This renders your original index.html content


# New About Us route
@app.route('/about')
def about():
    return render_template('about.html')

# New Contact route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# ... (your existing @app.route('/predict', methods=['POST']) and everything below it) ...

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        try:
            # Preprocess the image
            img = image.load_img(filename, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            img_array = img_array / 255.0 # Normalize pixel values

            # Make prediction
            predictions = model.predict(img_array)[0]
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(predictions[predicted_class_index]) * 100

            return jsonify({
                'class': predicted_class_name,
                'confidence': f"{confidence:.2f}%",
                'image_url': f"/{filename}" # URL to display the uploaded image
            })
        except Exception as e:
            return jsonify({'error': f"Prediction failed: {e}"}), 500
    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg, gif'}), 400

if __name__ == '__main__':
    app.run(debug=True) # debug=True allows for automatic reloading on code changes
