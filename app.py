from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import os

port = int(os.environ.get('PORT', 5000))  # 5000 là cổng mặc định nếu không có biến môi trường PORT

app = Flask(__name__)

# Load pre-trained ResNet50 model and saved features
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

with open('features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    return features.flatten()

# Helper function to convert numpy types to native Python types
def convert_to_native_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    if isinstance(data, (np.float32, np.float64)):
        return float(data)  # Convert numpy float to native Python float
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_native_types(i) for i in data]
    return data

# API route to receive image and return similar image
@app.route('/find_similar', methods=['POST'])
def find_similar():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = './temp_image.jpg'  # Save temporary file
    file.save(file_path)

    query_features = extract_features(file_path)

    max_similarity = -1
    most_similar_image_path = None

    for img_path, data_features in features_dict.items():
        similarity = cosine_similarity([query_features], [data_features])[0][0]
        similarity = convert_to_native_types(similarity)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image_path = img_path

    # Convert the max_similarity to a native Python float
    max_similarity = convert_to_native_types(max_similarity)

    return jsonify({'similar_image': most_similar_image_path, 'similarity': max_similarity})

# Route for the home page
@app.route('/')
def home():
    return 'Trang chủ'

if __name__ == '__main__':
    app.run(debug=True, port=port)  # Specify the port here
