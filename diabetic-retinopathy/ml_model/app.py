# ml_model/app.py
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image, ImageDraw
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import os
from model import predict

app = Flask(__name__)
CORS(app)

# Function to annotate image based on model predictions
def annotate_image(image_path, prediction):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), prediction, fill='red')
    annotated_image_path = f'./static/uploads/{os.path.basename(image_path).split(".")[0]}_annotated.png'
    img.save(annotated_image_path)
    return annotated_image_path

# Route for handling image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    upload_folder = './static/uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    prediction = predict(file_path)
    annotated_image_path = annotate_image(file_path, prediction)
    annotated_image_name = os.path.basename(annotated_image_path)
    
    return jsonify({
        'filename': file.filename,
        'prediction': prediction,
        'annotated_image_url': f'http://localhost:5001/static/uploads/{annotated_image_name}'
    })

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_file(os.path.join('./static/uploads', filename))

# Route to render the HTML form
#@app.route('/')
#def index():
    #return render_template('index.html')no

if __name__ == '__main__':
    app.run(port=5001, debug=True)
