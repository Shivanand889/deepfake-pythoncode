from flask import Flask , jsonify ,request
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model , Sequential
import os
from PIL import Image
import random
from keras.models import load_model
# Define the directory to save the uploaded files

import cv2


def extract_faces(image_path, output_folder , name):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path, 1)
    gray_images = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale
    face_images = face_cascade.detectMultiScale(gray_images, 1.3, 5)

    try:
        for (x, y, w, h) in face_images:
            region_of_interest = image[y:y+h, x:x+w]
        resized = cv2.resize(region_of_interest, (128, 128))
        cv2.imwrite(f"{output_folder}/{name}.jpg", resized)    
    except:
        print("No Faces Detected.")
# Ensure the upload folder exists

app = Flask(__name__) 
CORS(app)
UPLOAD_FOLDER = 'src/public'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
image_dimensions = {'height':256, 'width':256, 'channels':3}
model = load_model("m_icept.h5")




@app.route('/api/model' , methods=['POST'] )
def predict() :
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        extract_faces(filepath,"src/public" ,"x") 
        extract_faces("src/public/x.jpg","src/public" ,"x") 
        filepath = 'src/public/x.jpg'
        image = Image.open(filepath)

        image = image.resize((224,224))
        x = []
        x = np.array(image)
        x = np.array(x)
        x = np.expand_dims(x, axis=0)
        print(filepath)
  
        # Perform prediction here
        data = float(model(x)[0][0])
        print(data)
        return jsonify({'me' : data}) 
        
    except Exception as e :
        return jsonify({'error': str(e)})