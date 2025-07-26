from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os

# Initialize app and model
app = Flask(_name_)
model = load_model("/content/fruit_classifier_mobilenetv2.h5")

# Define classes
class_names = ['APPLE', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']

# Read nutrition data
nutrition_df = pd.read_csv("nutrition_data.csv")

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    nutrition_info = None

    if request.method == "POST":
        img_file = request.files["image"]

        if img_file:
            img_filename = img_file.filename
            img_path = os.path.join(UPLOAD_FOLDER, img_filename)
            img_file.save(img_path)

            # Preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            class_idx = np.argmax(preds[0])
            prediction = class_names[class_idx]

            # Fetch nutrition info (case insensitive match)
            nutrition_row = nutrition_df[nutrition_df['Food'].str.upper() == prediction]
            if not nutrition_row.empty:
                nutrition_info = nutrition_row.to_dict('records')[0]

    return render_template("index.html", prediction=prediction, img_path=img_path, nutrition=nutrition_info)

if _name_ == "_main_":
    app.run(debug=True)
