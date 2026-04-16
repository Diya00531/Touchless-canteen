from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("food_model.h5")

# Load class names
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
classes = {v: k for k, v in class_indices.items()}

print("Classes Loaded:", classes)

# Prices
prices = {
    "Chaat": 30,
    "Maggie": 30,
    "Noodles": 70,
    "Paratha": 50,
    "Samosa": 20
}

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    files = request.files.getlist("file")

    detected_items = []
    total_bill = 0

    for file in files:

        if file.filename == "":
            continue

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        print("\nProcessing:", file.filename)

        # Load image
        img = Image.open(path).convert("RGB")
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        img = img.reshape(1, 128, 128, 3)

        # Predict
        prediction = model.predict(img)

        print("Prediction:", prediction)

        class_index = np.argmax(prediction)
        print("Class Index:", class_index)

        food_name = classes[class_index]
        print("Food Name:", food_name)

        price = prices.get(food_name, 0)
        print("Price:", price)

        total_bill += price

        detected_items.append({
            "name": food_name,
            "price": price
        })

    print("Total Bill:", total_bill)

    return render_template(
        "index.html",
        items=detected_items,
        total=total_bill
    )


if __name__ == "__main__":
    app.run(debug=True)