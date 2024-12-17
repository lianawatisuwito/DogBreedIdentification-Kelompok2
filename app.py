from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("dogclassification.h5")

# Define class names
class_names = {
    "0": "Afghan",
    "1": "African Wild Dog",
    "2": "Airedale",
    "3": "American Hairless",
    "4": "American Spaniel",
    "5": "Basenji",
    "6": "Basset",
    "7": "Beagle",
    "8": "Bearded Collie",
    "9": "Bermaise",
    "10": "Bichon Frise",
    "11": "Blenheim",
    "12": "Bloodhound",
    "13": "Bluetick",
    "14": "Border Collie",
    "15": "Borzoi",
    "16": "Boston Terrier",
    "17": "Boxer",
    "18": "Bull Mastiff",
    "19": "Bull Terrier",
    "20": "Bulldog",
    "21": "Cairn",
    "22": "Chihuahua",
    "23": "Chinese Crested",
    "24": "Chow",
    "25": "Clumber",
    "26": "Cockapoo",
    "27": "Cocker",
    "28": "Collie",
    "29": "Corgi",
    "30": "Coyote",
    "31": "Dalmation",
    "32": "Dhole",
    "33": "Dingo",
    "34": "Doberman",
    "35": "Elk Hound",
    "36": "French Bulldog",
    "37": "German Sheperd",
    "38": "Golden Retriever",
    "39": "Great Dane",
    "40": "Great Perenees",
    "41": "Greyhound",
    "42": "Groenendael",
    "43": "Irish Spaniel",
    "44": "Irish Wolfhound",
    "45": "Japanese Spaniel",
    "46": "Komondor",
    "47": "Labradoodle",
    "48": "Labrador",
    "49": "Lhasa",
    "50": "Malinois",
    "51": "Maltese",
    "52": "Mex Hairless",
    "53": "Newfoundland",
    "54": "Pekinese",
    "55": "Pit Bull",
    "56": "Pomeranian",
    "57": "Poodle",
    "58": "Pug",
    "59": "Rhodesian",
    "60": "Rottweiler",
    "61": "Saint Bernard",
    "62": "Schnauzer",
    "63": "Scotch Terrier",
    "64": "Shar_Pei",
    "65": "Shiba Inu",
    "66": "Shih-Tzu",
    "67": "Siberian Husky",
    "68": "Vizsla",
    "69": "Yorkie"
}

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    uploaded_image_url = None  # URL for displaying the uploaded image

    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="No file uploaded. Please try again.")

        uploaded_file = request.files["file"]

        # Generate a unique filename to avoid overwriting
        filename = str(uuid.uuid4()) + os.path.splitext(uploaded_file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded file
        uploaded_file.save(filepath)

        # Convert the uploaded file to a BytesIO object and load the image
        img = load_img(filepath, target_size=(224, 224))  # Resize the image
        img_array = img_to_array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=-1)
        confidence = round(np.max(predictions) * 100, 2)  # Round to 4 decimal places

        result = class_names[str(predicted_class_index[0])]
        uploaded_image_url = '/uploads/' + filename  # Set URL for uploaded image

    return render_template("index.html", result=result, confidence=confidence, uploaded_image_url=uploaded_image_url)


# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True, port=8080)  # Change the port to 8080 or any other port you prefer
