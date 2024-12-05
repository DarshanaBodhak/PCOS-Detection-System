from flask import Flask, request, jsonify, render_template
from model_loader import ModelLoader  # Import ModelLoader class
from src.prediction import predict_pcos
import src.feature_extraction
import os

# Initialize Flask app
app = Flask(__name__)

# Temporary directory for uploaded images
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load models at the start of the application
resnet_model, vggnet_model, xception_model, ensemble_model = ModelLoader.load_models()

@app.route("/")
def home():
    return render_template("index.html")  # Serves the frontend HTML

@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Save the file temporarily
    image_path = os.path.join(TEMP_DIR, image_file.filename)
    image_file.save(image_path)

    try:
        # Call the predict_pcos function using pre-loaded models
        prediction = predict_pcos(image_path, resnet_model, vggnet_model, xception_model, ensemble_model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup: Remove the saved file
        if os.path.exists(image_path):
            os.remove(image_path)

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
