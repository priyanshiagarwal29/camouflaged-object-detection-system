from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import subprocess
import gdown

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
WEIGHT_PATH = "snapshot/SINet_V2/Net_epoch_best.pth"
WEIGHT_URL = "https://drive.google.com/uc?id=1SWzKqDCK3i8m0vjMojZPLU9fKNWqR8Bg"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("snapshot/SINet_V2", exist_ok=True)

if not os.path.exists(WEIGHT_PATH):
    print("Downloading model weights...")
    gdown.download(WEIGHT_URL, WEIGHT_PATH, quiet=False)


@app.route("/")
def home():
    return "SINet-V2 Backend is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file received"}), 400

        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        subprocess.run([
            "python",
            "infer_single.py",
            "--image",
            image_path
        ], check=True)

        result_folder = os.path.join("res", "single")

        files = [
            f for f in os.listdir(result_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        latest_file = max(
            files,
            key=lambda f: os.path.getmtime(os.path.join(result_folder, f))
        )

        result_path = os.path.join(result_folder, latest_file)

        return send_file(result_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False)
