from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "SINet-V2 Backend is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
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
    app.run(debug=False, use_reloader=False)
