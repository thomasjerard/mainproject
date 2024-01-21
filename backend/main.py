from flask import Flask, request
from ultralytics import YOLO
from PIL import Image
import io
import waitress
import json

app = Flask(__name__)
model = YOLO("best.pt")  # Load the YOLOv8 model

@app.route("/detect", methods=["POST"])
def detect_objects():
    image = request.files["image"].read()
    img = Image.open(io.BytesIO(image))  # Load the image from request
    results = model(img)
    detections = results.pandas().xyxy[0].to_json(orient="records")  # Convert results to JSON
    return detections, 200

if __name__ == "__main__":
    app.run(debug=True)
