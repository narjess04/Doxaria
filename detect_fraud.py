import os
import cv2
import time
import requests
import numpy as np
import pytesseract
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====== CONFIG AZURE OCR ======
AZURE_ENDPOINT = "https://npl-ocr.cognitiveservices.azure.com/"
AZURE_API_KEY = "9NV7bUaKBwO4I8yAolJYADwhEgNZhH7CVpRYDwtOv3BzbsJcqddDJQQJ99BBAC5RqLJXJ3w3AAAFACOGsQ8R"
OCR_URL = f"{AZURE_ENDPOINT}vision/v3.1/read/analyze?language=fr"
headers = {
    "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
    "Content-Type": "application/octet-stream"
}

# ====== CHARGER MODELE YOLO ======
model = YOLO("models/fraud_detection_.pt")  # à adapter si chemin différent

def azure_ocr(image):
    _, img_encoded = cv2.imencode('.png', image)
    image_data = img_encoded.tobytes()
    try:
        response = requests.post(OCR_URL, headers=headers, data=image_data)
        response.raise_for_status()
        operation_url = response.headers.get("Operation-Location")
        if not operation_url:
            return []

        time.sleep(5)
        while True:
            result = requests.get(operation_url, headers=headers).json()
            if result.get("status") == "succeeded":
                break
            time.sleep(2)

        extracted_text = []
        for page in result["analyzeResult"]["readResults"]:
            for line in page["lines"]:
                extracted_text.append(line["text"])
        return extracted_text
    except:
        return []

def run_yolo_and_extract(image_path):
    img = cv2.imread(image_path)
    results = model.predict(image_path, conf=0.3)[0]  # résultat YOLO unique
    img_height, img_width = img.shape[:2]

    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # convert to YOLO format (normalized)
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        detections.append({
            "class_id": class_id,
            "coords": (x1, y1, x2, y2),
            "yolo": (class_id, x_center, y_center, width, height)
        })
    return detections, img

def extract_text_by_class(detections, img, target_class=0, method="tesseract"):
    for det in detections:
        if det["class_id"] == target_class:
            x1, y1, x2, y2 = det["coords"]
            crop = img[y1:y2, x1:x2]
            if method == "tesseract":
                return pytesseract.image_to_string(crop).strip()
    return ""

def compare_stamps_with_doc(detections, img, text_doc, stamp_class_id=1):
    vectorizer = TfidfVectorizer()
    similarities = []

    for det in detections:
        if det["class_id"] == stamp_class_id:
            x1, y1, x2, y2 = det["coords"]
            crop = img[y1:y2, x1:x2]
            extracted_lines = azure_ocr(crop)
            text_stamp = " ".join(extracted_lines)

            if not text_stamp or not text_doc:
                continue

            vectors = vectorizer.fit_transform([text_stamp, text_doc])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            similarities.append(similarity)
    return similarities

def detect_fraud(image_path, label_path=None):
    detections, img = run_yolo_and_extract(image_path)

    has_stamp = any(d['class_id'] == 1 for d in detections)
    if not has_stamp:
        return "Fraud detected"

    text_doc = extract_text_by_class(detections, img, target_class=0, method="tesseract")
    if not text_doc.strip():
        return "Fraud detected"

    similarities = compare_stamps_with_doc(detections, img, text_doc, stamp_class_id=1)
    for sim in similarities:
        if sim * 100 > 1:
            return "No fraud"
    return "Fraud detected"
