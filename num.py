# num.py
from ultralytics import YOLO
from PIL import Image
import pytesseract
import re
model_path = "models/mon_model.pt"

def detect_and_save(model_path, image_path, save_image_path, save_txt_path):
    model = YOLO(model_path)
    results = model(image_path)
    results[0].save(filename=save_image_path)

    with open(save_txt_path, "w") as f:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            f.write(f"{label} {conf:.2f} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}\n")

def extract_bulletin_numbers(detections_path, image_path):
    bulletin_numbers = []

    # Lire le fichier de détection
    with open(detections_path, "r") as f:
        lines = f.readlines()

    # Charger l'image une fois
    image = Image.open(image_path)

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue  # ignorer les lignes mal formatées

        label, conf, x1, y1, x2, y2 = parts
        if label != "num_bulletin":
            continue

        # Convertir les coordonnées
        x1, y1, x2, y2 = map(lambda v: int(float(v)), [x1, y1, x2, y2])

        # Rogner l'image
        cropped = image.crop((x1, y1, x2, y2))

        # OCR
        raw_text = pytesseract.image_to_string(cropped, config="--psm 6")

        # Garder uniquement les chiffres
        digits_only = re.sub(r"\D", "", raw_text)

        if digits_only:
            bulletin_numbers.append(digits_only)

    return bulletin_numbers
