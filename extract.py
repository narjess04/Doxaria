import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from load_trocr import get_trocr_model

# Récupérer le modèle voulu
processor, model_trocr = get_trocr_model()  # ou get_trocr_handwritten()

def merge_boxes(boxes, iou_threshold=0.5):
    # Fonction dummy, à adapter si tu veux fusionner les boxes qui se chevauchent
    return boxes  # Ici on retourne tel quel

def detect_and_save(model_path, image_path, save_image_path, save_txt_path):
    model = YOLO(model_path)
    results = model(image_path)
    results[0].save(filename=save_image_path)

    boxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append({'label': label, 'conf': conf, 'coords': [x1, y1, x2, y2]})

    merged_boxes = merge_boxes(boxes)

    with open(save_txt_path, "w") as f:
        for box in merged_boxes:
            x1, y1, x2, y2 = box['coords']
            f.write(f"{box['label']} {box['conf']:.2f} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}\n")

    return merged_boxes

def extract_doctor_name(image_path, boxes, output_txt_path, output_img_path):
    doctor_boxes = [box for box in boxes if box['label'] == 'docteur']
    if not doctor_boxes:
        return

    top_doctor_box = min(doctor_boxes, key=lambda b: b['coords'][1])
    x1, y1, x2, y2 = map(int, top_doctor_box['coords'])

    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    cropped = image[y1:y2, x1:x2]

    reader = easyocr.Reader(['fr'], gpu=False)
    results = reader.readtext(cropped)
    extracted_texts = [text for (_, text, conf) in results if conf > 0.5]

    if extracted_texts:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for text in extracted_texts:
                f.write(text.strip() + "\n")

    blank_image = np.ones((original_height, original_width, 3), dtype=np.uint8) * 255
    ch, cw = cropped.shape[:2]

    if ch > original_height or cw > original_width:
        cropped = cv2.resize(cropped, (original_width, original_height))
        ch, cw = cropped.shape[:2]

    blank_image[0:ch, 0:cw] = cropped
    cv2.imwrite(output_img_path, blank_image)

def detect_and_recognize_text(model_path, image_path, save_image_path, save_txt_path):
    model = YOLO(model_path)
    image = Image.open(image_path).convert("RGB")
    results = model(image_path)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    with open(save_txt_path, "w", encoding="utf-8") as f_txt:
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
            cropped_img = image.crop((x1, y1, x2, y2))

            pixel_values = processor(images=cropped_img, return_tensors="pt").pixel_values
            generated_ids = model_trocr.generate(pixel_values)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            conf = float(box.conf[0].item())

            f_txt.write(f"{label} {conf:.2f} {x1} {y1} {x2} {y2} : {transcription}\n")
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            draw.text((x1, max(0, y1 - 20)), transcription, fill="red", font=font)

    image.save(save_image_path)

def sort_and_save_only_text(input_txt_path, output_txt_path):
    texts_with_positions = []

    with open(input_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                x1 = int(parts[2])
                y1 = int(parts[3])
                text = " ".join(parts[6:])
                texts_with_positions.append((y1, x1, text))
            except ValueError:
                continue

    sorted_texts = sorted(texts_with_positions, key=lambda t: (t[0], t[1]))

    with open(output_txt_path, "w", encoding="utf-8") as f_out:
        for _, _, text in sorted_texts:
            cleaned_text = text.lstrip(":").strip()
            f_out.write(cleaned_text + "\n")

# 🔁 Fonction globale pour Flask
def process_image(image_path, nom_docteur_path, ocr_sorted_text_path):
    temp_detections_path = "processed/detections.txt"
    temp_predicted_path = "processed/predicted.jpg"
    temp_predicted_with_text = "processed/predicted_with_text.jpg"
    temp_detection_with_text = "processed/detections_with_text.txt"
    crop_output_path = "processed/a4_crop_docteur.jpg"

    # 1. Détection docteur
    boxes = detect_and_save(
        model_path="models/fraud_detection_.pt",
        image_path=image_path,
        save_image_path=temp_predicted_path,
        save_txt_path=temp_detections_path
    )

    # 2. Extraction nom docteur
    extract_doctor_name(
        image_path=image_path,
        boxes=boxes,
        output_txt_path=nom_docteur_path,
        output_img_path=crop_output_path
    )

    # 3. OCR sur zones détectées
    detect_and_recognize_text(
        model_path="models/multi_line1.pt",
        image_path=image_path,
        save_image_path=temp_predicted_with_text,
        save_txt_path=temp_detection_with_text
    )

    # 4. Tri du texte
    sort_and_save_only_text(
        input_txt_path=temp_detection_with_text,
        output_txt_path=ocr_sorted_text_path
    )
