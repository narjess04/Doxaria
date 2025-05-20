import os
import cv2
import numpy as np
from PIL import Image
import easyocr
import unicodedata
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from ultralytics import YOLO
import cv2
import torch
import requests
from load_trocr_extract import get_trocr_stage1

# Récupérer le modèle voulu
processor, model_trocr = get_trocr_stage1()  # ou get_trocr_handwritten()

# Constantes des chemins de modèles
MODEL_DETECTION_PATH = "models/mon_model.pt"
MODEL_TEXT_SEG_PATH = "models/text_seg.pt"

def detect_and_save(model_path, image_path, save_image_path, save_txt_path):
    model = YOLO(model_path)
    results = model(image_path)
    results[0].save(filename=save_image_path)
    with open(save_txt_path, "w") as f:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            f.write(f"{label} {conf:.2f} {x1} {y1} {x2} {y2}\n")

def reconstruct_image_from_labels(image_path, label_path, output_path, classes_voulues=None):
    if classes_voulues is None:
        classes_voulues = {"cin", "patient", "nom_adherent", "adher", "num_bulletin", "date_nai", "cnam"}

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    h, w, _ = img.shape
    reconstructed = np.zeros_like(img)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        label = parts[0]
        if label not in classes_voulues:
            continue
        x1, y1, x2, y2 = map(int, map(float, parts[2:]))
        reconstructed[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    cv2.imwrite(output_path, reconstructed)

def remove_text_with_yolo(model_path, image_path, output_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    h, w, _ = image.shape
    results = model(image_path)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        image[y1:y2, x1:x2] = 0
    cv2.imwrite(output_path, image)


def detect_and_remove_text_yolo_trocr(model_path, image_path, output_image_path, output_text_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    h, w, _ = image.shape
    results = model(image_path)[0]

    with open(output_text_path, 'w') as text_file:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, w - 1), min(y2, h - 1)

            cropped_image = image[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue

            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_image, return_tensors="pt")

            with torch.no_grad():
                generated_ids = model_trocr.generate(**inputs)
                predicted_text = processor.decode(generated_ids[0], skip_special_tokens=True)

            text_file.write(f"Position: ({x1}, {y1}), ({x2}, {y2}) - Texte: {predicted_text}\n")

            image[y1:y2, x1:x2] = 0  # suppression du texte

    cv2.imwrite(output_image_path, image)
    print(f"✅ Image sans texte sauvegardée sous : {output_image_path}")
    print(f"✅ Texte prédit sauvegardé sous : {output_text_path}")


def rebuild_text_image_without_accents(image_path, output_image_path):
    def is_numeric_only(text):
        return re.fullmatch(r'\d+(\.\d+)?', text.strip()) is not None

    def remove_accents(text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    image = cv2.imread(image_path)
    blank_image = np.ones_like(image) * 255
    reader = easyocr.Reader(['fr'])
    results = reader.readtext(image)
    for (bbox, text, _) in results:
        if not is_numeric_only(text):
            (x, y) = map(int, bbox[0])
            text_no_accents = remove_accents(text.strip())
            cv2.putText(blank_image, text_no_accents, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(output_image_path, blank_image)

def reinject_specific_elements(original_image_path, reconstructed_image_path, detection_file_path, output_image_path, labels_to_keep=('signa', 'tab1')):
    image_originale = Image.open(original_image_path).convert("RGB")
    image_reconstruite = Image.open(reconstructed_image_path).convert("RGB")
    with open(detection_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        label = parts[0]
        if label in labels_to_keep:
            x1, y1, x2, y2 = map(int, parts[2:6])
            cropped = image_originale.crop((x1, y1, x2, y2))
            image_reconstruite.paste(cropped, (x1, y1))
    image_reconstruite.save(output_image_path)

def overlay_text_from_file(image_path, text_file_path, output_image_path,
                           font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2,
                           color=(255, 0, 0), thickness=2):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    
    with open(text_file_path, 'r') as text_file:
        lines = text_file.readlines()

    for line in lines:
        parts = line.strip().split(' - Texte: ')
        if len(parts) != 2:
            continue

        position_part = parts[0].replace('Position: ', '').strip()
        text = parts[1].strip()

        position_coords = position_part.split('), (')
        x1, y1 = map(int, position_coords[0].replace('(', '').split(','))
        x2, y2 = map(int, position_coords[1].replace(')', '').split(','))

        box_width = x2 - x1
        box_height = y2 - y1

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x1 + (box_width - text_width) // 2
        text_y = y1 + (box_height + text_height) // 2

        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(output_image_path, image)
    print(f"✅ Image avec texte sauvegardée sous : {output_image_path}")
    return output_image_path



# Fonction principale appelée par Flask
def process_document(input_image_path):
    filename = os.path.basename(input_image_path)
    name, _ = os.path.splitext(filename)
    static_dir = "static/results"
    os.makedirs(static_dir, exist_ok=True)

    # Chemins intermédiaires
    detect_txt = os.path.join(static_dir, f"{name}_detect.txt")
    detected_img = os.path.join(static_dir, f"{name}_detected.jpg")
    reconstructed_img = os.path.join(static_dir, f"{name}_reconstructed.jpg")
    no_text_img = os.path.join(static_dir, f"{name}_no_text.jpg")
    no_accents_img = os.path.join(static_dir, f"{name}_no_accents.jpg")
    final_img = os.path.join(static_dir, f"{name}_final.jpg")
    final_with_text = os.path.join(static_dir, f"{name}_final_with_text.jpg")
    trocr_text_file = os.path.join(static_dir, f"{name}_text_output.txt")

    # Étape 1 : Détection des éléments
    detect_and_save(MODEL_DETECTION_PATH, input_image_path, detected_img, detect_txt)

    # Étape 2 : Reconstruction avec les labels utiles (cin, patient, etc.)
    reconstruct_image_from_labels(input_image_path, detect_txt, reconstructed_img)

    # Étape 3 : Suppression du texte avec détection + OCR (TrOCR) + enregistrement texte
    detect_and_remove_text_yolo_trocr(MODEL_TEXT_SEG_PATH, reconstructed_img, no_text_img, trocr_text_file)

    # Étape 4 : Réécriture du texte sans accents (EasyOCR + réinjection sur fond blanc)
    rebuild_text_image_without_accents(no_text_img, no_accents_img)

    # Étape 5 : Réinjection des éléments spécifiques comme les tableaux ou signatures
    reinject_specific_elements(input_image_path, no_accents_img, detect_txt, final_img)

    # Étape 6 : Superposition du texte reconnu par TrOCR sur l’image finale
    if os.path.exists(trocr_text_file):
        overlay_text_from_file(final_img, trocr_text_file, final_with_text)
    else:
        final_with_text = final_img  # fallback si le fichier texte n'existe pas

    return {
        "text_path": detect_txt,
        "image_cleaned": final_img,
        "image_with_text": final_with_text
    }
