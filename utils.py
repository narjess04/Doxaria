import os
import cv2
import pytesseract
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

# Charger une seule fois le modèle
MODEL_PATH = "models/fraud_detection_.pt"
model = YOLO(MODEL_PATH)

def extract_stamps(image_path, label_path, stamp_class_id=0):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    img_height, img_width = img.shape[:2]

    if not os.path.exists(label_path):
        raise ValueError(f"Fichier d'annotations non trouvé : {label_path}")
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    extracted_texts = []
    last_text = ""

    for i, line in enumerate(lines):
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        if int(class_id) == stamp_class_id:
            # Convertir les coordonnées YOLO vers pixels
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            stamp_crop = img[y1:y2, x1:x2]

            text = pytesseract.image_to_string(stamp_crop)
            extracted_texts.append(text)
            last_text = text

    return extracted_texts, last_text


def extract_and_compare_stamps(image_path, label_path, reference_text, stamp_class_id=1):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    img_height, img_width = img.shape[:2]

    if not os.path.exists(label_path):
        raise ValueError(f"Fichier d'annotations non trouvé : {label_path}")
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    comparisons = []

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        if int(class_id) == stamp_class_id:
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            stamp_crop = img[y1:y2, x1:x2]

            text = pytesseract.image_to_string(stamp_crop)

            # Calculer la similarité
            vectorizer = TfidfVectorizer().fit_transform([reference_text, text])
            similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

            comparisons.append({
                'text': text,
                'similarity': similarity
            })

    return comparisons


def detect_fraud(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    if not os.path.exists(label_path):
        raise ValueError(f"Fichier d'annotations non trouvé : {label_path}")

    with open(label_path, 'r') as f:
        lines = f.readlines()

    has_stamp_class_1 = any(int(line.strip().split()[0]) == 1 for line in lines)

    if not has_stamp_class_1:
        print("Aucun stamp de classe 1 trouvé → FRAUDE")
        return "Fraud"

    extracted_texts, main_text = extract_stamps(image_path, label_path, stamp_class_id=0)

    results = extract_and_compare_stamps(image_path, label_path, main_text, stamp_class_id=1)

    for result in results:
        if result['similarity'] * 100 > 40:
            print("Similarité suffisante → PAS DE FRAUDE")
            return "Pas de fraude"

    print("Aucune similarité suffisante → FRAUDE")
    return "Fraud"


def predict_and_save(image_path):
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        conf=0.25
    )

    for result in results:
        original_path = result.path
        labels = [model.names[int(cls)] for cls in result.boxes.cls]

    # Récupérer la dernière image annotée
    annotated_images = sorted(glob.glob("runs/detect/predict*/*.jpg"))
    predicted_img_path = annotated_images[-1] if annotated_images else None

    return labels, predicted_img_path
