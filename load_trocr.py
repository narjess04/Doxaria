# models/trocr_loader.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Chargement unique du modèle TrOCR manuscrit
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# On exporte pour l'utiliser ailleurs
def get_trocr_model():
    return processor, model
