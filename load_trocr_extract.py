from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Chargement du modèle stage1
processor_stage1 = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model_stage1 = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# Fonctions d'accès
def get_trocr_stage1():
    return processor_stage1, model_stage1
