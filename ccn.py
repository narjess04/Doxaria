import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Définition de l'architecture
class OrdonnanceClassifierCNN(nn.Module):
    def __init__(self):
        super(OrdonnanceClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes : ordonnance ou bulletin

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 32, 32]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 128, 16, 16]
        x = self.dropout(x)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Création du modèle et chargement des poids
model = OrdonnanceClassifierCNN()
model_path = 'models/modele_cnn_ordonnance_bulletin.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Les classes (dans l'ordre où ton modèle a été entraîné)
classes = ['Bulletin','Ordonnaces']

# Fonction de prédiction
def predict_image_class(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # car ton modèle attend du 128x128 en entrée
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

if __name__ == "__main__":
    image_path = "uploads/0730--8979909--20230705_page_4.jpg"
    result = predict_image_class(image_path)
    print(f"Classe prédite : {result}")

