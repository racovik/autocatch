import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ----- CONFIG -----
EMBEDDINGS_PATH = "pokedex_embeddings-w-pokeapi.pt"
IMAGE_TESTE = "test/teste.png"  # imagem que você quer identificar
DEVICE = "cpu"  # ou "cuda"/"rocm" se estiver usando GPU
IMAGE_SIZE = 224

# ----- TRANSFORM (TEM QUE SER IGUAL AO TREINO) -----
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Normalização padrão do ImageNet (Média e Desvio Padrão)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# ----- MODELO BASE -----
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.eval()
model.to(DEVICE)

# ----- LOAD DOS EMBEDDINGS -----
base_embeddings = torch.load(EMBEDDINGS_PATH, map_location=DEVICE, weights_only=False)


# ----- FUNÇÃO DE EMBEDDING -----
# def get_embedding(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = transform(img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         emb = model(img)
#         emb = F.normalize(emb, dim=1)
#     return emb.squeeze(0)


def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze().cpu()
    emb = emb / np.linalg.norm(emb)  # normalização
    return emb


# ----- EMBEDDING DA IMAGEM TESTE -----
test_emb = get_embedding(IMAGE_TESTE)

# ----- COMPARAÇÃO -----
melhor_pokemon = None
melhor_score = -1

for nome, emb in base_embeddings.items():
    score = F.cosine_similarity(test_emb, emb, dim=0).item()
    if score > melhor_score:
        melhor_score = score
        melhor_pokemon = nome

print(f"Pokémon identificado: {melhor_pokemon}")
print(f"Similaridade: {melhor_score:.4f}")
