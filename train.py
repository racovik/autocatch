import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# =============================
# Configurações
# =============================
BASE_DIR = "dataset/base"
TEST_IMAGE = "test/teste.png"
IMAGE_SIZE = 224
torch.backends.nnpack.enabled = False


device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Transformações
# =============================
transform = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
)

# =============================
# Modelo (extrator de embeddings)
# =============================
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Identity()  # remove a cabeça de classificação
model.eval()
model.to(device)


# =============================
# Função para gerar embedding
# =============================
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze().cpu()
    emb = emb / np.linalg.norm(emb)  # normalização
    return emb


# =============================
# Criar banco de embeddings
# =============================
base_embeddings = {}

for pokemon in os.listdir(BASE_DIR):
    pokemon_dir = os.path.join(BASE_DIR, pokemon)
    if not os.path.isdir(pokemon_dir):
        continue

    images = os.listdir(pokemon_dir)
    if len(images) == 0:
        continue

    img_path = os.path.join(pokemon_dir, images[0])
    base_embeddings[pokemon] = get_embedding(img_path)


torch.save(base_embeddings, "pokedex_embeddings-wnumpy.pt")

print(f"[OK] {len(base_embeddings)} Pokémon carregados.")


# =============================
# Identificação por similaridade
# =============================
def identify_pokemon(image_path):
    test_emb = get_embedding(image_path)

    best_pokemon = None
    best_score = -1.0

    for pokemon, emb in base_embeddings.items():
        score = np.dot(test_emb, emb)  # cosine similarity
        if score > best_score:
            best_score = score
            best_pokemon = pokemon

    return best_pokemon, best_score


# =============================
# Teste
# =============================
pokemon, score = identify_pokemon(TEST_IMAGE)

print("=================================")
print(f"Pokémon identificado: {pokemon}")
print(f"Score de similaridade: {score:.4f}")
print("=================================")
