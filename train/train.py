import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Barra de progresso
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# =============================
# Configurações
# =============================


BASE_DIR = "dataset/base"
TEST_IMAGE = "test/teste.png"
IMAGE_SIZE = 224  # igual tambem no identify
torch.backends.nnpack.enabled = False
basedir = "models/"
name = input("Digite o nome do modelo: ")
if os.path.exists(f"{basedir}{name}.pt"):
    print(f"Modelo {name} já existe!")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    logging.debug("running on cuda")
    logging.debug(f"current device: {torch.cuda.current_device()}")
else:
    logging.info(
        "CUDA not available, falling back to CPU | ou seja, usando a CPU para treinar, tlgd."
    )

# =============================
# Transformações
# =============================
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Normalização padrão do ImageNet (Média e Desvio Padrão)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# =============================
# Modelo (extrator de embeddings)
# =============================
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)  # tem que ser igual no identify
model.classifier = nn.Identity()  # remove a cabeça de classificação
model.eval()
model.to(device)


# =============================
# Função para gerar embedding
# =============================
# Fundo branco pois o bot pode bugar se ficar processando com fundo branco
def process_with_white_background(image_path):
    img_rgba = Image.open(image_path).convert("RGBA")
    fundo_branco = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    img_final = Image.alpha_composite(fundo_branco, img_rgba)
    return img_final.convert("RGB")


def get_embedding(image_path):
    img = process_with_white_background(image_path).convert("L").convert("RGB")
    # img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze()  # removing cpu
    emb = emb / torch.norm(emb)  # normalização
    return emb


# =============================
# Criar banco de embeddings
# =============================
logging.info("Creating base embeddings...")
base_embeddings = {}

for pokemon in tqdm(os.listdir(BASE_DIR), desc="Processing Pokémon"):
    pokemon_dir = os.path.join(BASE_DIR, pokemon)
    if not os.path.isdir(pokemon_dir):
        continue

    images = [
        f
        for f in os.listdir(pokemon_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if len(images) == 0:
        continue

    all_embs = []
    for img_name in images:
        img_path = os.path.join(pokemon_dir, img_name)
        all_embs.append(get_embedding(img_path))

    # Faz a média de todos os vetores daquele pokemon
    avg_emb = torch.stack(all_embs).mean(dim=0)
    # Normaliza a média para manter o vetor unitário
    avg_emb = avg_emb / torch.norm(avg_emb)

    base_embeddings[pokemon] = avg_emb

# for pokemon in os.listdir(BASE_DIR):
#     pokemon_dir = os.path.join(BASE_DIR, pokemon)
#     if not os.path.isdir(pokemon_dir):
#         continue

#     images = os.listdir(pokemon_dir)
#     if len(images) == 0:
#         continue

#     img_path = os.path.join(pokemon_dir, images[0])
#     base_embeddings[pokemon] = get_embedding(img_path)


torch.save(
    base_embeddings,
    f"{basedir}{name}.pt",
)

logging.info(f"Base embeddings saved. on {basedir}{name}.pt")

print(f"[OK] {len(base_embeddings)} Pokémon carregados.")


# =============================
# Identificação por similaridade
# =============================
def identify_pokemon(image_path):
    test_emb = get_embedding(image_path)

    best_pokemon = None
    best_score = -1.0

    for pokemon, emb in base_embeddings.items():
        score = F.cosine_similarity(test_emb, emb, dim=0).item()  # cosine similarity
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
