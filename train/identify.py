import logging
import os


import argparse

# import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


# ----- CONFIG -----
EMBEDDINGS_PATH = "models/classifie.pt"
IMAGE_TESTE = "test/blue_minior.png"  # imagem que você quer identificar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # cuda ou CPU.
IMAGE_SIZE = 224

if DEVICE == "cuda":
    logging.info("Using GPU for inference.")
else:
    logging.info("Using CPU for inference.")
# ----- TRANSFORM (TEM QUE SER IGUAL AO TREINO) -----
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Normalização padrão do ImageNet (Média e Desvio Padrão)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# ----- MODELO BASE -----
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.eval()
model.to(DEVICE)


def load_embedings(path):
    # ----- LOAD DOS EMBEDDINGS -----
    logging.info("loading embeddings...")
    base_embeddings = torch.load(path, map_location=DEVICE, weights_only=False)
    return base_embeddings


# ----- FUNÇÃO DE EMBEDDING -----
# def get_embedding(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = transform(img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         emb = model(img)
#         emb = F.normalize(emb, dim=1)
#     return emb.squeeze(0)


# Fundo branco pois o bot pode bugar se ficar processando com fundo branco
def process_with_white_background(image_path):
    img_rgba = Image.open(image_path).convert("RGBA")
    fundo_branco = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    img_final = Image.alpha_composite(fundo_branco, img_rgba)
    return img_final.convert("RGB")


def get_embedding(image_path):
    img = process_with_white_background(image_path).convert("L").convert("RGB")
    # img = Image.open(image_path).convert("RGB")
    img = (
        transform(img).unsqueeze(0).to(DEVICE)
    )  # unsqueeze TO Device, ou seja, vai para GPU ou CPU.

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze()  # removing .cpu()
    emb = F.normalize(emb, dim=0)
    # emb = emb / torch.norm(emb)  # normalização np.linalg.norm --> torch.norm para desempenho em GPU
    return emb


def flags_parse():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument(
        "--m",
        "--model-path",
        dest="model_path",
        type=str,
        default=EMBEDDINGS_PATH,
        help="change model path",
    )
    parser.add_argument(
        "--i",
        "--image-test",
        dest="test_image",
        type=str,
        default=IMAGE_TESTE,
        help="test image name",
    )
    args = parser.parse_args()
    return args


def main():
    args = flags_parse()
    # ----- EMBEDDING DA IMAGEM TESTE -----
    test_emb = get_embedding(args.test_image)
    base_embeddings = load_embedings(args.model_path)
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


if __name__ == "__main__":
    main()
