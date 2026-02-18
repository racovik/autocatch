import asyncio
import logging
import random
from io import BytesIO
import os
import re
import json

import discord
import httpx
import torch
import torch.nn.functional as F
from discord.ext import commands
from PIL import Image
from torchvision import models, transforms

from config import guild_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


description = "Autocatch"


bot = commands.Bot(command_prefix="?", description=description, self_bot=True)
# ----- CONFIG -----
EMBEDDINGS_PATH = "models/classifier.pt" # model
assert os.path.exists(EMBEDDINGS_PATH), f"{EMBEDDINGS_PATH} not found. please download model"

DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # ou "cuda"/"rocm" se estiver usando GPU
logging.info(f"Torch device: {DEVICE}")
if DEVICE == "cuda":
    logging.info(f"Cuda Device: {torch.cuda.current_device()}")

IMAGE_SIZE = 224

# ISSO TEM Q SER IGUAL AO DO TREINAMENTO, ENT N MUDAR
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Normalização padrão do ImageNet (Média e Desvio Padrão)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# ----- MODELO BASE -----
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)  # MobileNEt Modelo feito para uso no Mobile
model.classifier = torch.nn.Identity()
model.eval()  # model.eval() feito para parar o treinamento do modelo, pelo q eu entendi
model.to(DEVICE)

# ----- LOAD DOS EMBEDDINGS -----
base_embeddings = torch.load(EMBEDDINGS_PATH, map_location=DEVICE, weights_only=False)
logging.info(f"Embeddings model loaded (Path: {EMBEDDINGS_PATH})")


# white backgroud pois no preto a IA poderia confundir as bordas dos pokemons com o fundo preto.
def process_with_white_background(image: bytes):
    img_rgba = Image.open(BytesIO(image)).convert("RGBA")
    fundo_branco = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    img_final = Image.alpha_composite(fundo_branco, img_rgba)
    return img_final.convert("RGB")


def get_embedding_from_url(url):
    logging.debug(f"Getting image embedding from {url}")
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    img = process_with_white_background(response.content)
    # img = Image.open(BytesIO(response.content)).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze()
    emb = emb / torch.norm(emb)  # normalização
    return emb


@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")


# identify pokemon padrãozinho
# def identify_pokemon(embedding):
#     melhor_pokemon = None
#     melhor_score = -1

#     for nome, emb in base_embeddings.items():
#         score = F.cosine_similarity(embedding, emb, dim=0).item()
#         if score > melhor_score:
#             melhor_score = score
#             melhor_pokemon = nome

#     return melhor_pokemon, melhor_score


def identify_pokemon(embedding):
    resultados = []

    # 1. Calcula o score para TODOS da base
    for nome, emb in base_embeddings.items():
        score = F.cosine_similarity(embedding, emb, dim=0).item()
        resultados.append({"nome": nome, "score": score})

    # 2. Ordena do maior score para o menor
    # (O melhor será o índice 0)
    resultados.sort(key=lambda x: x["score"], reverse=True)

    # 3. Pega o melhor para o retorno padrão
    melhor_pokemon = resultados[0]["nome"]
    melhor_score = resultados[0]["score"]

    # 4. Cria a string com o ranking dos outros (Top 5, por exemplo)
    ranking_str = "\n--- Ranking de Similaridade ---\n"
    for i, res in enumerate(resultados[:5], 1):  # Pega os 5 primeiros
        ranking_str += f"{i}º. {res['nome']}: {res['score']:.4f}\n"

    return melhor_pokemon, melhor_score, ranking_str


def save_message_log(data: dict):
    os.makedirs("logs", exist_ok=True)
    message_id = data["message_id"]
    with open(f"logs/{message_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def format_pokemon_name(pk) -> str:
    """remover informacoes da pasta. ex 'minior [core]' -> 'minior'"""
    return re.sub(r"(\[.*?\])", "", pk).strip()

ranking = False


@bot.event
async def on_message(message: discord.Message):
    if message.author.id == 665301904791699476:
        if message.channel.guild.id == guild_id:
            if message.embeds:
                embed = message.embeds[0]
                if embed.title and "A wild pokémon has аppeаred!" in embed.title:
                    async with message.channel.typing():
                        image_url = embed.image.url
                        melhor_pokemon = None
                        melhor_score = -1
                        sleep_time = random.uniform(3, 5)
                        await asyncio.sleep(sleep_time)
                        embed_image_emb = await asyncio.to_thread(
                            get_embedding_from_url, image_url
                        )
                        (
                            melhor_pokemon,
                            melhor_score,
                            ranking_str,
                        ) = await asyncio.to_thread(identify_pokemon, embed_image_emb)
                        pokemon = format_pokemon_name(melhor_pokemon)
                        await message.channel.send(
                            f"<@665301904791699476> c {pokemon}"
                        )

                        logging.info(f"Identified pokemon: {melhor_pokemon}")
                        data = {
                            "message_id": message.id,
                            "best_score": melhor_score,
                            "best_match": melhor_pokemon,
                            "ranking_str": ranking_str
                        }
                        save_message_log(data)
                        if ranking:
                            await asyncio.sleep(1)
                            await message.reply(ranking_str)


from config import token

bot.run(token)
