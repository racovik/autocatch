import asyncio
import random
from io import BytesIO

import discord
import httpx
import numpy as np
import torch
import torch.nn.functional as F
from discord.ext import commands
from PIL import Image
from torchvision import models, transforms

description = "Pokereal Autocatch"

bot = commands.Bot(command_prefix="?", description=description, self_bot=True)
# ----- CONFIG -----
EMBEDDINGS_PATH = "pokedex_embeddings-w-pokeapi.pt"
IMAGE_TESTE = "test/teste.png"  # imagem que você quer identificar
DEVICE = "cpu"  # ou "cuda"/"rocm" se estiver usando GPU
IMAGE_SIZE = 224
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


def get_embedding_from_url(url):
    response = httpx.get(url, timeout=10)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze().cpu()
    emb = emb / np.linalg.norm(emb)  # normalização
    return emb


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")


def identify_pokemon(embedding):
    melhor_pokemon = None
    melhor_score = -1

    for nome, emb in base_embeddings.items():
        score = F.cosine_similarity(embedding, emb, dim=0).item()
        if score > melhor_score:
            melhor_score = score
            melhor_pokemon = nome

    return melhor_pokemon, melhor_score


@bot.event
async def on_message(message: discord.Message):
    if message.author.id == 665301904791699476:
        if message.channel.id == 1422970015183011900:
            if message.embeds:
                embed = message.embeds[0]
                print(embed)
                if "A wild pokémon has аppeаred!" in embed.title:
                    async with message.channel.typing():
                        image_url = embed.image.url
                        print(image_url)
                        melhor_pokemon = None
                        melhor_score = -1
                        embed_image_emb = await asyncio.to_thread(
                            get_embedding_from_url, image_url
                        )
                        melhor_pokemon, melhor_score = await asyncio.to_thread(
                            identify_pokemon, embed_image_emb
                        )
                        await message.channel.send(
                            f"<@665301904791699476> c {melhor_pokemon}"
                        )
                        await message.channel.send(f"Similaridade: {melhor_score:.4f}")


from config import token

bot.run(token)
