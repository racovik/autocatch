import asyncio
import logging
import random
from io import BytesIO
import os
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import discord
import httpx
import torch
import torch.nn.functional as F
from discord.ext import commands
from PIL import Image
from torchvision import models, transforms


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


description = "Autocatch"


class Autocatcher(commands.Bot):
    def __init__(
        self,
        emb_model_path,
        target_guild_id,
        is_sending_rank,
        pokemon_bot_support,
        is_realm_disabled,
        catch_delay=3,
    ) -> None:
        super().__init__(command_prefix="?", description=description, self_bot=True)

        # configs do model
        self.EMBEDDINGS_PATH = emb_model_path
        self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.base_embeddings = load_model(
            self.model_device, self.EMBEDDINGS_PATH
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        assert os.path.exists(self.EMBEDDINGS_PATH), (
            f"{self.EMBEDDINGS_PATH} not found in files."
        )

        self.is_sending_rank = is_sending_rank
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.target_guild_id = target_guild_id
        self.http_client = httpx.Client()
        self.queue = asyncio.Queue()
        self.pokemon_bot_support = pokemon_bot_support
        self.is_realm_disabled = is_realm_disabled
        # delay depois de processar a imagem
        self.catch_delay = catch_delay

    async def setup_hook(self) -> None:
        self.loop.create_task(self.process_queue())

    async def process_queue(self):
        while True:
            item = await self.queue.get()
            message: discord.Message
            image_url: str
            message, image_url, command = item

            try:
                async with message.channel.typing():
                    emb = await self.loop.run_in_executor(
                        self.executor,
                        get_embedding_from_url,
                        image_url,
                        self.http_client,
                        self.model,
                        self.transform,
                        self.model_device,
                    )

                    nome, score, rank = await self.loop.run_in_executor(
                        self.executor, identify_pokemon, emb, self.base_embeddings
                    )

                    pokemon = format_pokemon_name(nome)
                    await asyncio.sleep(
                        self.catch_delay + random.uniform(0.2, 1)
                    )  # delay aleatório
                    await message.channel.send(f"{command} {pokemon}")

            except Exception as e:
                logging.error(f"Erro no worker: {e}")
            finally:
                self.queue.task_done()

    async def on_ready(self):
        logging.info(f"Logged in as {bot.user} (ID: {bot.user.id})")  # type: ignore
        print("------")

    async def on_message(self, message: discord.Message):
        if isinstance(message.channel, discord.TextChannel):
            if message.channel.guild.id == self.target_guild_id:  # type: ignore
                if message.embeds:
                    embed = message.embeds[0]
                    title_clean = (embed.title or "").lower()
                    if "wild" in title_clean and "pokémon" in title_clean:
                        image_url = embed.image.url
                        if embed.image is None:
                            return
                        if (
                            message.author.id == 665301904791699476
                            and not self.is_realm_disabled
                        ):
                            command = "<@665301904791699476> c"
                            await self.queue.put((message, image_url, command))
                        if (
                            message.author.id == 669228505128501258
                            and self.pokemon_bot_support
                        ):
                            command = "<@669228505128501258> c"
                            await self.queue.put((message, image_url, command))

    async def close(self):
        await super().close()
        self.http_client.close()
        self.executor.shutdown()


def load_model(
    device,
    emb_path,
):
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )  # MobileNEt Modelo feito para uso no Mobile
    model.classifier = torch.nn.Identity()  # type: ignore
    model.eval()  # model.eval() feito para parar o treinamento do modelo, pelo q eu entendi
    model.to(device)

    # ----- LOAD DOS EMBEDDINGS -----
    base_embeddings = torch.load(emb_path, map_location=device, weights_only=False)
    logging.info(f"Embeddings model loaded (Path: {emb_path}) (Device: {device}")
    return model, base_embeddings


# white backgroud pois no preto a IA poderia confundir as bordas dos pokemons com o fundo preto.
def process_with_white_background(image: bytes) -> Image.Image:
    img_rgba = Image.open(BytesIO(image)).convert("RGBA")
    fundo_branco = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    img_final = Image.alpha_composite(fundo_branco, img_rgba)
    return img_final.convert("RGB")


def get_embedding_from_url(
    url: str,
    http_client: httpx.Client,
    model: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    torch_device: str,
):
    logging.debug(f"Getting image embedding from {url}")
    response = http_client.get(url, timeout=10)
    response.raise_for_status()
    img = process_with_white_background(response.content)
    # img = Image.open(BytesIO(response.content)).convert("RGB")
    img = transform(img).unsqueeze(0).to(torch_device)

    with torch.no_grad():
        emb = model(img)

    emb = emb.squeeze()
    emb = emb / torch.norm(emb)  # normalização
    return emb


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


def identify_pokemon(embedding, base_embeddings):
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


def get_args():
    argument_parser = argparse.ArgumentParser(description="just a autocatch")
    argument_parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default="models/classifier.pt",
        dest="emb_model_path",
    )
    argument_parser.add_argument(
        "--ap", "--active-pokemon", action="store_true", dest="is_pokemon_bot_supported"
    )
    argument_parser.add_argument(
        "--dr--deactive-pokerealm", action="store_true", dest="is_realm_disabled"
    )
    argument_parser.add_argument(
        "-d",
        "--delay",
        type=int,
        default=3,
        dest="delay",
    )
    argument_parser.add_argument("-r", "--ranking", action="store_true")
    return argument_parser.parse_args()


if __name__ == "__main__":
    from config import token
    from config import guild_id

    args = get_args()
    bot = Autocatcher(
        emb_model_path=args.emb_model_path,
        is_sending_rank=args.ranking,
        target_guild_id=guild_id,
        pokemon_bot_support=args.is_pokemon_bot_supported,
        is_realm_disabled=args.is_realm_disabled,
        catch_delay=args.delay,
    )
    bot.run(token)
