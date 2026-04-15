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
import aiohttp
import torch
import torch.nn.functional as F
from discord.ext import commands
from PIL import Image
from torchvision import models, transforms

torch.set_num_threads(1)

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
        items = list(self.base_embeddings.items())
        self.names = [k for k, _ in items]
        self.names = list(self.base_embeddings.keys())
        self.base_tensor = torch.stack(list(self.base_embeddings.values())).to(
            self.model_device
        )  # shape: [N, D]
        self.base_tensor = F.normalize(self.base_tensor, dim=1)
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
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.target_guild_id = target_guild_id
        self.queue = asyncio.Queue()
        self.pokemon_bot_support = pokemon_bot_support
        self.is_realm_disabled = is_realm_disabled
        # delay depois de processar a imagem
        self.catch_delay = catch_delay

        self.session: aiohttp.ClientSession | None = None

    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession()
        assert self.session is not None
        self.loop.create_task(self.process_queue())

    async def process_queue(self):
        while True:
            item = await self.queue.get()
            message: discord.Message
            image_url: str
            message, image_url, command = item

            try:
                img_b = await self.get_img(image_url)
                async with message.channel.typing():
                    nome, _ = await self.loop.run_in_executor(
                        self.executor,
                        self.identify_pokemon,
                        img_b,
                    )
                    pokemon = format_pokemon_name(nome)
                    await asyncio.sleep(self.catch_delay)  # delay aleatório
                    await message.channel.send(f"{command} {pokemon}")

            except Exception as e:
                logging.error(f"Erro no worker: {e}")
            finally:
                self.queue.task_done()

    async def get_img(self, url) -> bytes:
        assert self.session is not None
        async with self.session.get(url) as response:
            response.raise_for_status()
            img_bytes = await response.read()
            return img_bytes

    # run in thread
    def identify_pokemon(self, img_bytes: bytes):
        embedding = self._get_embedding_from_bytes(img_bytes)
        embedding = embedding.unsqueeze(0)

        scores = torch.matmul(
            embedding, self.base_tensor.T
        )  # eq a: "emb @ tensor" [N, D] -> [D, N] | [1, D] @ [D, N] = [1, N]
        scores = scores.squeeze(0)  # [N]
        # pega só o melhor índice (mais rápido que topk)
        best_idx = int(torch.argmax(scores).item())

        melhor_pokemon = self.names[best_idx]
        melhor_score = scores[best_idx].item()

        return melhor_pokemon, melhor_score

    def _get_embedding_from_bytes(
        self,
        img_bytes: bytes,
    ):
        img = process_with_white_background(img_bytes)
        # img = Image.open(BytesIO(response.content)).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.model_device)  # type: ignore

        with torch.no_grad():
            emb = self.model(img)

        emb = emb.squeeze()
        emb = F.normalize(emb, dim=0)
        return emb

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
        if self.session is not None:
            await self.session.close()
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
