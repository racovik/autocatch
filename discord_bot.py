import argparse
import asyncio
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import aiohttp
import discord
import torch
import torch.nn.functional as F
from discord.ext import commands
from PIL import Image
from torch.types import FileLike
from torchvision import models, transforms

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


description = "Autocatch"


class PokemonModel:
    def __init__(self, checkpoint_model_path: FileLike):
        self.MODEL_PATH = checkpoint_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'(model: {self.MODEL_PATH}) running on "{self.device}"')
        self.model, self.class_mapping = self.load_model()
        self.transform = transforms.Compose(
            [
                # transforms.Lambda(lambda img: process_with_white_background(img)), # only if img: bytes
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self) -> tuple[models.MobileNetV2, list[str]]:
        checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(
            num_ftrs,  # type: ignore
            len(checkpoint["classes"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        class_mapping = checkpoint["classes"]
        model.eval()
        return model, class_mapping

    def process_with_white_background(self, image: Image.Image) -> Image.Image:
        img_rgba = image.convert("RGBA")
        white_back = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        img_final = Image.alpha_composite(white_back, img_rgba)
        return img_final.convert("RGB")

    def predict(self, img: Image.Image | bytes) -> str:
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img))
        # importante o white background
        img = self.process_with_white_background(img)

        batch = self.transform(img).unsqueeze(0).to(self.device)  # type: ignore
        with torch.no_grad():
            outputs = self.model(batch)
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            predicted_class_idx = int(preds.item())
            predicted_class = self.class_mapping[predicted_class_idx]
            # percentage = conf.item() * 100
        return predicted_class


class Autocatcher(commands.Bot):
    def __init__(
        self,
        model_path,
        target_guild_id,
        pokemon_bot_support,
        is_realm_disabled,
        catch_delay=3,
    ) -> None:
        super().__init__(command_prefix="?", description=description, self_bot=True)

        assert os.path.exists(model_path)
        self.model = PokemonModel(model_path)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.queue = asyncio.Queue()
        self.session: aiohttp.ClientSession | None = None

        # verificacoes das flags --ap --dr
        self.pokemon_bot_support = pokemon_bot_support
        self.is_realm_disabled = is_realm_disabled

        self.target_guild_id = target_guild_id
        # catch delay mas sempre adicionado 0.2~1 de delay a mais, ou seja, 0 não é 0
        self.catch_delay = catch_delay

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
                    nome = await self.loop.run_in_executor(
                        self.executor,
                        self.model.predict,
                        img_b,
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

    async def get_img(self, url) -> bytes:
        assert self.session is not None
        async with self.session.get(url) as response:
            response.raise_for_status()
            img_bytes = await response.read()
            return img_bytes

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


# white background como foi treinado


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
        dest="model_path",
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
    from config import guild_id, token

    args = get_args()
    bot = Autocatcher(
        model_path=args.model_path,
        target_guild_id=guild_id,
        pokemon_bot_support=args.is_pokemon_bot_supported,
        is_realm_disabled=args.is_realm_disabled,
        catch_delay=args.delay,
    )
    bot.run(token)
