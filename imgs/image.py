"""
CRIAR TODAS AS IMAGENS PARA O DATASET DE UM POKEMON APARTIR DE UMA BASE
QUE PODE SER A IMAGEM DA DEX DO BOT
"""

import logging
import argparse
import os 
from PIL import Image


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

DATASET_PATH = "dataset/base/"
BASE_IMAGE_NAME = "001.png"


def flags_parse():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument(
        "--d",
        "--dataset-path",
        dest="dataset_path",
        type=str,
        default=DATASET_PATH,
        help="change dataset path",
    )
    parser.add_argument(
        "--b",
        "--base-image",
        dest="base_image",
        type=str,
        default=BASE_IMAGE_NAME,
        help="base image name",
    )
    args = parser.parse_args()
    return args

def transverse_img(folder_path, img_name):
    path = os.path.join(folder_path, img_name)
    img = Image.open(path)
    img_invertida = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_invertida.save(os.path.join(folder_path, "pokeapi_invertida.png"))

def save_inverted_version(img: Image, path):
    inverted = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    inverted.save(path)
    logging.info(f"Saved inverted version at {path}")



def generate_smaller_images(folder_path, img_name):
    path = os.path.join(folder_path, img_name)
    largura, altura = 475, 475
    canvas = Image.new("RGBA", (475, 475), (0, 0, 0, 0))

    pokemon = Image.open(path)

    pokemon = pokemon.resize((250, 250), Image.Resampling.LANCZOS)
    canvas_sm = Image.new("RGBA", (475, 475), (0, 0, 0, 0))

    smaller = pokemon
    
    x = (largura - smaller.width) // 2
    y = (altura - smaller.height) // 2
    smaller = canvas_sm.paste(smaller, (x, y), smaller)
    canvas_sm.save(os.path.join(folder_path, "smaller.png"))
    logging.info("Saved smaller version")
    save_inverted_version(canvas_sm, os.path.join(folder_path, "smaller-inverted.png"))


    offset_x = -40  # pixels para a direita
    offset_y = -50  # pixels para baixo



    x = (largura - pokemon.width) // 2 + offset_x
    y = (altura - pokemon.height) // 2 + offset_y

    canvas.paste(pokemon, (x, y), pokemon)
    canvas.save(os.path.join(folder_path, "smaller-up.png"))
    logging.info("Saved smaller-up version")
    save_inverted_version(canvas, os.path.join(folder_path, "smaller-up-inverted.png"))
    

def check_if_exists_base_image(dataset_path) -> bool:
    return os.path.exists(f"{dataset_path}/001.png")


def main():
    flags = flags_parse()
    DATASET_PATH = flags.dataset_path
    BASE_IMAGE_NAME = flags.base_image
    logging.info(f"(DATASET_PATH: {DATASET_PATH})")
    logging.info(
        f"AVISO - Precisa conter uma imagem com nome {BASE_IMAGE_NAME} no dataset para servir como base."
    )
    transverse_img(DATASET_PATH, BASE_IMAGE_NAME)
    generate_smaller_images(DATASET_PATH, BASE_IMAGE_NAME)
    with open(os.path.join(DATASET_PATH, "config.txt"), "w") as f:
        f.write(f"base_image: {BASE_IMAGE_NAME}\n")
    
    

    


if __name__ == "__main__":
    main()
