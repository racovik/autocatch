import os

from PIL import Image

pokefolders = os.listdir("dataset/base")
for folder in pokefolders:
    img = Image.open(f"dataset/base/{folder}/pokeapi.png")
    img_invertida = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_invertida.save(f"dataset/base/{folder}/pokeapi_invertida.png")
    print(f"Processed {folder}")
