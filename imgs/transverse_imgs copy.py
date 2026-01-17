import os

from PIL import Image

pokefolders = os.listdir("dataset/base")
for folder in pokefolders:
    print(f"Removing from {folder}")
    os.remove(f"dataset/base/{folder}/pokeapi.png")
