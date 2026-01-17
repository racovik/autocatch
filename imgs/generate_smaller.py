import os

from PIL import Image

inverted = False

pokefolders = os.listdir("dataset/base")
for folder in pokefolders:
    largura, altura = 475, 475
    canvas = Image.new("RGBA", (475, 475), (0, 0, 0, 0))

    pokemon = Image.open(f"dataset/base/{folder}/001.png")
    if inverted:
        pokemon = pokemon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    pokemon = pokemon.resize((250, 250), Image.Resampling.LANCZOS)

    offset_x = -40  # pixels para a direita
    offset_y = -50  # pixels para baixo

    x = (largura - pokemon.width) // 2 + offset_x
    y = (altura - pokemon.height) // 2 + offset_y

    canvas.paste(pokemon, (x, y), pokemon)
    canvas.save(f"dataset/base/{folder}/smaller-up.png")
    print(f"Processed {folder}")
