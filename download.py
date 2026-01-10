import os
import time
from pathlib import Path

import httpx


def get_pokemon_name_by_number(n: int):
    response = httpx.get(
        f"https://pokeapi.co/api/v2/pokemon/{n}", follow_redirects=True
    )
    response.raise_for_status()
    if response.status_code == 200:
        data = response.json()
        name = data["name"]
        return name
    else:
        return None


def get_pokemon_image_bytes(n: int):
    url = f"https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/{n:03d}.png"
    response = httpx.get(url)
    response.raise_for_status()
    return response.content


def save_pokemon_image(pknm: str, image_bytes: bytes):
    folder = Path(f"dataset/{pknm}")
    folder.mkdir(parents=True, exist_ok=True)
    with open(f"{folder}/001.png", "wb") as f:
        f.write(image_bytes)


for i in range(1, 1026):
    try:
        name = get_pokemon_name_by_number(i)
        if os.path.exists(f"dataset/{name}"):
            print(f"Pokemon {i}: {name} already exists")
            continue
        image_bytes = get_pokemon_image_bytes(i)
        save_pokemon_image(name, image_bytes)
        print(f"Pokemon saved {i}: {name}")
        time.sleep(1)
    except:
        print("Error")
        continue
