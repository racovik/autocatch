import os

import httpx

client = httpx.Client(timeout=10)

for n in range(1, 1026):
    response = client.get(
        f"https://pokeapi.co/api/v2/pokemon/{n}", follow_redirects=True
    )
    response.raise_for_status()
    if response.status_code == 200:
        data = response.json()
        name = data["name"]
        if os.path.exists(f"dataaset/base/{name}/pokeapi.png"):
            print("this pokemon is already in database.")
            continue
        artwork_front_default = data["sprites"]["other"]["official-artwork"][
            "front_default"
        ]
        print("Downloading artwork for", name)
        response = client.get(artwork_front_default)
        response.raise_for_status()
        if response.status_code == 200:
            with open(f"dataset/base/{name}/pokeapi.png", "wb") as f:
                f.write(response.content)
            print("Downloaded artwork for", name)
