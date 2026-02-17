#!/bin/bash

if [ -f "requirements.txt" ]; then
    OLD_HASH=$(md5sum requirements.txt)
else
    OLD_HASH=""
fi

git config pull.rebase true
echo "--- Atualizando... ---"
git pull --rebase --autostash


if [ -f "venv/bin/activate" ]; then

  source venv/bin/activate
  pip install -r requirements.txt
else
  echo "--- Criando Virtual Env ---"
  rm -rf venv
  python -m venv --system-site-packages venv
  source venv/bin/activate
  NEW_HASH=$(md5sum requirements.txt)
  if [ "$OLD_HASH" != "$NEW_HASH" ]; then
    echo "--- Atualizando Requirements ---"
    pip install -r requirements.txt
  else
    echo "--- Requirements is already update ---"
  fi
fi

echo "--- Iniciando o Bot ---"
python discord_bot.py
