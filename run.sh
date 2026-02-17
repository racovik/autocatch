#!/bin/bash

git config pull.rebase true
echo "--- Atualizando... ---"
git pull


if [ -d "venv" ]; then

  source venv/bin/activate
  pip install -r requirements.txt
else
  echo "--- Criando Virtual Env ---"

  python -m venv --system-site-packages venv
  source venv/bin/activate
  pip install -r requirements.txt
fi

echo "--- Iniciando o Bot ---"
python discord_bot.py
