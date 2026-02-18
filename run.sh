#!/bin/bash

VERSION="1.1.0"

if [ -f "requirements.txt" ]; then
    OLD_HASH=$(md5sum requirements.txt)
else
    OLD_HASH=""
fi

git config pull.rebase true
echo "--- Atualizando... ---"
git pull --rebase --autostash

if [ ! -f "models/classifier.pt" ]; then
  echo "--- Baixando Modelo ---"
  curl -L -o models/classifier.pt https://github.com/racovik/autocatch/releases/download/${VERSION}/classifier.pt
fi

if [ -f "venv/bin/activate" ]; then

  source venv/bin/activate
  NEW_HASH=$(md5sum requirements.txt)
  if [ "$OLD_HASH" != "$NEW_HASH" ]; then
    echo "--- Atualizando Requirements ---"
    pip install -r requirements.txt
  else
    echo "--- Requirements is already update ---"
  fi

else
  echo "--- Criando Virtual Env ---"
  rm -rf venv
  python -m venv --system-site-packages venv
  source venv/bin/activate
  pip install -r requirements.txt
fi

# fix discord.py-self 2.0.0 bug
if grep -q "pending_payments', \[\]" venv/lib/python3.*/site-packages/discord/state.py 2>/dev/null; then
    echo "--- fixing discord.py-self... ---"
    sed -i "s/data.get('pending_payments', \[\])/data.get('pending_payments') or \[\]/g" \
        venv/lib/python3.*/site-packages/discord/state.py
fi
echo "--- Iniciando o Bot ---"
python discord_bot.py
