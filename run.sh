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
sed -i "s/data.get('pending_payments', \[\])/data.get('pending_payments') or \[\]/g" /data/data/com.termux/files/home/autocatch/venv/lib/python3.12/site-packages/discord/state.py

echo "--- Iniciando o Bot ---"
python discord_bot.py
