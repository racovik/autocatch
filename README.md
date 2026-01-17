# termux setup
```
pkg install python-torch python-torchvision
```
```
git clone https://github.com/racovik/autocatch.git 
```
```
cd autocatch
```
```
pip install -r requirements.txt
```
## configs
```
nano config.py
```
```
token =
guild_id = 
```
## download model
```
mkdir -p models && curl -L -o models/pokerealmac_v1.0.pt https://github.com/racovik/autocatch/releases/download/v1.0.0/pokerealmac_v1.0.pt
```
## run
```
python discord_bot.py
```
