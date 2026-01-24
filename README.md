# AutoCatch
- Autocatch using Pokemon Classifier with Torch using embeddings
- Requires Python>=3.12
## termux setup
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
### configs
```
nano config.py
```
```
token =
guild_id = 
```
### download model
```
mkdir -p models && curl -L -o models/pokerealmac_v1.0.pt https://github.com/racovik/autocatch/releases/download/v1.0.0/classifier.pt
```
### run
```
python discord_bot.py
```

## windows setup

```
winget install Git.Git
```
### Install python on Microsoft Store
### Install torch 
<details>
<summary>if you have an Nvidia card</summary>

<pre>
| CUDA Version | Supported GPUs |
|-------------|------------------|
| **12.6** | Maxwell and newer (GTX 9xx/10xx/20xx/30xx/40xx/50xx) with limitações |
| **12.8** | Turing and newer with official support, including Blackwell |
| **13.0** | Turing and newer officially (GTX 16xx/20xx/30xx/40xx/50xx) |
</pre>

<h3>12.6</h3>
<pre>
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
</pre>

<pre>
<h3>12.8</h3>
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
</pre>
<h3>13.0</h3>
<pre>
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
</pre>

</details>

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
