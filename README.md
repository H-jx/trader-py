## Trader
The code is generated by chatGPT

## Dev

```shell
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# CUDA 11.1

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip3 freeze > requirements.txt

```

```shell
pip3 install pipenv -i https://mirrors.aliyun.com/pypi/simple

pipenv lock -r > requirements.txt
```