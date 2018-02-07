# Explain Black Box

[Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://arxiv.org/abs/1704.03296) の pytorch 実装

## 準備(GPUで使う)

nvidia-docker がインストールされていることが前提です。また以下のコマンドは当リポジトリのルートにて実行してください。

```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
docker build -t pytorch ./
cd ../
nvidia-docker run -it -v $(pwd):/workspace --name explain-black-box pytorch bash
```

## Usage

```bash
$ python main.py -h
run with gpu
usage: main.py [-h] [--tv_beta TV_BETA] [--lr LR]
               [--max_iterations MAX_ITERATIONS]
               [--l1_coefficient L1_COEFFICIENT]
               [--tv_coefficient TV_COEFFICIENT]
               img_path

positional arguments:
  img_path              path to image

optional arguments:
  -h, --help            show this help message and exit
  --tv_beta TV_BETA
  --lr LR               learning rate
  --max_iterations MAX_ITERATIONS
                        max iteration
  --l1_coefficient L1_COEFFICIENT
                        l1 loss weight
  --tv_coefficient TV_COEFFICIENT
                        tv loss weight
```