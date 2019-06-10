# Robust Universal Neural Vocoding

A PyTorch implementation of [Robust Universal Neural Vocoding](https://arxiv.org/abs/1811.06292).

## Quick Start

1. Ensure you have Python 3 and PyTorch 1.

2. Clone the repo:
  ```
  git clone https://github.com/bshall/UniversalVocoding
  cd ./UniversalVocoding
  ```
3. Install requirements:
  ```
  pip install -r requirements.txt
  ```
4. Download and extract ZeroSpeech2019 TTS without the T English dataset:
  ```
  wget https://download.zerospeech.com/2019/english.tgz
  tar -xvzf english.tgz
  ```
5. Extract Mel spectrograms and preprocess audio:
  ```
  python preprocess.py
  ```

6. Train the model:
  ```
  python train.py
  ```
