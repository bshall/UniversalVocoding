# Robust Universal Neural Vocoding

A PyTorch implementation of [Robust Universal Neural Vocoding](https://arxiv.org/abs/1811.06292).
Audio samples can be found [here](https://bshall.github.io/UniversalVocoding/).


![network](network.png?raw=true "Robust Universal Neural Vocoding")

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
  
7. Generate:
  ```
  python generate.py --checkpoint=/path/to/checkpoint.pt --wav-path=/path/to/wav.wav
  ```

## Pretrained Models

Pretrained weights for the 9-bit model are available [here](https://github.com/bshall/UniversalVocoding/releases/tag/v0.1).

## Notable Differences from the Paper

1. Trained on 16kHz audio from 102 different speakers ([ZeroSpeech 2019: TTS without T](https://zerospeech.com/2019/) English dataset)
2. The model generates 9-bit mu-law audio (planning on training a 10-bit model soon)
3. Uses an embedding layer instead of one-hot encoding

### Acknowlegements

- https://github.com/fatchord/WaveRNN
