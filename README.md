# Towards Achieving Robust Universal Neural Vocoding

A PyTorch implementation of [Towards Achieving Robust Universal Neural Vocoding](https://arxiv.org/abs/1811.06292).
Audio samples can be found [here](https://bshall.github.io/UniversalVocoding/).

<div align="center">
    <img width="788" height="508" alt="Architecture of the vocoder." 
      src="https://github.com/bshall/UniversalVocoding/raw/master/univoc.png"><br>
    <sup><strong>Fig 1:</strong>Architecture of the vocoder.</sup>
</div>

## Quick Start

Ensure you have Python 3.8 and PyTorch 1.7 or greater installed. Then install the package with:
```
pip install univoc
```

## Example Usage

```python
import torch
import soundfile as sf
from univoc import Vocoder

# download pretrained weights (and optionally move to GPU)
vocoder = Vocoder.from_pretrained().cuda()

# load log-Mel spectrogram from file or tts
mel = ...

# generate waveform
with torch.no_grad():
    wav, sr = vocoder.generate(mel)

# save output
sf.write("path/to/save.wav", wav, sr)
```

## Train from Scratch

1. Clone the repo:
```
git clone https://github.com/bshall/UniversalVocoding
cd ./UniversalVocoding
```
2. Install requirements:
```
pip install -r requirements.txt
```
3. Download and extract the [LJ-Speech dataset](https://keithito.com/LJ-Speech-Dataset/):
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvjf LJSpeech-1.1.tar.bz2
```
4. Download the train split here and extract it in the root directory of the repo. 
5. Extract Mel spectrograms and preprocess audio:
```
python preprocess.py in_dir=path/to/LJSpeech-1.1 out_dir=datasets/LJSpeech-1.1
```
6. Train the model:
```
python train.py checkpoint_dir=ljspeech dataset_dir=datasets/LJSpeech-1.1
```

## Pretrained Models

Pretrained weights for the 10-bit LJ-Speech model are available [here]().

## Notable Differences from the Paper

1. Trained on 16kHz audio from a single speaker. For an older version trained on 102 different speakers form the [ZeroSpeech 2019: TTS without T](https://zerospeech.com/2019/) English dataset click [here](https://github.com/bshall/UniversalVocoding/releases/tag/v0.1).
2. Uses an embedding layer instead of one-hot encoding

### Acknowlegements

- https://github.com/fatchord/WaveRNN
