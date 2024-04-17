<div align="center">



# AkylAI TTS


[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<img src="https://github.com/simonlobgromov/Matcha-TTS/blob/main/photo_2024-04-07_15-59-52.png" height="400"/>
</div>

# AkylAI-TTS for Kyrgyz language

We present to you a model trained in the Kyrgyz language, which has been trained on 13 hours of speech and 7,000 samples, complete with source code and training scripts. The architecture is based on Matcha-TTS.
It`s a new approach to non-autoregressive neural TTS, that uses [conditional flow matching](https://arxiv.org/abs/2210.02747) (similar to [rectified flows](https://arxiv.org/abs/2209.03003)) to speed up ODE-based speech synthesis. Our method:

- Is probabilistic
- Has compact memory footprint
- Sounds highly natural
- Is very fast to synthesise from

You can try our *AkylAI TTS* by visiting [SPACE](https://huggingface.co/spaces/the-cramer-project/akylai-tts-mini) and read [ICASSP 2024 paper](https://arxiv.org/abs/2309.03199) for more details.

# Inference

## Run via terminal


It is recommended to start by setting up a virtual environment using `venv`.

1. Clone this repository and install all modules and dependencies by running the commands:

```
git clone https://github.com/Akyl-AI/tts-mini
cd Matcha-TTS
pip install -e .
apt-get install espeak-ng
```


2. Run with CLI arguments:

- To synthesise from given text, run:

```bash
matcha-tts --text "<INPUT TEXT>"
```

- To synthesise from a file, run:

```bash
matcha-tts --file <PATH TO FILE>
```
- Speaking rate

```bash
matcha-tts --text "<INPUT TEXT>" --speaking_rate 1.0
```

- Sampling temperature

```bash
matcha-tts --text "<INPUT TEXT>" --temperature 0.667
```

- Euler ODE solver steps

```bash
matcha-tts --text "<INPUT TEXT>" --steps 10
```


# Train with your own dataset.

## Dataset

For training this model, it is suitable to organize data similar to [LJ Speech](https://keithito.com/LJ-Speech-Dataset/). Each audio file should be single-channel 16-bit PCM WAV with a sample rate of 22050 Hz. WAV files must have unique names, for example:

```
file_1.wav
file_2.wav
file_3.wav
file_4.wav
....
file_12454.wav
file_12455.wav
```


They should also be placed at the root of the project directory in a separate folder.

Additionally, the project should include two `.txt` files for Train and Test with metadata for the files. The names of these files can be arbitrary, and their structure is as follows:
```
.../Matcha-TTS/<your folder name>/wavs/<filename>.wav|Баарыңарга салам, менин атым Акылай.
.../Matcha-TTS/<your folder name>/wavs/<filename>.wav|Мен бардыгын бул жерде Инновация борборунда көргөнүмө абдан кубанычтамын.
.../Matcha-TTS/<your folder name>/wavs/<filename>.wav|<your sentence>
.../Matcha-TTS/<your folder name>/wavs/<filename>.wav|<your sentence>
.../Matcha-TTS/<your folder name>/wavs/<filename>.wav|<your sentence>
........
```
Where each line is the FULL path to the file located in the folder with the uploaded audio, and a sentence in its original form with punctuation is written after the delimiter '|'. 
It is advisable to clean the text of unnecessary and unwanted characters beforehand. Be careful with abbreviations and contractions.
The text preprocessing does not include functionality for processing abbreviations and contractions; however, the built-in phonemizer can transcribe numbers, but to avoid errors, it is better to write numbers in words.

## Dataset from Hugging Face

If you want to use a dataset that you store on Hugging Face, it would be convenient to use the `create-dataset` script, which will handle the downloading and all the data preparation, including .txt files with metadata.
Here's what its structure might look like:

```
DatasetDict({
    train: Dataset({
        features: ['id', 'raw_transcription', 'transcription', 'sentence_type', 'speaker_id', 'gender', 'audio'],
        num_rows: 7016
    })
    test: Dataset({
        features: ['id', 'raw_transcription', 'transcription', 'sentence_type', 'speaker_id', 'gender', 'audio'],
        num_rows: 31
    })
})
```

Where the most important and mandatory features are:
```
['raw_transcription', 'audio']
```

Where:

`raw_transcription` - this is the text of your sentences in the original version (the requirements are the same as in the previous method).

`audio` - these are audio files with metadata, which are dictionaries with keys:

* `array` - audio in the form of a `numpy.ndarray` with a `float32` data type
* `path` - file name
* `sampling_rate` - Sampling rate, which should be no less than 22050 Hz.
  
Example a row:

```
{'array': array([-3.05175781e-05, -3.05175781e-05,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
 'path': '1353.wav',
 'sampling_rate': 44100}
```




## Process by Terminal

* **Load this repo and connect to HF**

```
git clone https://github.com/Akyl-AI/tts-mini
cd Matcha-TTS
pip install -e .
```

Install this:

```
apt-get install espeak-ng
```
Connect to HF (Skip this step if you are not using data from Hugging Face.)

```
git config --global credential.helper store
huggingface-cli login
```

* **Load the Data** (Skip this step if you are not using data from Hugging Face.)

The script will automatically create a folder with audio recordings and text files with metadata. During the process, enter the HF repository name and the dataset name.
  

```
create-dataset

# If you see a cat, then everything is fine!
```

* Go to `configs/data/akylai<OR YOUR FILE NAME>.yaml` and change

```yaml
train_filelist_path: data/filelists/akylai_audio_text_train_filelist.txt # path to your TXT with metadata
valid_filelist_path: data/filelists/akylai_audio_text_val_filelist.txt # path to your TXT with metadata
```

* Generate normalisation statistics with the yaml file of dataset configuration

```bash
matcha-data-stats -i akylai.yaml
# Output:
#{'mel_mean': -5.53662231756592, 'mel_std': 2.1161014277038574}
```

* Update these values in `configs/data/akylai.yaml` under `data_statistics` key.

```bash
data_statistics:  # Computed for akylai(or your) dataset
  mel_mean: -5.536622
  mel_std: 2.116101
```



* **Train**

```
python matcha/train.py experiment=akylai
```

OR

```
python matcha/train.py experiment=akylai trainer.devices=[0,1]
```


* **Checkpoints**

Checkpoints will be saved in `./Matcha-TTS/logs/train/<MODEL_NAME>/runs/<DATE>_<TIME>/checkpoints`. Unload them or select the last few checkpoints.



# Credits


- Shivam Mehta ([GitHub](https://github.com/shivammehta25))
- The Cramer Project (Data collection and preprocessing) [Official Space](https://thecramer.com/)
- Amantur Amatov (Expert)
- Timur Turatali (Expert, Research)
- Den Pavlov (Research, Data preprocessing and ML engineering) [GitHub](https://github.com/simonlobgromov/Matcha-TTS)
- Ulan Abdurazakov (Environment Developer)
- Nursultan Bakashov (CEO)

## Citation information

If you use our code or otherwise find this work useful, please cite our paper:

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

## Acknowledgements

Since this code uses [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), you have all the powers that come with it.

Other source code we would like to acknowledge:

- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev): For helping me figure out how to make cython binaries pip installable and encouragement
- [Hugging Face Diffusers](https://huggingface.co/): For their awesome diffusers library and its components
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): For the monotonic alignment search source code
- [torchdyn](https://github.com/DiffEqML/torchdyn): Useful for trying other ODE solvers during research and development
- [labml.ai](https://nn.labml.ai/transformers/rope/index.html): For the RoPE implementation

