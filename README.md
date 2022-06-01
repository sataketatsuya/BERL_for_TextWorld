## Installation

This project requires Python 3.7+. It was tested on a Linux system with a CUDA GPU.

It is recommended to install the conda virtual environment.

You can install conda command on Linux [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

```
cd BERL_FOR_TEXTWORLD

conda create -n satake python=3.7 numpy scipy ipython matplotlib pandas scikit-learn seaborn
conda activate berl
conda install pytorch torchvision cudatoolkit -c pytorch

pip install -r requirements.txt
python -m nltk.downloader 'punkt'
```

Download textworld dataset. You can check the dataset detail at [First TextWorld Problems Competition website](https://competitions.codalab.org/competitions/21557#learn_the_details-data).
```
wget https://aka.ms/ftwp/dataset.zip
unzip games
```
The [Apex package](https://github.com/NVIDIA/apex) is also required to train the model in the GPU with 16-bit to allow larger batches in the limited GPU memory (this probably can be optional when running on a
GPU with 16Gb or more).

## Model training

To train the models run the following command with the folder having the games but you must download dataset.zip from the Textworld Competition wensite.

```
python src/playgame.py --display <games file>
```

This will output to the screen the game with the agent commands. You can also execute the command with a folder having several games.

## Ner model

For named entities this project uses the [BERT-NER model](https://github.com/kamalkraj/BERT-NER).
