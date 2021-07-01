# Saarthi Task
 
<p align="center">

  [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

  [![GitHub contributors](https://img.shields.io/github/contributors/rstak/saarthi-task)](https://github.com/RsTaK/saarthi-task/graphs/contributors/)
  [![GitHub license](https://img.shields.io/github/license/rstak/saarthi-task)](https://github.com/RsTaK/saarthi-task/blob/master/LICENSE)
</p>  

<img src="assets\Saarthi-Logo.jpg"/>

# About

Given the audio data and transcription of the audio data, action that needs to be taken, action to be taken on which object and location where that object is present.

> Achieved F1 score of 1.0 on validation data (Text) for all required columns

> Achieved WER (Word Error Rate) of aproxx 5.8% on validation audio data for Speech to Text

> Added DVC for data and model versioning

Task is performed in PyTorch as well as in Tensorflow with custom models with the support of Tensorboard. HuggingFace is used to traing Speech to text engine. 

Script automatically identifies CPU/GPU support and works accordingly. Currently no support for Multi-GPU training due to hardware constraints.

# Approach

Treated the problem as Multi-class multi-label. Used a divide and conquer concept and build 3 seperate models to deal with each label handled by each model seperately. At the time of inferencing, output from all models combine to give "action", "location", "object" present in the text.

Used Wav2Vec2 model to convert Speech to text.

# How to use?

First clone the repository via:

>git clone https://github.com/RsTaK/saarthi-task.git

Create a new virtual environment with python 3.6 and then install the requirements:

> pip install -r requirements.txt 

Now, we need to pull the data and model from remote storage (Google Drive in our case) using dvc:
> dvc pull

We have the following options:

* Train a new model
* Evaulate csv file using pretrained model provided (Text or Audio)
* Run Inference file for single text/audio input

Feel free to checkout config to edit parameters as per the convinence.

label_keys directory contains the encoding used while training models and the csv for k-fold 

## Train a new model

We have support for PyTorch as well as Tensorflow. To train a new model, use:
> python train.py <specify the target_name> <choice of framework i.e., tf or pt>

Example:

> python train.py "action" "pt"

This will train a new model with action as a target in pytorch. Model will be saved in model/pt directory

> python train.py "location" "tf"

This will train a new model with location as a target in tensorflow. Model will be saved in model/tf directory

To train Wav2Vec2 model, checkout Speech.ipynb notebook in src/transformer_speech folder. Due to the hardware constraint, it was orinigally trained on Google Colab


## Evaulate csv file using pretrained model provided

To evaulate on PyTorch model, use:

> python evaulate_pt.py <path to the csv> <Enter the columns name to be treated as target>

Example:

> python evaulate_pt.py "data/valid_data.csv" "action"

Same goes for Tensorflow model, i.e.,

> python evaulate_tf.py "data/valid_data.csv" "action"

To evaulate Speech to Text model, use:

> python evaulate_sst.py <path to the csv> <Base dir where all audio files are located>

Feel free to checkout SST.csv that contains SST results on valid_data.csv

## Run Inference file for single text/audio input

> python cpu_inferencing.py <Enter the text you want to check on> <Pass audio else Enter None> <pt or tf for the choice of model from specific framework>

# Tensorboard

We have support of tensorboard too. Feel free to check training logs for PyTorch, Tensorflow or SST model using

>tensorboard -logdir=run/<Enter the name whose logs is to be checked>

Example:

> tensorboard -logdir=run/SST


# License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/RsTaK/saarthi-task/blob/master/LICENSE) file for details.

Datased used in the project sorely belongs to Saarthi
