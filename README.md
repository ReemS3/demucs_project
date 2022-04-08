# demucs_project

This project is repulication of the [Demucs model](https://github.com/facebookresearch/demucs). But it is written in TensorFlow using Colab Pro (P100 GP with 32GB of RAM), the authors have 8 V100 GPUs with 16GB of RAM. Here is the [link](https://colab.research.google.com/drive/1ZEEqwD3nYR21x2Y0_syG_7tWOhLfZ18W?usp=sharing#scrollTo=xQig2cISsGhM&uniqifier=4) to run the notebook in ColabPro. 

The dataset used is clipped version of the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html). First, we considered the first 30 seconds of each track, but this also exhausted the resource. Thus, we divided each 30-second clip to 3 audios of lenght 10 seconds. This is done in create_clipped_dataset.py. If you want to run this, please comment out the libraries needed for the IANNwTF Final Project.ipynb and the libraries of the API.

To run the API, please run the app.py.

If you want to run the notebook on your machine, you will need [pip-compile](https://github.com/jazzband/pip-tools) by running pip-compile requirements.in.