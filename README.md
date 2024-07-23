# Digits-Recognition

A computer vision project about a simple MLP that recognizes handwritten digits. Using a raw UI it is possible to create a model, train and save it, load a pre-saved model and draw a digit asking for a prediction.

The project was made for educational purpose by a mix of University of Padua studies and personal researches.
It was made using Python 3.12.4, pip 24.0 and pytorch of CUDA 12.1 package for Windows from [here](https://pytorch.org) (NVIDIA GeForce RTX 3060 was used). CUDA cores are not mandatory but are strongly recommended. It consists of:
* 3 modules
  * ann for the MLP logic
  * ui for the UI composed by 4 modules
    * menu for the menu window
    * select for the select window
    * test for the test window
    * train for the training window
  * log for a better view of the logs on the terminal
* files for model saving
  * models.txt (models hyper parameters)
  * folder for models state (.pth.tar)
* TMNIST_Data.csv (dataset from [Kaggle](https://www.kaggle.com))
* demo MLP model

Fine tuning is allowed via code as shown in the main.py script: it allows fine tuning on a single paramter at a time and shows loss and accuracy graphs.
