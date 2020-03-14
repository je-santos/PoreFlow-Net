<img align="right" width="40%" height="40%" src="https://github.com/je-santos/PoreFlow-Net/blob/master/ilustrations/flow.PNG"/>

# PoreFlow-Net
Implementation of
**PoreFlow-Net: a 3D convolutional neural network to predict fluid flow through porous media**
## Instructions
1. Download the desired data from the digital rock portal (or create your own via your preferred simulation method)
2. Use the train.py script to train a model

## Model architecture
This is how our network looks like:
![architecture](https://github.com/je-santos/PoreFlow-Net/blob/master/ilustrations/net.JPG)
## Methodology
![Process Overview](https://github.com/je-santos/PoreFlow-Net/blob/master/ilustrations/workflow.PNG)
## Pre-requisites
To train/test the model we used tensorflow 1.12, newer versions should work

The rest of the necessary packages should be available via pip

## Room for improvement
The keras tunner could be used to optimize the number of filters on each encoding branch

## Collaborations
We welcome collaborations


## Citation
If you use our code for your own research, we would be grateful if you cite our publication
[AWR](https://www.sciencedirect.com/science/article/pii/S0309170819311145)
```
@article{PFN2020,
title = "PoreFlow-Net: a 3D convolutional neural network to predict fluid flow through porous media",
journal = "Advances in Water Resources",
pages = "103539",
year = "2020",
issn = "0309-1708",
doi = "https://doi.org/10.1016/j.advwatres.2020.103539",
url = "http://www.sciencedirect.com/science/article/pii/S0309170819311145",
author = "Javier E. Santos and Duo Xu and Honggeun Jo and Christopher J. Landry and Maša Prodanović and Michael J. Pyrcz",
}
```
