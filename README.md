# PBLMD
The repository aims to identify the performance boundary of lane marking detection algorithms accelerately
1. [Dataset](#1-Dataset)
2. [Getting started](#2-getting-started)
### 1. Dataset
#### Dataset
The dataset we use during research are provided in [dataset](dataset). There are some challenging images we collected from nuScenes and TuSimple, and we mark the lane marking by using GroundTruthLabeler in MATLAB, then provide lane marking areas of each image in their corresponding text files.
### 2. Getting started
#### Adding wear
Use [wear_addition.py](wear_addition.py) to adding wear. Adjusting wear ratio at line 25 and wear type (regional distribution/ random distribution) at line 27. Press 's' to save the images after adding wear.
#### Adding noise & Adjusting brightness/contrast/saturation
Use [GUI.py](GUI.py) to adding noise and adjusting brightness/contrast/saturation. While adding noise, choose noise type (salt-pepper noise/ Gaussian noise) and ratio in GUI, press 's' to save the images. While adjusting brightness/contrast/saturation, use slider to change these parameters of images and press 's' to save the images.
