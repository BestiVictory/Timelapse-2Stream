# Timelapse-2Stream
Regarding how to train the dual-stream model for video aesthetics and the dual-stream model for time-lapse aesthetics, the code is derived from the paper 'Aesthetics-Driven Virtual Time-Lapse Photography Generation'.
## Environment
To execute the UE4-Carla related code (open UE4 and run Step1_offine-v2.py\Step2_showone.py), please install the Conda environment environment_carla.yml. 
To execute the code related to the 2-stream model training (train.py\test.py), please install the Conda environment environment_2stream.yml.
The installation process of the UE4-Carla engine is detailed in TLP System.md.
## Train
Place the data and data_old folders in the root directory, then execute train.py or train_old.py. The former trains with the Video Aesthetic Dataset, while the latter trains with the Time-Lapse Aesthetic Dataset.
## Test
Place the image sequence in a folder and change the variable paths in line 71 of test.py. This will score and compare all folders in the specified path. If you want to score the results shot by UE4-Carla, please set the path to the last folder containing the image sequence, such as carla_origin/PythonAPI/examples/output/result/scene1-point2/ame.
## Dataset and whole environment files
If you want to get the dataset and the entire environment files from Google Drive or Baidu Netdisk, please read the reproducibility paper.
## P.S.
If you want to fully reproduce the ablation experiments in the paper, please sequentially execute lines 423 to 426 in Step2_showone.py (using only one line at a time and commenting out the others). Then, copy the results to the AblationResult folder under the current directory, following the names in the comments: AblationResult/result3-all, AblationResult/result3--1, AblationResult/result3--2, AblationResult/result3--3, and execute test.py.
