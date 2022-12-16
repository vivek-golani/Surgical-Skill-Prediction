# Surgical-Skill-Prediction

Code for 'Surgical Skill Prediction using Unsupervised Surgical Instrument Segmentation via Anchor Generation'.

Papers for [Surgical skill prediction](https://arxiv.org/abs/2106.01035) and [Unsupervised Segmentation.](https://arxiv.org/abs/2008.11946)

Code for [Surgical skill prediction](https://github.com/Finspire13/Towards-Unified-Surgical-Skill-Assessment) and [Unsupervised Segmentation.](https://github.com/finspire13/agsd-surgical-instrument-segmentation)


## Setup

Recommended Environment for Unsupervised Segmentation repository: 

* Recommended Environment: Python 3.5, Cuda 10.0, PyTorch 1.3.1
* Install dependencies: `pip3 install -r requirements.txt`.

Recommended Environment for Surgical Skill Prediction repository: 

* Recommended Environment: Python 3.7, Cuda 10.1, PyTorch 1.6.0
* Install dependencies: pip3 install -r requirements.txt.


## Data
Data for our project can be found [here](https://drive.google.com/drive/folders/1-JY1BFskqhO-u-RJDk1cTsM1EgDzuwwF?usp=sharing)

 1. Complete [the access form of the JIGSAWS dataset](https://cs.jhu.edu/~los/jigsaws/info.php) and get the permission.
 2. Download our processed data for JIGSAWS from [Google Drive](https://drive.google.com/drive/folders/1-JY1BFskqhO-u-RJDk1cTsM1EgDzuwwF?usp=sharing)
 3. Put the data into the parent directory of the codes.
 4. Please maintain directory structure [like](https://drive.google.com/drive/folders/1-JY1BFskqhO-u-RJDk1cTsM1EgDzuwwF?usp=sharing) as we did in our experiments

The Data contains the following folders:

'raw JIGSAWS dataset': This is only shared here for the purpose of this project and not for external circulation. Please complete [the access form of the JIGSAWS    dataset](https://cs.jhu.edu/~los/jigsaws/info.php) and get the permission to get the dataset officially.
 
 'img_frames' : raw frames extracted from JIGSAWS videos for each task.
 
 'cues' : This folder contains handcrafted cues (color, optical flow, location based on color, location based on optical flow) for 5 random videos of each task - Knot-Tying, Needle Passing and Suturing.
 
 'anchors' : anchors generated using color and location based on color cues.
 
 'anchors_of' : anchors generated using color and location based on optical flow cues.
 
 
 ## Code
 
 


