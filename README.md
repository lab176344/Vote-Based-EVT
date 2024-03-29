## Vote-Based-EVT

The official code for the repo: Open-set Recognition for the Detection of Unknown Classes Based on theCombination of Convolutional Neural Networks and Random Forests 
## Dependencies
- Python (>=3.6)
- scikit-learn (0.21.0)
- scipy (1.5.2)
- numpy (0.19.0)
- Matlab 2020a

## Table of Contents
1. [MNIST](#mnist)
2. [CIFAR](#cifar)
3. [Traffic Scenarios](#ts)

## General descriptiopn

For all the datasets, the classes are divided into known and unknown classes. The known classes are chosen randomly and the process is repeated 5 times. The Macro F-Score of the known classes and the unknown class is calculated for the 5 different known class sets. The CNN trained in a supervised fashion for feature extraction and is used for feature extraction. Followed by which the Voter-Based EVT Model is trained and unknown classes are detected 


## MNIST<a name="mnist"></a>

Step 1: To train MNIST CNN and extract features run `sh Scripts/MNIST_CNN_Extraction.sh` and to train the vote based model run the script `\Mnist\RF+EVT\Vote_Based_EVT.m` is Matlab to train the Vote-Based EVT Model

Step 2: Evaulation can be done by using the script `\Mnist\RF+EVT\VoteBasedEVTStat.py`

## CIFAR<a name="cifar"></a>

Step 1: To train CIFAR CNN and extract features run `sh Scripts\CIFAR_CNN_Extraction.sh` and to train the vote based model run the script `\Cifar\RF+EVT\Vote_Based_EVT.m` is Matlab to train the Vote-Based EVT Model

Step 2: Evaulation can be done by using the script `\Cifar\RF+EVT\VoteBasedEVTStat.py`

## Traffic Scenarios<a name="ts"></a>

The traffic scenarios are generated from the HighD Dataset [1]. Please fill in the forms to request access to the HighD Data from https://www.highd-dataset.com/. 

Step 1: Generate scenario categories using the script `\Traffic_Scenarios\highD_generate_scenarios.m`, occupancy grids will be generated for the scenarios and saved for CNN+RF training

Step 2: Train the CNN and extract features for traffic scenarios using `python \Traffic_Scenarios\RF+EVT\ScenarioBasic.py`, followed by that to train the vote based model run the script `\Traffic_Scenarios\RF+EVT\Vote_Based_EVT.m`

Step 3:  Evaulation can be done by using the script `\Traffic_Scenarios\RF+EVT\VoteBasedEVTStat.py`

## Reference
[1] The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems, Krajewski et al., ITSC 2018
