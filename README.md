# ISY5002 Practice Module Project

## Authors: 
- Aryan Chakraborty ([@Aryan-Chakraborty](https://www.github.com/Aryan-Chakraborty))
- Chetana Buddhiraju ([@Chetana-Bud](https://www.github.com/Chetana-Bud))
- Harsh Tibrewal ([@htibrewal](https://www.github.com/htibrewal))
- Sanglap Dasgupta ([@AapseMatlb](https://www.github.com/AapseMatlb))
- Yashashwani Kashyap ([@SanglapD](https://www.github.com/SanglapD))

## User Guide

### To run the system in your machine:

#### Install Python if not installed already (Preferred version: >=3.9) 
 
> sudo apt-get install python3.9 
 
#### Install “virtualenv” package to create a virtual environment in which we will setup all the required packages for the project 
 
1. Installing via pip 
 
> pip install virtualenv 
 
2. Installing via apt 
 
> sudo apt install python3.9-venv 
 
#### Create a virtual environment with a name of your choice 
 
1. Via virtualenv 
 
> python3.9 -m virtualenv <env_name> 
 
2. Via venv 
 
> python3.x9-m venv <env_name> 
 
#### Activate the virtual environment 
 
> source <env_name>/bin/activate 
 
#### Once the virtual environment is setup and activated, all you need to do is install the required packages via this command 
 
> pip install –r requirements.txt 
 
#### When the installation is completed of all the required packages successfully, to run the project and launch the UI, run the command 
 
> python frontend.py

## Dataset Link

Adding the data to the folder or the repository is not possible
due to the large size of the dataset. Source of the data: [Data Gov SG Website](https://data.gov.sg/datasets?page=1&coverage=&query=parking&resultId=d_ca933a644e55d34fe21f28b8052fac63#tag/default/GET/transport/carpark-availability) 

This exposes an API that hosts the data for all the HDB car parks from 2018 to current time.

We have worked with the data captured from the API provided on this website for the year of 2023.