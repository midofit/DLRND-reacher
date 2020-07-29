# DLRND-navigation
Navigation project for Udacity's Deep Reinforcement Learning Nanodegree

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Python 3:  
This project only runs on Python 3. For Mac, you can download python 3 [here](https://www.python.org/downloads/mac-osx/) or install with Homebrew

```
$ brew install python3
```

Virtualenv (Optional):  
`virtualenv` is a tool to create isolated Python environments. Use this if you prefer not to install the project dependencies globally, which is highly recommended.  
virtualenv can be installed with `pip`, python package installer:  
```
$ pip install virtualenv
```

### Installation

A step by step series of examples that tell you how to get a development env running

#### 1. Environment setup
Clone the project repository

```
$ git clone git@github.com:midofit/DLRND-navigation.git
```

Navigate to the project root. If you have installed `virtualenv` from the previous step, create a new python virtual environment for this project:  

```
$ virtualenv navigation -p python3
```
and activate the virtual environment. Skip this step if you choose to install the python packages globally

```
$ source navigation/bin/activate
```
#### 2. Install dependencies  

With the virtual environment activated, you can install the required packages using `pip`:  
```
$ pip install -r requirements.txt
```
To install the dependencies without the virtual environment, use `pip3`:  
```
$ pip3 install -r requirements.txt
```

#### 3. Download Unity's Banana environment
Create `data` folder in the project root
```
$ mkdir data
```
You need only select the environment that matches your operating system:  
* Linux: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip):
* Windows (64-bit): [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)       
  
Download and extract the environment file, then place the file into `data` folder. Remember to update the `BANANA_FILE_PATH` variable in `settings.py` to match the file name

### Train the agent

To train the agent, run the following command:
```
$ python train_agent.py
```
To modify the hyper parameters for the Q network and the training process, update the corresponding variables in `settings.py`

### Results

The agent solved the game successfully after 450 episodes using fixed target Q network. For more details, please check out the notebook `Report.ipynb`. The notebook can be opened after running Jupyter lab in the project root directory.
```
$ jupyter lab
```