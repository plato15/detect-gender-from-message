
Author: Gabriel Olokunwolu
Date Created: 04/16/2021
Last Updated: 

# Directories And Uses
1. env: You dont have to worry about this directory, its for the python virtual environment and you will generate yours in later step
2. models: This is the directory to store the result of our training when we finish training.
3. scripts: This directory contains all the scripts used for the training and also classification (Including the Notebook file)
4. training_dataset: This directory contains our dataset (male.txt, female.txt) used for training
5. classify.sh: The bash script that is run by the user when they want to predict the gender of a text.
6. train.sh: The bash script used to train the data, fit the model and save
7. requirements.txt: This file contains all packages used in this project. (It will be installed in later step) 
8. evealuation.md: contains all relavant information about my evaluation metrics additional info

#Prerequistes (It will be nice to have python 3.8 installed on your Machine to run this process successfully)
* First Navigate into the directory of the downloaded zip on your terminal of choice
    >> cd Detect-Gender-FromText
    - Next is to create a new python environmen by running below command
    >> python -m venv env

* Once this is done, activate the new environment on a Mac or Linux computer, runnning:
    >> source env/bin/activate
    -If you are using a Windows computer, activate the environment as follows:
    >> source venv\Scripts\activate

* All needed packages for this project are listed in requirements.txt file and you should install all of them in your new environment by running the below
    >> pip install -r requirements.txt
# Training the Model
* Once you have done all the above, you should be set 
  
* To Train the model, Make sure you are in the project root diretory then  run the command below:
    >> ./train.sh train_dataset pred_model.sav
  
* Wait till model is done training and once that is done, you can predict by running below
    >> .sh ./classify.sh pred_model.sav "the text you will like to predict"



# 