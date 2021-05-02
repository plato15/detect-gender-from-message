#Overview
For this technical challenge we are looking at building a machine learning model that can predict
the gender of a user just by the content of a chat message being sent. The provided data is
separated into two files, one for each class to be inferred: male or female.
All training data will be contained in the training_data folder which will have two files:
male.txt and female.txt which will contain in each line a chat message sent by a user of
the specified class.
#Requirements
Create both a training and a classification script which will be used to build a model and then
use the outputted model for inference of specific classes by passing strings as the script’s
arguments. These scripts should be written in either Java, Javascript, Python or C/C++.
The following files are required to be zipped and sent for review:
1. A README file that contains clear instructions on how to install locally all the
dependencies required to run both the training and classification scripts. Plus, any
important design decisions like libraries, algorithms, arguments, etc, must be explained.
2. A training script that should be executed by taking the location of the training data folder
and the name of the model file to be saved after training. Training arguments for the
model’s parameters can be passed as optional values if needed:
    >> ./train.sh training_data model_file
3. A classification script that should take in the model file to use and a string to infer. It’s
output should be through stdout and print the class that’s most likely to fit the provided
string (in this example, second line is the output of the classification):
    >> ./classify.sh model_file &quot;This is a test string”
male
4. An EVALUATION file that contains benchmarks and information on how the trained
model is performing. Use whatever method you think is most appropriate to evaluate the
model. Please provide information on why you chose a specific algorithm and
methodology.
Feel free to add any additional scripts or files to the attached zip file that would complement
your model and project.