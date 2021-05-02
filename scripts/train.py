
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from time import time
from sklearn.metrics import accuracy_score, plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import sys
import pickle
from models import TextSelector, cleaner, DenseTransformer, tokenizer

arguments= sys.argv

training_data=arguments[1]
model_filename=arguments[2]


def generate_data(gender):
    file = open('./' + training_data + '/'+gender+'.txt', 'r')
    file_line = file.readlines()
    data_list=[]
    for line in file_line:
        data_list.append(line)
    df=pd.DataFrame(data_list)
    df['gender']=gender
    return df

male_df=generate_data('male')
female_df=generate_data('female')

all_df=male_df.append(female_df)
#reset the index
all_df.reset_index(drop=True, inplace=True)

#remove the last 2 strings
def remove_last2char(txt):
    txt=txt[:-2]
    return txt
all_df['text'] = all_df[0].apply(lambda txt: remove_last2char(txt))
del all_df[0]

#covert gender field to numeric
all_df['gender'] = all_df['gender'].map( {'male': 1, 'female': 0})
all_df

# Custom transformer

from sklearn.base import BaseEstimator, TransformerMixin
#we will implememnt technique called nftdf. This will display word relevance
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=tokenizer)
#tfidf_vec=tfidf_model.fit_transform(text_df)

#before applying the functions, for transformation, we will split our data into Train and Test
target = all_df['gender']
all_df['gender'] = target
Y = all_df['gender']
X = all_df.drop('gender',axis=1)

# splitting training data into train and validation using sklearn
from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=.2, random_state=42)


# Create the  pipeline to clean, tokenize, vectorize
feature_pipe = Pipeline([
                 ('selector', TextSelector(key='text')),
                 ("cleaner", cleaner()),
                 ('tfidf', vectorizer),
                 ('to_dense', DenseTransformer()),
                ]
               )

# We will create a small function that will predict and return few metrics on accuracy and training duraration
def run_prediction_and_deploy_model(model):
    t0 = time()
    model.fit(X_train,y_train)
    print ("training time:", round(time()-t0, 3), "s")

    t0 = time()
    labels_pred=model.predict(X_test)
    print ("prediction time:", round(time()-t0, 3), "s")
    # Calculate accuracy percentage
    # actual=labels_test
    # predicted=labels_pred
    y_pred=labels_pred
    print('This Model gives an Accuracy of ', accuracy_score(y_test, y_pred)*100, '%')
    # save the model to disk

    pickle.dump(model, open('./models/'+ model_filename, 'wb'))
    print('Uploading Model to Model directory as ',model_filename, '...')

    # we will export the y_pred and y_test for later evaluation use
    from numpy import savetxt
    # define data
    savetxt('./evaluation_data/y_test.csv', y_test, delimiter=',')
    savetxt('./evaluation_data/y_pred.csv', y_pred, delimiter=',')

    #create ROC Curve and export
    plot_roc_curve(model, X_test, y_test)
    plt.savefig("./evaluation_data/roc_curve.png")

#our Algorthm of choice is
pred_pipe = Pipeline([
        ('feature_processing', feature_pipe),
        ('classifier', MultinomialNB())
    ]
    )

run_prediction_and_deploy_model(pred_pipe)




