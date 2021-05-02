
import sys, os
import pickle
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from models import TextSelector, cleaner, DenseTransformer, tokenizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics

arguments= sys.argv
print('Here are the arguments ', arguments)
model_file=arguments[1]

# load numpy array from csv file

# load array
def loadtxt_numpy(filename):
    np_obj=loadtxt('./evaluation_data/'+ filename + '.csv')
    return np_obj
y_test=loadtxt_numpy('y_test')
y_pred=loadtxt_numpy('y_pred')
X_test=loadtxt_numpy('x_test')

print(y_test)
print(y_pred)


loaded_model = pickle.load(open('./models/'+ model_file, 'rb'))
conf_mat = confusion_matrix(y_test, y_pred)


import seaborn as sns
sns.heatmap(conf_mat, annot=True)
print(conf_mat)




group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                conf_mat.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_mat.flatten()/np.sum(conf_mat)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues')
plt.savefig("./evaluation_data/confusion_matrix.png")
# plt.savefig('saving-a-seaborn-plot-as-png-file-transparent.png',transparent=True)



