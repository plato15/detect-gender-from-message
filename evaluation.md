My Model of choice is Multinomial Naive Bayes Algorithm for the reasons below

- Faster training time
- High Accuracy score compared to other Models I tried
- Naive Based models cannot represent complex behaviors like ensemble trees 
- Alway we are not worried about the problem of overfitting unlike Decision tree, 

Since this is a Binary Classification problem, our Evaluation focus on Binary Classification Evaluation

The data is equally distributed between both classes so Accuracy will be the metrics of interest

1. Confusion matrix
Confusion Matric gave me a good visual representation of model performace
The first step is to choose a decision threshold τ to label the instances as positives or negatives. If the probability assigned to the instance by the classifier is higher than τ, it is labeled as positive, and if lower, it is labeled as negative. The default value for the decision threshold is τ = 0.5.

After Classifiying all the instanced I compared the label out to the target out and generated 4 metrics listed below


True positives (TP): Number of messages sent by males and are classified as male [4180].
False positives (FP): Number of messages sent by males and are classified as female [1501].
False negatives (FN): Number of messages sent by females and are classified as male [1820].
True negatives (TN): Number of messages sent by females and are classified as female.[4486]


The confusion matrix then takes the following form:
------------------------------------------------------------------
|           | Predicted male         |	Predicted female         |
------------------------------------------------------------------
Real male   | true positives [4180]  |   false negatives  [1820] |
------------------------------------------------------------------
Real female | false positives [1501] |   true negatives  [4486] |
------------------------------------------------------------------

From the above, the rows represent the target classes while the columns represent the output classes.

The diagonal cells show the number of correctly classified cases, and the off-diagonal cells show the misclassified cases.

* Run the evaluation script to confirm
    >> python scripts/evaluation.py pred_model.sav  
  > 
* The confusion matrix is saved in evaluation_data [Named Confusion Matrix.PNG]


After labeling the outputs, the number of true positives is 6, the number of false positives is 3, the number of false negatives is 2, and the number of true negatives is 9.

This information is arranged in the confusion matrix as follows.


As we can see, my model classifies most of the cases correctly.


* However, I am not too impressed so I will be doing further evaluation

The next evaluation I did was to calculate the accuracy score.

    * Classification accuracy, which is the ratio of instances correctly classified. This generated a score 72%
The Error Rate:

    * Error rate, which is the ratio of instances misclassified. 28%

AUC/ROC Curve:

    * Another Metric of interest I checked was plotting the ROC(Receiver Operating Characteristics) curve saved in evaluation_data/roc_curve.png
      The ROC curve generated an AUC score of 80% which is a good score



