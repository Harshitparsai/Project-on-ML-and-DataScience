#importing the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
data = pd.read_csv('creditcard.csv')

#exploring the dataset
print(data.columns)
print(data.shape)
print(data.describe())

#taking only 10% of the dataset for fast and safe running
data = data.sample(frac = 0.1 ,random_state = 1)
print(data.shape)

#plotting the histogram
data.hist(figure = (29,20))
plt.show()

#determining the number of froud comes in dataset
froud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(froud) / float(len(valid))

print(outlier_fraction)
print(len(froud))
print(len(valid))

#corelation matrix
cormat = data.corr()
sns.heatmap(cormat,vmax = 1,square=True)
plt.show()


#get all the columns from the dataset
columns = data.columns.tolist()

#filter all the columns except the class
columns = (c for c in columns if c not in ["Class"])
target = "Class"

X = data[columns]
Y = data[target]

#print the shape of the columns and the target columns
print(X.shape)
print(Y.shape)

#importing sklearn packages
from sklearn.metrics import classification_report , accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#define the random state
state = 1

#define the classifier 
classifiers = {
                "Isolation Forest" : IsolationForest(max_samples = len(X),
                                                     contamination = outlier_fraction,
                                                     random_state = state),
                "Local Outlier factor": LocalOutlierFactor(n_neighbors = 20,
                                                           contamination = outlier_fraction)
        
}

#fit the model
n_outlier = len(froud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    
    #fit the data with tad outlier
    if clf_name == 'Local Outlier factor':
        y_pred = clf.fit_predict(X)
        score_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        score_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    #reshape the prediction values
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    #number of error using this clf
    n_errors = (y_pred != Y).sum()
    
    #run the classification matrix
    print(clf_name,n_errors)
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
































