import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Ignore the warnings 
def warn(*args,**kwargs):
    pass
import warnings
warnings.warn = warn
np.seterr(all = 'ignore')



file = 'pulsar_stars.csv'
dataSet = pd.read_csv(file)
dataSet.head()
dataSet.tail()
dataSet = dataSet.rename(
    columns = {' Mean of the integrated profile':'MuIP',
               ' Standard deviation of the integrated profile':'SigIP',
               ' Excess kurtosis of the integrated profile':'kurIP',
               ' Skewness of the integrated profile':'SkewnessIP',
               ' Mean of the DM-SNR curve':'MuDMSNR',
               ' Standard deviation of the DM-SNR curve':'SigDMSNR',
               ' Excess kurtosis of the DM-SNR curve':'kurDMSNR',
               ' Skewness of the DM-SNR curve':'SkewnessDMSNR',
               'target_class':'label'})
#scaler = MinMaxScaler(feature_range = (0,1))
#dataSet['SkewnessDMSNR'] = scaler.fit_transform(dataSet['SkewnessDMSNR'])
dataSet.info()
cols = [col for col in dataSet.columns if col != 'label']
features = dataSet[cols]
targets = dataSet['label']
features.info()

"""
#Visualize the data
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = 'whitegrid',color_codes = True)
sns.set(rc = {'figure.figsize':(11.7,8.27)})
sns.countplot('MuDMSNR',data = dataSet,hue = 'label')
sns.despine(offset = 10,trim = True)
plt.show()
"""

#Split the data
X,x,Y,y = train_test_split(features,targets,test_size = 0.2,random_state = 10)

#Let us try different algorithms to find the best match
#1. Naive-Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#Create a GaussianNB object
gnb = GaussianNB()
pred = gnb.fit(X,Y).predict(x)
print("Naive-Bayes accuracy: ",accuracy_score(y,pred,normalize = True))

#2. Linear Support Vector Classifier
from sklearn.svm import LinearSVC
svc_model = LinearSVC(random_state = 0)
pred = svc_model.fit(X,Y).predict(x)
print("Linear SVC accuracy: ",accuracy_score(y,pred,normalize = True))

#3. k-Nearest-Neighbours classifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(X,Y)
pred = neigh.predict(x)
print("k-Nearest-Neighbours score: ",accuracy_score(y,pred))

#4. Decision trees
from sklearn import tree
#Create a tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)
preds = clf.predict(x)
print("Decision tree score: ",accuracy_score(y,preds))

#5. Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
forClf = RandomForestClassifier(n_estimators = 10)
forClf.fit(X,Y)
preds = forClf.predict(x)
print("Random Forest classifier score: ",accuracy_score(y,preds))


#Comparing the performance
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(gnb,classes = [0,1])
visualizer.fit(X,Y)
visualizer.score(x,y)
g = visualizer.poof()


visualizer = ClassificationReport(svc_model,classes = [0,1])
visualizer.fit(X,Y)
visualizer.score(x,y)
g = visualizer.poof()


visualizer = ClassificationReport(neigh,classes = [0,1])
visualizer.fit(X,Y)
visualizer.score(x,y)
g = visualizer.poof()


visualizer = ClassificationReport(clf,classes = [0,1])
visualizer.fit(X,Y)
visualizer.score(x,y)
g = visualizer.poof()


visualizer = ClassificationReport(forClf,classes = [0,1])
visualizer.fit(X,Y)
visualizer.score(x,y)
g = visualizer.poof()




