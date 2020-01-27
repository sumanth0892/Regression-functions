import os
import numpy as np,pandas as pd
from keras import models,layers

trainSet = 'titanicTrain.csv'
testSet = 'titanicTest.csv'
testLabels = 'gender_submission.csv'

def loadDataSet(fileName):
    female_age = 28.7; male_age = 30.6
    X = pd.read_csv(fileName)
    X = X.drop(['Name','Cabin','Ticket'],axis = 1)
    X['Age'] = X.apply(
        lambda row:male_age if np.isnan(row['Age']) and row['Sex'] == 'male'
        else row['Age'],axis = 1)
    X['Age'] = X.apply(
        lambda row:female_age if np.isnan(row['Age']) and row['Sex'] == 'female'
        else row['Age'],axis = 1)
    X.Sex[X.Sex == 'female'] = 1
    X.Sex[X.Sex == 'male'] = 0
    X['Embarked'].fillna(0,inplace = True)
    X.Embarked[X.Embarked == 'Q'] = 1
    X.Embarked[X.Embarked == 'S'] = 2
    X.Embarked[X.Embarked == 'C'] = 3
    X = X.astype({'Sex':'int64','Embarked':'int64'})
    return X

X = loadDataSet(trainSet) #Comes as a PANDAS DATAFRAME
Y = np.array(X['Survived'])
X = X.drop(['Survived'],axis = 1)
X = np.array(X)
Y = np.reshape(Y,(len(Y),1))
x = loadDataSet(testSet)
y = pd.read_csv(testLabels)
y = np.array(y.drop(['PassengerId'],axis = 1))
print(X.shape)
print(Y.shape)
print(x.shape)
print(y.shape)

#Building the classifier
#We build the classifier using numpy because for records < 100000, numpy is
#faster than pandas

def sigmoid(X):
    return 1.0/(1 + np.exp(-X))

def classify(X,W):
    prob = sigmoid(sum(X*W))
    if prob>0.5: return 1.0
    else: return 0.0

def stochGradDescent(X,labels,epochs = 150):
    m,n = X.shape; weights = np.ones((n,1))
    alpha = 0.01; errors = []
    for i in range(m):
        x = np.reshape(X[i],(n,1))
        h = sigmoid(sum(x*weights))
        error = labels[i] - h
        errors.append(error)
        weights = weights + alpha*error*x
    return weights

trainWeights = stochGradDescent(X,Y,500)
print(trainWeights.shape)



"""
errorCount = 0; numTestVec = 0.0
predictions = []
for i in range(len(x)):
    predictions.append(classify(x[i,:],trainWeights))
for i in range(len(y)):
    errorCount += abs(predictions[i] - y[i])"""

#Building a deep learning model
model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(X.shape[1],)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
history = model.fit(X,Y,epochs = 100,batch_size = 32)


    



