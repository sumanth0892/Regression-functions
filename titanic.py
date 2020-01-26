import os
import numpy as np,pandas as pd

trainSet = 'titanicTrain.csv'
testSet = 'titanicTest.csv'
testLabels = 'gender_submission.csv'

#Exploratory data analysis
#We explore the data through pandas and then use numpy for analysis
#Consider the average male and female ages on Titanic
#pandas is also used for preparing the data
female_age= 28.7; male_age = 30.6;
X = pd.read_csv(trainSet)
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
Y = np.array(X['Survived'])
X = X.drop(['Survived'],axis = 1)
X = np.array(X)
Y = np.reshape(Y,(len(Y),1))

#Prepare the test set
x = pd.read_csv(testSet)
y = pd.read_csv(testLabels)
x = x.drop(['Name','Cabin','Ticket'],axis = 1)
x['Age'] = x.apply(
	lambda row:male_age if np.isnan(row['Age']) and row['Sex'] == 'male'
	else row['Age'],axis = 1)
x['Age'] = x.apply(
	lambda row:female_age if np.isnan(row['Age']) and row['Sex'] == 'female'
	else row['Age'],axis = 1)
x.Sex[x.Sex == 'female'] = 1
x.Sex[x.Sex == 'male'] = 0
x.Embarked[x.Embarked == 'Q'] = 1
x.Embarked[x.Embarked == 'S'] = 2
x.Embarked[x.Embarked == 'C'] = 3
x = x.astype({'Sex':'int64','Embarked':'int64'})
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

def stochGradDescent(X,labels,nIter = 150):
    m,n = X.shape
    Weights = np.ones(n); dataIndex = range(m)
    for j in range(nIter):
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(X[randIndex] * Weights))
            error = labels[randIndex] - h
            Weights = Weights + alpha * error * X[randIndex]
            #del(dataIndex[randIndex])
    return Weights

trainWeights = stochGradDescent(X,Y,500)
errorCount = 0
predictions = []
for i in range(len(x)):
    predictions.append(classify(x[i,:],trainWeights))
for i in range(len(y)):
    errorCount += abs(predictions[i] - y[i])
    



