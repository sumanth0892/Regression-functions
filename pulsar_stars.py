import os
import pandas as pd
import numpy as np
from keras import layers,models
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Ignore the warnings 
def warn(*args,**kwargs):
    pass
import warnings
warnings.warn = warn
np.seterr(all = 'ignore')


fileName = 'pulsar_stars.csv'
totalData = pd.read_csv(fileName)
totalData = totalData.rename(
    columns = {' Mean of the integrated profile':'MuIP',
               ' Standard deviation of the integrated profile':'SigIP',
               ' Excess kurtosis of the integrated profile':'kurIP',
               ' Skewness of the integrated profile':'SkewnessIP',
               ' Mean of the DM-SNR curve':'MuDMSNR',
               ' Standard deviation of the DM-SNR curve':'SigDMSNR',
               ' Excess kurtosis of the DM-SNR curve':'kurDMSNR',
               ' Skewness of the DM-SNR curve':'SkewnessDMSNR',
               'target_class':'label'})
#From an exploratory data analysis, it is seen that the metric
#Skewness of the DM-SNR curve has the maximum variance
#Normalize that between 0 and 1 by dividing each value by the difference between min and max
diff = max(totalData['SkewnessDMSNR']) - min(totalData['SkewnessDMSNR'])
totalData['SkewnessDMSNR'] = totalData.apply(lambda x:x['SkewnessDMSNR']/diff,axis = 1)
labels = np.array(totalData['label'])
totalData = np.array(totalData.drop(['label'],axis = 1))
m,n = totalData.shape
labels = np.reshape(labels,(m,1))
#Split the data into training and test sets - 10000 for training and 7898 for testing
X = totalData[:10000]; Y = labels[:10000]
x = totalData[10000:]; y = labels[10000:]

#We can explore two models here
#The first is a Logistic regression classifier which works well for binary classification
#The second is a two-layer Dense Neural network classifier.
#Consider a classifier based on Logistic regression
def sigmoid(X):
    return 1.0/(1.0 + np.exp(-X))

def classify(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5: return 1.0
    else: return 0.0

def gradDescent(X,Y):
    #X is the data matrix and Y is the array of labels
    dataMat = np.mat(X); labelMat = np.mat(Y)
    m,n = dataMat.shape
    alpha = 0.002
    nIters = 500
    weights = np.ones((n,1))
    for k in range(nIters):
        h = sigmoid(dataMat*weights)
        error = labelMat - h
        weights = weights + alpha*dataMat.T*error
    return weights

W = gradDescent(X,Y)
M = x.shape[0]
regPreds = np.zeros((M,1))
for i in range(M):
    regPreds[i,:] = classify(x[i,:],W)
print(regPreds.shape)
print(sum(abs(regPreds - y)))

#Neural network model
#Start with a two layer model which should be good enough. 
model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(n,)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics = ['acc'])
model.fit(X,Y,epochs = 20,batch_size = 256,verbose = 1)
netPreds = model.predict(x)

for i in range(len(netPreds)):
    if netPreds[i]>0.5:
        netPreds[i] = 1.0
    else:
        netPreds[i] = 0.0

print(netPreds.shape)
print(sum(abs(netPreds - y)))
print(netPreds[1:10])






    
    


               
               
               
