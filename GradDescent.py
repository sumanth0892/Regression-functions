#After the data is collected and sorted into vectors using Pandas or reading
#File and sorting the data
#Gradient Ascent works well for small dataSets.

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
def sigmoid(x):
    a = 1/(1+exp(-x))
    return a

def gradientAscent(dataMatrix,Labels):
    dataMatrix = np.mat(dataMatrix)
    LabelsMat = np.mat(Labels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    trials = 500
    weights = np.ones((n,1))
    for k in range(trials):
        h = sigmoid(dataMatrix*weights)
        error = Labelsmat - h
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights

#Plotting the best fit line
def BestFit(we):
    weights = we.getA()
    dataMatrix,Labels = loadDataSet() #A generic function/class to give us the data set and labels
    dArray = np.array(dataMatrix)
    n = shape(dArray)[0]
    xcoord1=[]; ycoord1=[]
    xcoord2=[]; ycoord2=[]
    for i in range(n):
        if int(Labels[i])==1:
            xcoord1.append(dArray[i,1]); ycoord1.append(dArray[i,2])
        else:
            xcoord2.append(dArray[i,1]); ycoord2.append(dArray[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoord1,ycoord1,s=30,c='red',marker='s')
    ax.scatter(xcoord2,ycoord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x/weights[2])
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
                           
    
