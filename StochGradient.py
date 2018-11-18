#Stochastic gradienet works for bigger values vs Gradient descent
#Plotting the values works as the same in Gradient descent
#After parsing the data values which are missing

def StochasticGradient(dataMat, Labels):
    m,n = shape(dataMat)
    weights = ones(n)
    alpha = 0.01
    for k in range(m):
        h = sigmoid(sum(dataMatrix[i]*w))
        error = Labels[k] - h
        weights = weights+alpha*error*dataMatrix[i]
    return weights

        
        
