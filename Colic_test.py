#An example about horses and Colic.
#The data has 328 instances with 28 features.
#Use logistic regression to predict if the horse lives or dies.

def Classify(X,weights):
    prob = sigmoid(sum(X*weights))
    if prob>0.5: return 1.0
    else: return 0.0

def ReadColic():
    frTrainingSet = open('horseColicTraining.txt')
    frTestSet = open('horseColicTest.txt')
    trainingSet=[]; trainingLabel=[]
    for line in frTrainingSet.readlines():
        currentLine = line.strip().split('\t')
        lineArray=[]
        for i in range(21):
            lineArray.append(float(currentLine[i]))
        trainingSet.append(lineArray)
        trainingLabels.append(float(currentLine[21]))
    trainWeights = Stochastic(array(trainingSet),trainingLabel,500)
    errorC = 0; numTestVectors = 0.0
    for line in frTestSet.readlines():
        numTestVectors+=1.0
        currentLine = line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if (int(Classify(array(lineArr),trainweights))!=int(currentLine[21])):
            errorC+=1
        errorRate = (float(errorC)/numTestVectors)
        print "The error rate is %f" %errorRate
    return errorRate

def MultipleTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum+=ReadColic()
    print "After %d iterations the average error rate is %f" %(numTests, errorSum/float(numTests))
    
    
