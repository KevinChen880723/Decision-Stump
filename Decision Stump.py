import numpy as np
import random

def targetFunciton(x):
    if x>=0:
        return 1.
    else:
        return -1.

def CreateData(N, tau):
    TrainingSet = []
    TrainingLabel = []
    TestingSet = []
    TestingLabel = []

    tau = tau*100   #Change probability to 100%

    # Generate x by a uniform distribution in [−1, +1].
    for i in range(N):
        TrainingSet.append(random.uniform(-1, 1))
        TestingSet.append(random.uniform(-1, 1))
    data = TrainingSet

    # Sort the training datas
    data.sort()

    # Generate y from x by y = f(x) and then flip y to −y with τ probability independently
    for d in data:
        realistic = random.randint(1, 100)
        if realistic > tau:
            TrainingLabel.append(targetFunciton(d))
        else:
            if targetFunciton(d) == 1.:
                TrainingLabel.append(-1.)
            else:
                TrainingLabel.append(1.)

    for d in TestingSet:
        realistic = random.randint(1, 100)
        if realistic > tau:
            TestingLabel.append(targetFunciton(d))
        else:
            if targetFunciton(d) == 1.:
                TestingLabel.append(-1.)
            else:
                TestingLabel.append(1.)
    
    return ( np.array(TrainingSet), np.array(TrainingLabel), np.array(TestingSet), np.array(TestingLabel) )

def test(data, label, theta, s):   
    # Check the correctness.
    errCnt = 0
    for i in range(len(data)):
        y = s*np.sign(data[i] - theta)
        if y != label[i]:
            errCnt = errCnt+1
    
    return errCnt/len(data)

def train(data, label):

    theta = 0
    s = 0

    # Generate theta and s sets
    thetaSet = [-1, -1]
    sSet = [-1, 1]
    for i in range(len(data)-1):
        if data[i] != data[i+1]:
            thetaSet.append((data[i]+data[i+1])/2)
            thetaSet.append((data[i]+data[i+1])/2)
            sSet.append(-1)
            sSet.append(1)

    # Change these two parameter sets's type to numpy array that we can calculate all Ein at once.
    thetaSet = np.array(thetaSet)
    sSet = np.array(sSet)
    
    cntErr = np.zeros(thetaSet.shape)
    for i in range(len(data)):
        dSet = data[i]*np.ones(thetaSet.shape)
        lSet = label[i]*np.ones(thetaSet.shape)
        y = sSet*np.sign(dSet-thetaSet)
        
        compare = (y!=lSet) # Ceck y and label is same or not. The result will be True/False
        cntErr = cntErr+compare   # Count error times of every parameter sets.

    # See which set of parameter can generate smallest Ein
    Ein = 1000000
    bestIndex = 0
    for i in range(len(cntErr)):
        if cntErr[i] < Ein:
            Ein = cntErr[i]
            bestIndex = i

    theta = thetaSet[bestIndex]
    s = sSet[bestIndex]
    Ein = Ein/len(data)

    return (theta, s, Ein)

def start():
    N = 200
    tau = 0.1
    repeatTime = 10000
    EoutMinusEin = []

    for i in range(repeatTime):
        
        # Get the data set
        (TrainingData, TrainingLabel, TestingData, TestingLabel) = CreateData(N, tau)
        
        # Find a (theta, s) with the smallest Ein
        (theta, s, Ein) = train(TrainingData, TrainingLabel)
        Eout = test(TestingData, TestingLabel, theta, s)
        #print("Ein = {0}, Eout = {1}".format(Ein, Eout))
        EoutMinusEin.append(Eout-Ein)

    print(np.mean(EoutMinusEin))


if __name__ == "__main__":
    start()