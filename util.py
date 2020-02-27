import gensim
import os
import numpy as np

trainPercent = 0.3

def loaddata():
    model = gensim.models.Word2Vec.load('embbeding.model')
    userdirs = os.listdir('out/')
    trainU = []
    trainT = []
    trainL = []
    testU = []
    testT = []
    testL = []
    for dir in userdirs:
        trajectories = []
        uid = int(dir)
        for txt in os.listdir('out/' + dir):
            f = open('out/' + dir + '/' + txt).read().split()
            T = [model[node] for node in f]
            trajectories.append(T)
        trainPortion = int(len(trajectories) * trainPercent)
        #index = np.random.randint(len(trajectories), size=testPortion)
        for i in range(trainPortion):
            if len(trajectories[i])==0:
                continue
            trainU.append(uid)
            trainT.append(trajectories[i])
            trainL.append(len(trajectories[i]))
        for i in range(trainPortion,len(trajectories)):
            testU.append(uid)
            testT.append(trajectories[i])
            testL.append(len(trajectories[i]))
    return trainU,trainT,trainL,testU,testT,testL
