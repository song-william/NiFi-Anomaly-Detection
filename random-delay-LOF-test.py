# -*- coding: utf-8 -*-
import numpy
import scipy
import random
import json
import glob
import sklearn
import copy
from lof import outliers
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
print "scipy version: " + scipy.__version__
print "numpy version: " + numpy.__version__
print "sklearn version: " + sklearn.__version__


def main():

    # actual provenance data of flow files
    flowFileData = []

    # list of features used in model
    modelFeatures = ["eventType", "componentId", "entitySize", "durationMillis"]
    dedupeFeatures = ['eventId', "eventType", "componentId", "entitySize", "durationMillis", 'componentType', 'updatedAttributes']
    # "eventType", "componentId", "entitySize", "durationMillis"
    # features that need to be type casted to int
    intFeatures = ["entitySize", "durationMillis"]

    fileDirectory = "/Users/wsong/Desktop/nifi/provenance-data/random-50000delay-mod-1000/*"
    saveFigureDirectory = "/Users/wsong/Desktop/Flow Provenance Graphs/Working with CSV/"
    flowName = "Random Time Delay"
    flowFileData = loadProvenanceData(fileDirectory, 500000)
    removeProvenanceReporterContamination(flowFileData)
    print "list size after contamination removed", len(flowFileData)
    cleanFeatures(flowFileData, dedupeFeatures)

    """# populate random times so not all identical points
    for event in flowFileData:
        event["durationMillis"] = random.uniform(0, 1)"""
        # obtain anomaly count and anomaly locations
    groundTruth = findGroundTruth(flowFileData)
    anomalyIndexList = []
    count = 0
    for num in list(enumerate(groundTruth)):
        if num[1][1] == 1:
            anomalyIndexList.append(num[0])
            count += 1
    print "number of anomalies", count
    print "number of events:", len(flowFileData)
    print "anomaly indicies", anomalyIndexList
    # populate anomalous times
    for index in anomalyIndexList:
        flowFileData[index]["durationMillis"] = random.uniform(50, 100)
        print flowFileData[index]["durationMillis"]

    # [dict(t) for t in set([tuple(sorted(d.items())) for d in flowFileData])]
    print "removing dupilcates"
    # the below solution cant even finish
    # [i for n, i in enumerate(flowFileData) if i not in flowFileData[n + 1:]]
    print "done removing duplicates"
    rawData = copy.deepcopy(flowFileData)

    cleanFeatures(flowFileData, modelFeatures)

    # cast integer features to int
    for dataPoint in flowFileData:
        for feature in intFeatures:
            dataPoint[feature] = float(dataPoint[feature])

    # loads features from a dictionary
    # link for reference:
    # http://scikit-learn.org/stable/modules/feature_extraction.html#dict-feature-extraction
    vec = DictVectorizer()
    data = vec.fit_transform(flowFileData).toarray()

    dataScaled = preprocessing.scale(data)
    # dataScaled = preprocessing.MinMaxScaler().fit_transform(data)


    print "Original data Dimensions:", dataScaled.shape
    instances = []
    for dataPoint in dataScaled:
        instances.append(tuple(dataPoint))

    print 'starting lof'
    lof = outliers(5, instances)
    for outlier in lof:
        value = outlier["lof"]
        index = outlier["index"]
        print value, index

    """# run PCA
    # sklearn_pca = sklearnPCA(n_components=.99)
    sklearn_pca = sklearnPCA(n_components=3)
    dataReduced = sklearn_pca.fit_transform(dataScaled)
    print "Variance Accounted for:", sklearn_pca.explained_variance_ratio_

    print "PCA Data Dimensions:", dataReduced.shape"""



    """
    # use_colours = {0: 'green', 1: 'red'}
    use_colours = {'LogAttribute': 'blue', 'GenerateFlowFile': 'green', 'ExecuteScript': 'red', 'Input Port': 'black', 'PutFile': 'purple'}
    use_sizes = {0: 10, 1: 50}
    use_markers = {0: 'o', 1: 'x'}
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(flowName)
    ax.set_xlabel('Column a')
    ax.set_ylabel('Column b')
    ax.set_zlabel('Column c')
    ax.view_init(elev=50, azim=60)              # elevation and angle
    ax.dist = 12
    ax.scatter(
           dataReduced[0:len(dataReduced), 0], dataReduced[0:len(dataReduced), 1], dataReduced[0:len(dataReduced), 2],  # data
           color=[use_colours[x["componentType"]] for x in rawData],     # marker colour
           marker='o',  # marker shape
           s=[use_sizes[x[1]] for x in groundTruth]          # marker size
           )
    classes = ['LogAttribute', 'GenerateFlowFile', 'ExecuteScript', 'Input Port', 'PutFile']
    class_colours = ['blue', 'green', 'red', 'black', 'purple']
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes, loc = 4, fontsize=10)
    # color=[use_colours[x[1]] for x in groundTruth]
    plt.show()"""
    """for i in xrange(0, 80, 20):
        for j in xrange(0, 100, 45):
            ax.view_init(elev=i, azim=j)
            plt.savefig(saveFigureDirectory + flowName + " elev"+str(i)+" angle"+str(j)+".png")"""
    
    print "script complete"




def divideDataSet(dataSet, trainingSetProportion, clusterinSetProportion):
    # partition dataset into training set, and test set
    totalDataCount = dataSet.shape[0]
    trainingDataLength = int(totalDataCount*trainingSetProportion)
    testDataLength = totalDataCount - trainingDataLength

    print "total data count:", totalDataCount
    print "training data count:", trainingDataLength
    print "test data count:",  testDataLength

    trainingData = dataSet[0:trainingDataLength, :]
    testData = dataSet[trainingDataLength: totalDataCount, :]

    # partition training set into clustering set and threshold calculation set
    clusteringDataLength = int(trainingDataLength*clusterinSetProportion)
    thresholdDataLength = trainingDataLength - clusteringDataLength

    print "clustering data count :", clusteringDataLength
    print "threshold calculation data count:", thresholdDataLength

    clusteringData = trainingData[0: clusteringDataLength, :]
    thresholdData = trainingData[clusteringDataLength: trainingDataLength, :]
    return [testData, clusteringData, thresholdData]


# removes all features not considered in model
def cleanFeatures(listOfEvents, features):
    for eventDict in listOfEvents:
        for key in eventDict.keys():
            if key not in features:
                del eventDict[key]


# load provenance data from local machine
def loadProvenanceData(directory, maxNumberOfDataEntries):
    listOfProvenanceFiles = []
    for file in glob.glob(directory):
        listOfProvenanceFiles.append(file)
    random.shuffle(listOfProvenanceFiles)
    provenanceData = []
    index = 0
    while len(provenanceData) < maxNumberOfDataEntries and index < len(listOfProvenanceFiles):
        with open(listOfProvenanceFiles[index], 'r') as myfile:
            jsonString = myfile.read()
        provenanceData = provenanceData + json.loads(jsonString)
        print len(provenanceData)
        index += 1
    return provenanceData


def findGroundTruth(provenanceData):
    groundTruth = []
    for event in provenanceData:
        if event['componentType'] == 'ExecuteScript' and event['updatedAttributes']['anomaly'] == 'y':
            groundTruth.append((event['eventId'], 1))
        else:
            groundTruth.append((event['eventId'], 0))
    return groundTruth


def removeProvenanceReporterContamination(flowFileData):
    flowFileData[:] = [event for event in flowFileData if event['componentName'] != 'ProvenanceData' and event['componentName'] != 'PutProvenance']



main()
