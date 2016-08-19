
# -*- coding: utf-8 -*-
import numpy
import scipy
import csv
import random
import json
import glob
import sklearn
import kmodes
import copy
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
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
    listFeatures = ["entitySize", "durationMillis", "componentId", "eventType", "lineageDuration"]
    # "eventType", "componentId", "componentType", "entitySize", "durationMillis"
    # features that need to be type casted to float
    intFeatures = ["entitySize", "durationMillis", "timestampMillis", "lineageStart"]
    # fileDirectory = "/Users/wsong/Desktop/nifi/provenance-data/working-with-csv-sample2/*"
    fileDirectory = "/Users/wsong/Desktop/nifi/provenance-data/random-delay-mod-100/*"
    saveFigureDirectory = "/Users/wsong/Desktop/Flow Provenance Graphs/Working with CSV/"
    flowName = "Working with CSV"
    flowFileData = loadProvenanceData(fileDirectory, 500000)
    removeProvenanceReporterContamination(flowFileData)
    rawData = copy.deepcopy(flowFileData)
    groundTruth = findGroundTruth(flowFileData)
    count = 0
    for num in groundTruth:
        count += num[1]
    print "number of anomalies", count
    print "number of events:", len(flowFileData)

    # cast string features to float and add lineage duration feature
    for dataPoint in flowFileData:
        for feature in intFeatures:
            dataPoint[feature] = float(dataPoint[feature])
            dataPoint["lineageDuration"] = dataPoint["timestampMillis"] - dataPoint["lineageStart"]

    cleanFeatures(flowFileData, listFeatures)
    



    # loads features from a dictionary
    # link for reference:
    # http://scikit-learn.org/stable/modules/feature_extraction.html#dict-feature-extraction
    vec = DictVectorizer()
    data = vec.fit_transform(flowFileData).toarray()


    dataScaled = preprocessing.scale(data)
    # dataScaled = preprocessing.MinMaxScaler().fit_transform(data)

    # run PCA
    print "Original data Dimensions:", dataScaled.shape
    # sklearn_pca = sklearnPCA(n_components=.99)
    sklearn_pca = sklearnPCA(n_components=3)
    dataReduced = sklearn_pca.fit_transform(dataScaled)
    print "Variance Accounted for:", sklearn_pca.explained_variance_ratio_

    print "PCA Data Dimensions:", dataReduced.shape

    dataSetList = divideDataSet(dataReduced, .7, .7)
    testData = dataSetList[0]
    clusteringData = dataSetList[1]
    thresholdData = dataSetList[2]
    """plt.plot(dataReduced[0:len(dataReduced), 0], dataReduced[0:len(dataReduced), 1], 'o', markersize=7, color='blue', alpha=0.5, label="provenance data")
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')"""
    # use_colours = {0: 'green', 1: 'red'}
    # use_colours = {'LogAttribute': 'blue', 'GenerateFlowFile': 'green', 'ExecuteScript': 'red', 'Input Port': 'black', 'PutFile': 'purple'}
    classes = []
    for data in rawData:
        if data['componentName'] not in classes:
            classes.append(data['componentName'])
    print classes
    class_colours = cm.Set1(numpy.linspace(0, 1, len(classes)))
    """for i in xrange(len(classes)):
        value = i/float(len(classes))*255
        rgbTuple = (value, value, value)
        class_colours.append(rgbTuple)"""
    use_colours = dict(zip(classes, class_colours))
    use_sizes = {0: 10, 1: 50}
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
           color=[use_colours[x["componentName"]] for x in rawData],     # marker colour
           marker='o',                                # marker shape
           s=[use_sizes[x[1]] for x in groundTruth]          # marker size
           )
    
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes, loc = 4, fontsize=10)
    # color=[use_colours[x[1]] for x in groundTruth]
    plt.show()
    """for i in xrange(0, 80, 20):
        for j in xrange(0, 100, 45):
            ax.view_init(elev=i, azim=j)
            plt.savefig(saveFigureDirectory + flowName + " elev"+str(i)+" angle"+str(j)+".png")"""
    # print "Clustering Data Size:", clusteringData.shape
    # finding cluster centroids
    """sklearnKMeans = KMeans(n_clusters=8)
    sklearnKMeans.fit(clusteringData)
    clusters = sklearnKMeans.cluster_centers_
    threshold = calculateThreshold(thresholdData, clusters)
    print "cluster coordinates:"
    print clusters
    print "threshold value:", threshold"""
    print "script complete"

# def findAnomalyThreshold(clusters, thresholdDataSet):


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


def calculateThreshold(thresholdData, clusterCoordinateList):
    averageMaxDistance = 0
    for dataPoint in thresholdData:
        distanceList = []
        for clusterCoord in clusterCoordinateList:
            distance = numpy.linalg.norm(dataPoint - clusterCoord)
            distanceList.append(distance)
        maxDistance = numpy.amax(distanceList)
        averageMaxDistance = averageMaxDistance + maxDistance
    totalThresholdDataCount = thresholdData.shape[0]
    averageMaxDistance = averageMaxDistance/totalThresholdDataCount
    return averageMaxDistance


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
