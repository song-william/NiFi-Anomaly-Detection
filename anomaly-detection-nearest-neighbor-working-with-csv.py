import numpy
import scipy
import random
import copy
import glob
import json
import itertools
import time
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
print "scipy version: " + scipy.__version__
print "numpy version: " + numpy.__version__


def main():
    # actual provenance data of flow files
    flowFileData = []

    # list of features used in model
    listFeatures = ["eventType", "entitySize", "componentId", "durationMillis"]

    # features that need to be type casted to int
    intFeatures = ["entitySize", "durationMillis"]
    fileDirectory = "/Users/wsong/Desktop/nifi/provenance-data/working-with-csv-sample2/*"

    print "loading data"
    flowFileData = loadProvenanceData(fileDirectory, 400000)

    removeProvenanceReporterContamination(flowFileData)

    rawData = copy.deepcopy(flowFileData)
    groundTruth = findGroundTruth(flowFileData)

    cleanFeatures(flowFileData, listFeatures)

    # cast integer features to int
    for dataPoint in flowFileData:
        for feature in intFeatures:
            dataPoint[feature] = float(dataPoint[feature])

    """for i in range(0,10):
        print flowFileData[i]
    print flowFileData[len(flowFileData) - 1]
    dataset = np.loadtxt('/Users/wsong/Desktop/nifi/test-set-1-clean.csv')"""

    # loads features from a dictionary
    # link for reference:
    # http://scikit-learn.org/stable/modules/feature_extraction.html#dict-feature-extraction
    print "-"*40
    print "Training Model"

    start_time = time.time()
    vec = DictVectorizer()
    data = vec.fit_transform(flowFileData).toarray()

    # command to print list
    """numpy.set_printoptions(threshold='nan')
    for  i in xrange(10):
        print data[i]"""

    dataScaled = preprocessing.scale(data)

    """numpy.set_printoptions(threshold='nan')
    for i in xrange(10):
        randint = random.randrange(1,1000)
        print dataScaled[randint]"""

    print "Original data Dimensions:", dataScaled.shape

    # run PCA
    sklearn_pca = sklearnPCA(n_components=.99)
    # sklearn_pca = sklearnPCA(n_components=2)
    dataReduced = sklearn_pca.fit_transform(dataScaled)
    print "PCA Data Dimensions:", dataReduced.shape
    # randomize order of dataset

    # indexing tree
    print "indexing tree"
    dataTree = KDTree(dataReduced, leaf_size=30, metric='euclidean')
    anomalyScores = numpy.asarray(findAnomalyScores(dataReduced, dataTree, 10))
    min_max_scaler = preprocessing.MinMaxScaler()
    anomalyScores = min_max_scaler.fit_transform(anomalyScores.reshape(-1, 1))
    
    # return top ten anomalies
    # label time based anomalies


    trueValues = [x[1] for x in groundTruth]
    fpr, tpr, threshold = roc_curve(trueValues, anomalyScores)
    print "fpr"
    print fpr
    print "tpr"
    print tpr
    print "threshold"
    print threshold
    roc_auc = roc_auc_score(trueValues, anomalyScores)
    print "ROC_AUC value:", roc_auc
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC 1-1.1ms anomalies')
    plt.legend(loc="lower right")
    plt.show()
    exit()
    # clusters = sklearnKMeans.cluster_centers_
    # threshold = calculateThresholdGlobal(dataReduced, clusters)
    print "threshold value:", threshold
    labels = sklearnKMeans.labels_
    outlierList = findOutliers(clusters, dataReduced, labels, threshold)
    print "Training Complete"
    print "Time to train:", time.time() - start_time
    outlierIndexes = [x['index'] for x in outlierList]
    evaulations = evaulatePredictions(outlierIndexes, trueValues)
    print "precision score:", evaulations[0]
    print "recall score:", evaulations[1]

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


def calculateThresholdGlobal(thresholdData, clusterCoordinateList):
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
        index += 1
    return provenanceData


def findGroundTruth(provenanceData):
    groundTruth = []
    for event in provenanceData:
        componentType = event['componentType']
        if componentType == 'ExecuteScript' and event['updatedAttributes']['anomaly'] == 'y':
            groundTruth.append((event['eventId'], 1))
        else:
            groundTruth.append((event['eventId'], 0))
    return groundTruth


def removeProvenanceReporterContamination(flowFileData):
    flowFileData[:] = [event for event in flowFileData if event['componentName'] != 'ProvenanceData' and event['componentName'] != 'PutProvenance']


# data is coordinates of data points
# labels is correspoind centroid of datapoint


def findOutliers(clusterCenters, data, labels, threshold):
    # i is index of point, clusterIndex is the index of the cluster centroid
    outliers = []
    iterable = enumerate(itertools.izip(labels, data))
    for i, (clusterIndex, dataCoord) in iterable:
        distance = numpy.linalg.norm(clusterCenters[clusterIndex] - dataCoord)
        if distance > threshold:
            outliers.append({"distance": distance, "index": i})
    return outliers
"""for i, (f,b) in enumerate(itertools.izip(foo, bar)):
    print(f,b)"""


def evaulatePredictions(outlierIndexes, trueValues):
    predictions = [0]*len(trueValues)
    for index in outlierIndexes:
        predictions[index] = 1
    # percentage of positive prediction actually being correct
    precisionScore = precision_score(trueValues, predictions)
    # percentage of all true positives being detected
    recallScore = recall_score(trueValues, predictions)
    return (precisionScore, recallScore)


def findAnomalyScores(provenanceData, dataTree, k):
    anomalyScores = []
    for dataPoint in provenanceData:
        distances, indexes = dataTree.query(dataPoint.reshape(1, -1), k=k)
        anomalyScores.append(numpy.average(distances))
    return anomalyScores



main()
