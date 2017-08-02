import fastcluster
import json
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from matplotlib.mlab import PCA
from sklearn.manifold import locally_linear_embedding
from mpl_toolkits.mplot3d import Axes3D
import math
import Queue
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
if sys.hexversion < 0x03000000:
    from urllib2 import urlopen
else:
	from urllib2.request import urlopen

# ===================VALUES TO BE SET=======================================
thresholdValue = float(sys.argv[1])
dimensions = int(sys.argv[2])
inputFile = sys.argv[3]
# ==========================================================================
# Create and initialize files
# f0 = open("outputMatrixFull375.txt")
# f1 = open("outputMatrixFull375.txt")
# f2 = open("outputMatrixFull375.txt")
f0 = open(inputFile)
f1 = open(inputFile)
f2 = open(inputFile)

buildNums = []
for i in range(0, dimensions):
	buildNums.append(i)

# Initialize datapoints and their labels
fileNames = np.loadtxt(f0, dtype=np.str_, delimiter=',', usecols=[dimensions+1])
value = np.loadtxt(f1, delimiter=',', usecols=buildNums)
label = np.loadtxt(f2, dtype=np.str_, delimiter=',', usecols=[dimensions])


f0.close()
f1.close()
f2.close()
N = len(value)

finalLabels = []
for i in range(0, len(label)):
	temp = label[i].split("|")
	finalLabels.append(temp)


#Create a plot of the Merging of the clusters
def plot_with_labels(Z, num_clust):
	threshold = Z[-num_clust + 1, 2]
	dg = dendrogram(Z, no_labels = False, color_threshold = threshold)
	
	b=.1*Z[-1,2]
	plt.bar(np.arange(N)*10,np.ones(N)*b,bottom=-b,width=10,edgecolor='none')
	plt.gca().set_ylim((-b, None))
	plt.show()


#Checks to see if a string exists in a list already
def alreadyExists(listToCheck, word):
	for i in range(0, len(listToCheck)):
		if word == listToCheck[i]:
			return True 
	return False


#Prints the final clusters and its labels
def printClusterLabels(clusters, labels):
	labelDicts = {}
	for key,value in clusters.items():
		labelDicts[key] = []
		for i in range(0, len(value)):
			stringtoSplit = labels[value[i]]
			splitted = stringtoSplit.split("|")
			for j in range(0, len(splitted)):
				if splitted[j] != "" and not alreadyExists(labelDicts[key], splitted[j]):
					labelDicts[key].append(splitted[j])

	returnLabelDicts = {}
	for key,value in labelDicts.items(): 
		smallDict = {}
		for i in range(0, len(value)):
			splitted = value[i].split(":")
			smallDict[splitted[0]] = splitted[1]

		sortedSmallDict = sorted(smallDict.items(), key=operator.itemgetter(1), reverse=True)
		counter = 0
		buildString = ""
		while counter < 10 and counter < len(sortedSmallDict):
			# print sortedSmallDict[counter], "|",
			buildString = buildString + str(sortedSmallDict[counter]) + "|"
			counter = counter + 1
		returnLabelDicts[key] = buildString
		# print buildString,
		# print "\n============================================================================="
	return returnLabelDicts
	# print "\nFINAL CLUSTER DOCUMENTS:", clusters
	
	# Uncomment following to print out file names for each cluster-------------------------
	# for k,v in clusters.items():
	# 	print "CLUSTER", k, ":"
	# 	for i in range(0, len(v)):
	# 		print fileNames[i]
	# 	print "--------------------------------------------------------------"


#Function to actually merge 2 clusters
#Parameters:	
# cluster1 - First cluster to merge
# cluster2 - Second cluster to merge
# culsters - dicitonary within which to merge the 2 clusters
# num - Number/Key of new cluster to create
def mergeClusters(cluster1, cluster2, clusters, num):
	if cluster1 not in clusters:
		print cluster1, "does not exist"
		return clusters

	if cluster2 not in clusters:
		print cluster2, "does not exist"
		return clusters

	newCluster = []

	for i in range(0, len(clusters[cluster1])):
		newCluster.append(clusters[cluster1][i])
	for i in range(0, len(clusters[cluster2])):
		newCluster.append(clusters[cluster2][i])

	clusters.pop(cluster1, None)
	clusters.pop(cluster2, None)
	clusters[num] = newCluster
	return clusters


#Create initial clusters and then use Z and threshold to start merging the clusters
#Parameters:
# Z - List which dictates merging of clusters
# threshold - Threshold value at which to stop merging clusters
def utulizeFastCluster(Z, threshold, values, labels, fN):
	# print "Starting clustering ..."

	clusters = {}
	clusterFiles = {}
	for i in range(0, len(values)):
		clusters[i] = [i]

	print "Created initial Clusters"
	
	counter = 0
	ClusterNumber = len(values)
	while(counter < len(Z) and Z[counter][2] < threshold):
		clusters = mergeClusters(Z[counter][0], Z[counter][1], clusters, ClusterNumber)
		ClusterNumber = ClusterNumber + 1
		counter = counter + 1
	
	print "Creating JSON with Labels and Filenames of clusters."
	print "May take a few minutes . . ."
	labelClusters = printClusterLabels(clusters, labels)

	for k,v in clusters.items():
		files = []
		for i in range(0, len(v)):
			tempString = fN[v[i]]
			finalString = tempString.split("/")
			files.append(finalString[len(finalString) - 1])
		clusterFiles[k] = files

	return clusters, labelClusters, clusterFiles

#===============================RUN CLUSTERING======================================
D = pdist(value, metric = 'cosine')
for i in range(0, len(D)):
	if math.isnan(D[i]):
		D[i] = 0.0
Z = fastcluster.linkage(D, method = 'complete')

# plot_with_labels(Z, 3)
# print Z

clusters, labelClusters, fileClusters = utulizeFastCluster(Z, thresholdValue, value, label, fileNames)

finalJSON = {}
finalJSON["Clusters"] = clusters
finalJSON["ClusterLabels"] = labelClusters
finalJSON["ClusterFiles"] = fileClusters

with open('clusterData.txt', 'w') as outfile:  
    json.dump(finalJSON, outfile, indent=5)