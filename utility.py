import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn_extra.cluster import KMedoids

DATASET_DIR = "dataset"

class dataset():
    def loadData(self, file, selector):
        filename = DATASET_DIR+"/"+file
        if selector != None:
            filename = f"{filename}_{selector}.xlsx"
        self.dataframe = pd.read_excel(filename, skiprows=1, usecols="B:CN")

    def numpyArrTransform(self):
        return np.array(self.dataframe.iloc[:, 1:90])
    
    def getDataframe(self):
        return self.dataframe
    
    def getCentroids(self):
        return np.array([
            self.dataframe.iloc[28, 1:90], 
            self.dataframe.iloc[27, 1:90], 
            self.dataframe.iloc[7, 1:90]])
    
    def setLabel(self, lables):
        self.dataframe['lable'] = lables
    
    def searchLabel(self, lable):
        d = self.dataframe.loc[self.dataframe['lable'] == lable, 'Kabupaten/Kota']
        return d.values.tolist()
        

class Clustering:
    def __init__(self):
        self.score = 0
        
    def define(self, centroids, cluster_algorithm: str):
        match cluster_algorithm:
            case "K-Means":
                if centroids.size == 0:
                    centroids = "k-means++"
                self.cluster = KMeans(n_clusters=3, init=centroids, random_state=0, n_init=1, verbose=1, max_iter=5)
            case "K-Medoids":
                if centroids.size == 0:
                    centroids = "euclidean"
                self.cluster = KMedoids(n_clusters=3, init=centroids, random_state=0)
    def get(self):
        return self.data
    
    def fit(self, data):
        if data.size == 0:
            return None
        self.data = data
        self.cluster.fit(self.data)
        self.label = self.cluster.labels_
        return self.cluster.labels_
    
    def dbi(self):
        if self.label.size == 0:
            return None
        if self.score == 0:
            data = self.data
            self.score = davies_bouldin_score(data, self.label)
        return self.score

def loadDataSelector(dataset_file):
    dataselector = {}
    for i in dataset_file:
        s = i.split("_")
        if s[0] in dataselector:
            dataselector[s[0]].append(s[1][:-5])
            continue
        dataselector[s[0]] = [s[1][:-5]]
    return dataselector