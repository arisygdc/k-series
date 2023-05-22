import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn_extra.cluster import KMedoids

class dataset():
    def __init__(self, data_path):
        self.setData(data_path)
        self.dbi = 0
    
    def setData(self, data_path):
        self.dataframe = pd.read_excel(data_path, skiprows=1, usecols="B:CN")

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
    def __init__(self, cluster_algorithm: str):
        self.ClusterAlg = cluster_algorithm
    
    def define(self, centroids):
        match self.ClusterAlg:
            case "K-Means":
                if centroids.size == 0:
                    centroids = "k-means++"
                self.cluster = KMeans(n_clusters=3, init=centroids, random_state=0, n_init=1, verbose=1, max_iter=5)
            case "K-Medoids":
                if centroids.size == 0:
                    centroids = "euclidean"
                self.cluster = KMedoids(n_clusters=3, init=centroids, random_state=0)

    def fit(self, data):
        if data.size == 0:
            return None
        self.data = data
        self.cluster.fit(self.data)
        return self.cluster.labels_
    
    def dbi(self):
        if self.data.size == 0:
            return None
        if self.dbi == 0:
            self.dbi = davies_bouldin_score(self.data, self.cluster.predict(self.data))
        return self.dbi
            