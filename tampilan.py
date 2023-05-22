import streamlit as st
import pandas as pd
import numpy as np
import utility as ut


dataselector = {
    "K-Means": ["Median", "Mean"],
    "K-Medoids": ["Median"]
}

list_filename = {
    "K-Means_Median": "data_setelah_preprocessing_Lengkap_Median_K-Means2.xlsx",
    "K-Means_Mean": "data_setelah_preprocessingMeanLengkap.xlsx",
    "K-Medoids_Median": "Data_Setelah_Prepocessing_Median_K-Medoids.xlsx"
}

currAlg = 'K-Means'
currFile = 'Median'
onchange = 1
# Judul
st.title('K-Means dan K-Medoids')

# Dropdown
selectedAlg = st.selectbox('Pilih Algoritma', ['K-Means', 'K-Medoids'])

# File Selector
fileSelector = st.selectbox('Pilih dataset', dataselector[selectedAlg])

if currAlg != selectedAlg or currFile != fileSelector:
    onchange = 1
    currAlg = selectedAlg
    currFile = fileSelector

if onchange == 1:
    dataset = ut.dataset()
    cluster = ut.Clustering()

    # mengambil dataset
    filename = list_filename[selectedAlg + "_" + fileSelector]
    dataset.loadData(filename)
    st.write(dataset.getDataframe())

    # inisialisasi clustering
    st.write("Menentukan Centroid")
    st.write(dataset.getCentroids())
    cluster.define(dataset.getCentroids(), selectedAlg)

    # clustering
    arr = dataset.numpyArrTransform()
    label = cluster.fit(arr)

    st.write("Data setelah clustering")
    dataset.setLabel(label)
    clustered = pd.DataFrame(data=[
        dataset.searchLabel(0),
        dataset.searchLabel(1),
        dataset.searchLabel(2)
    ], index=["Cluster 1", "Cluster 2", "Cluster 3"])
    st.write(clustered)
    onchange = 0