import streamlit as st
import pandas as pd
import utility as ut
import os

dataset_file = os.listdir(ut.DATASET_DIR)

dataselector = ut.loadDataSelector(dataset_file)

keys = list(dataselector.keys())
currAlg = keys[0]
currFile = dataselector[currAlg][0]
onchange = 1

# Judul
st.title('K-Means dan K-Medoids')

# Dropdown
selectedAlg = st.selectbox('Pilih Algoritma', keys)

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
    dataset.loadData(selectedAlg, fileSelector)
    st.write(dataset.getDataframe())

    # inisialisasi clustering
    st.write("Menentukan Centroid")
    st.write(dataset.getCentroids())
    cluster.define(dataset.getCentroids(), selectedAlg)

    # clustering
    arr = dataset.numpyArrTransform()
    label = cluster.fit(arr)

    dbi = cluster.dbi()
    st.write(f"Data setelah clustering dengan dbi: {dbi}")
    dataset.setLabel(label)
    clustered = pd.DataFrame(data=[
        dataset.searchLabel(0),
        dataset.searchLabel(1),
        dataset.searchLabel(2)
    ], index=["Cluster 1", "Cluster 2", "Cluster 3"])
    st.write(clustered)
    onchange = 0