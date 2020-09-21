"""
config.py
~~~~~~~~~~~

Configuration options for CADE.

"""
import os

# WARNING: you need to change the values to the path where you download the dataset.
config = {
    'drebin': '/home/datashare/drebin/raw/feature_vectors/', # which contains 129,013 files, each containing the raw feature vectors of each sample
    'IDS2018': 'home/datashare/IDS2018/', # which contains 10 csv files (e.g., 02_14_2018.csv) download from IDS2018 dataset.
    'IDS2018_clean': '/home/datashare/IDS2018/clean/' # contains 10 numpy compressed data (*.npz) after running CADE/IDS_data_preprocess/clean_data.py, coressponding to the 10 cvs files.
}