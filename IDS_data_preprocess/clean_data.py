'''
clean and save compressed data. Also sort by timestamps, ascending order.

nohup python -u clean_data.py > logs/clean_data.log &

clean strategy:
Traffic with NaN and Infinity values was removed.
Duplicate traffic was removed (duplicate means exactly same traffic).

For "02_20_2018.csv" file, it has 4 extra features in the beninning (Flow ID, Src IP, Src Port, Dst IP,)
-> extra features were removed.

'''

import os, sys
import traceback
import numpy as np


from datetime import datetime
from timeit import default_timer as timer
from collections import Counter
from tqdm import tqdm

from cade.config import config


RAW_DATA_PATH = config['IDS2018']
SAVE_PATH = config['IDS2018_clean']

NORMAL_FILES = ['02_14_2018', '02_15_2018', '02_16_2018', '02_21_2018',
                '02_22_2018', '02_23_2018', '02_28_2018', '03_01_2018', '03_02_2018']

SPECIFIC_FILES = ['02_20_2018']

TRAFFIC_LABEL = {'Benign': 0,
                 'FTP-BruteForce': 1, 'SSH-Bruteforce': 1,
                 'DoS attacks-GoldenEye': 2, 'DoS attacks-Slowloris': 2,
                 'DoS attacks-SlowHTTPTest': 2, 'DoS attacks-Hulk': 2,
                 'Infilteration': 3,
                 'DDoS attacks-LOIC-HTTP': 4, 'DDOS attack-LOIC-UDP': 4, 'DDOS attack-HOIC': 4,
                 'Brute Force -Web': 5,
                 'Brute Force -XSS': 5,
                 'Bot': 6,
                 'SQL Injection': 7
                 }

def clean_single_file(filename, is_specific=False):
    traffic_contain_null_count = 0
    traffic_contain_infinity_count = 0
    traffic_invalid_timestamp_count = 0

    traffics = [] # a list of traffics, including feature vector and label names.

    print(f'cleaning file {filename}...')

    '''remove traffic with NaN and Infinity values, read the file content into a numpy array.'''
    with open(filename, 'r') as f:
        date_str = filename.replace('.csv', '').replace('_', '/').replace(RAW_DATA_PATH, '')
        date_str = date_str[3:5] + '/' + date_str[0:2] + date_str[5:]
        print(f'date_str: {date_str}')
        next(f)
        contents = f.readlines()
        for idx, line in enumerate(contents):
            try:
                line = line.strip().split(',')

                if is_specific:
                    # the 02_20_2018 file has 4 extra features in the beginning
                    line = line[4:]
                if not line[0].isdigit():  # useless traffic, which is not a feature vector of traffic
                    continue
                if 'NaN' in line:
                    traffic_contain_null_count += 1
                    continue
                if 'Infinity' in line:
                    traffic_contain_infinity_count += 1
                    continue
                if date_str not in line[2]:
                    traffic_invalid_timestamp_count += 1
                    continue
                # convert datetime to UNIX timestamp
                line[2] = str(datetime.strptime(line[2], '%d/%m/%Y %H:%M:%S').timestamp())
                traffics.append(line)
            except:
                print(f'{filename} line {idx} error')
                print(traceback.format_exc())
    print('===================\n')
    print(f'total # traffic is {len(contents)}')
    print(f'traffic that has NaN values: {traffic_contain_null_count}')
    print(f'traffic that has Infinity values: {traffic_contain_infinity_count}')
    print(f'traffic that has invalid timestamp: {traffic_invalid_timestamp_count}')

    traffics = np.array(traffics)
    print(f'after removing NaN, Infinity, and invalid, traffic shape: {traffics.shape}')

    '''remove duplicate traffics and save feature vectors, assigned labels, semantic labels to a compressed file.'''
    # np.unique will sort instead of keeping the original order.
    unique_traffics = np.unique(traffics, axis=0)
    sorted_traffics = unique_traffics[np.argsort(unique_traffics[:, 2])]  # sort by the timestamp of the traffic

    X = sorted_traffics[:, 0:-1].astype(np.float)  # feature vectors
    y_name = sorted_traffics[:, -1] # full name indicating the meaning of the label.

    y = [] # assigned labels
    for name in y_name:
        y.append(TRAFFIC_LABEL[name]) # convert label name to number
    y = np.array(y)

    base_filename = os.path.basename(filename)
    save_file_path = os.path.join(SAVE_PATH, base_filename.replace('csv', 'npz'))

    np.savez_compressed(save_file_path, X=X, y=y, y_name=y_name)
    print(f'sorted traffics shape: {X.shape}')
    print(f'labels shape: {y.shape}')
    print(f'label names shape: {y_name.shape}')
    print(f'percentage of kept traffic (removing NaN, Infinity, duplicates): {X.shape[0] / len(contents)}')
    print('===================\n')


def stats():
    for file in NORMAL_FILES + SPECIFIC_FILES:
        print(f'stats for file: {file}')
        file_path = os.path.join(SAVE_PATH, file + '.npz')
        data = np.load(file_path)
        y, y_name = data['y'], data['y_name']
        assigned_labels = Counter(label for label in y)
        semantic_labels = Counter(name for name in y_name)
        print(f'assigned labels: {assigned_labels}')
        print(f'semantic labels: {semantic_labels}')


def main():
    start = timer()
    for file in NORMAL_FILES:
        file_path = os.path.join(RAW_DATA_PATH, file + '.csv')
        clean_single_file(file_path)

    s1 = timer()
    for file in SPECIFIC_FILES:
        file_path = os.path.join(RAW_DATA_PATH, file + '.csv')
        clean_single_file(file_path, is_specific=True)
    e1 = timer()
    print(f'biggest file (4G): {e1 - s1}')

    end = timer()
    print(f'time elapsed: {end - start}')

    '''calc the # of each label in each file'''
    stats()


if __name__ == '__main__':
    main()
