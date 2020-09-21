'''
Note: the final generated data has a minor mismatch of numbers, this is mainly because I downsampling 10% on training and testing respectively instead of 10% on training+testing

for the generated data:
IDS_new_Infilteration.npz: Counter({0: 66245, 2: 43486, 1: 11731, 3: 9238})
IDS_new_SSH.npz: Counter({0: 66245, 2: 43486, 1: 11732, 3: 9237})
IDS_new_Hulk.npz: Counter({0: 66245, 2: 43487, 1: 11731, 3: 9237})
'''

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)

import sys
import numpy as np
import argparse
import logging

from collections import Counter
from pprint import pformat
from timeit import default_timer as timer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from cade.config import config


# On our own lab server
DATA_FOLDER = config['IDS2018_clean']
UNNORMALIZED_SAVE_FOLDER = os.path.join(config['IDS2018'], 'unnormalized')

cwd = os.getcwd() # IDS_data_preprocess
root_dir = os.path.dirname(cwd) # CADE
SAVE_FOLDER = os.path.join(root_dir, 'data/') # CADE/data

TRAFFIC_TYPE_LIST = np.array(['Benign', 'FTP-BruteForce', 'SSH-Bruteforce', 'DoS attacks-GoldenEye',
                             'DoS attacks-Slowloris', 'DoS attacks-SlowHTTPTest','DoS attacks-Hulk',
                             'DDoS attacks-LOIC-HTTP', 'DDOS attack-LOIC-UDP', 'DDOS attack-HOIC',
                             'Brute Force -Web', 'Brute Force -XSS', 'SQL injection',
                             'Infilteration', 'Bot'])
ALL_FILES_LIST = ['02_14_2018', '02_15_2018', '02_16_2018', '02_21_2018', '02_22_2018',
                  '02_23_2018', '02_28_2018', '03_01_2018', '03_02_2018', '02_20_2018']

def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def main():
    args = parse_args()

    ''' parse required training and testing files, concatenate and resplit them. '''
    saved_unnormalized_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, f'{args.name}_unnormalized.npz')
    X_train, X_test, y_train, y_test = split_data(args, saved_unnormalized_path)

    ''' normalize train, test and save them to file. '''
    save_path = os.path.join(SAVE_FOLDER, f'{args.name}.npz')
    normalize(X_train, X_test, y_train, y_test, args.sampling_ratio, save_path)


def parse_args():
    """Parse the command line configuration for a particular run.

    Raises:
        ValueError: if the tree value for RandomForest is negative.

    Returns:
        argparse.Namespace -- a set of parsed arguments.
    """
    p = argparse.ArgumentParser()

    p.add_argument('--name',
                    help='The name of the generated dataset would be as name.npz.')
    p.add_argument('--benign',
                    help='Specify which day of benign data will be used.')

    p.add_argument('--mal',
                    help='The date and type of malicious traffic, would also be split into training and testing. \
                          a list of file names (indicate by date) and type of traffic for training, separated by "/". \
                          e.g., "02_14_2018,"SSH-Bruteforce"/02_16_2018,"DoS attacks-Hulk"" includes the \
                          02/14 SSH traffic and 02/16 Hulk traffic.')
    p.add_argument('--new-mal',
                    help='The date and type of malicious traffic (as new family) that would be added into the \
                         testing set. Similar as --mal argument.')
    p.add_argument('--sampling-ratio', type=float,
                    default=True,
                    help='The ratio of downsampling.')
    args = p.parse_args()
    logging.warning('Running with configuration: \n' + pformat(vars(args)))

    return args


def split_data(args, saved_unnormalized_path):
    if os.path.exists(saved_unnormalized_path):
        raw_data = np.load(saved_unnormalized_path)
        X_train, X_test = raw_data['X_train'], raw_data['X_test']
        y_train, y_test = raw_data['y_train'], raw_data['y_test']
    else:
        seen_mal_types_dict = get_needed_file_types_dict(args.mal)

        new_file_types_dict = get_needed_file_types_dict(args.new_mal)

        # Extract benign feature vectors and labels
        X_benign, y_benign = extract_data_by_category(args.benign, 'Benign')

        # Extract malicious feature vectors and labels
        X_mal_list, y_mal_list = [], []
        for mal_day, mal_category in seen_mal_types_dict.items():
            X_mal, y_mal = extract_data_by_category(mal_day, mal_category)
            X_mal_list.append(X_mal)
            y_mal_list.append(y_mal)

        for mal_day, mal_category in new_file_types_dict.items():  # only one (key, value) in this dict.
            X_new, y_new = extract_data_by_category(mal_day, mal_category)

        for idx, (X, y) in enumerate(zip([X_benign] + X_mal_list, [y_benign] + y_mal_list)):
            X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X, y, test_size=0.2, shuffle=False)
            if idx == 0:
                X_train, X_test, y_train, y_test = X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp
            else:
                X_train = np.concatenate((X_train, X_train_tmp), axis=0)
                X_test = np.concatenate((X_test, X_test_tmp), axis=0)
                y_train = np.concatenate((y_train, y_train_tmp), axis=0)
                y_test = np.concatenate((y_test, y_test_tmp), axis=0)

        print(f'before X_train: {X_train.shape}, y_train: {y_train.shape}')
        print(f'before X_test: {X_test.shape}, y_test: {y_test.shape}')
        print(f'before y_train Counter: {Counter(y_train)}')
        print(f'before y_test Counter: {Counter(y_test)}')

        # add the new family benign and malicious samples to the testing set.
        X_test = np.vstack((X_test, X_new))
        y_test = np.hstack((y_test, y_new))

        print(f'After X_train: {X_train.shape}, y_train: {y_train.shape}')
        print(f'After X_test: {X_test.shape}, y_test: {y_test.shape}')
        print(f'After y_train Counter: {Counter(y_train)}')
        print(f'After y_test Counter: {Counter(y_test)}')

        np.savez_compressed(saved_unnormalized_path,
                            X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)
    return X_train, X_test, y_train, y_test


def get_needed_file_types_dict(args_str):
    # return a dict with filename as key, list of needed traffic types as value.
    # e.g., {'02_15_2018': ['DoS attacks-GoldenEye']}
    data_list = args_str.split('/')
    data_file_types_dict = {}
    for data in data_list:
        filename, traffic_type = data.split(',')
        data_file_types_dict[filename] = traffic_type
    print(f'data_file_types_dict: {data_file_types_dict}')
    return data_file_types_dict


def extract_data_by_category(single_day, category):
    data_file_path = os.path.join(DATA_FOLDER, single_day + '.npz')
    raw_data = np.load(data_file_path)
    X, y, y_name = raw_data['X'], raw_data['y'], raw_data['y_name']
    data_filter = np.where(y_name == category)[0]
    data = X[data_filter]
    label = y[data_filter]
    sort_idx = np.argsort(data[:, 2], kind='mergesort')
    sorted_data = data[sort_idx]
    sorted_label = label[sort_idx]
    print(f'sorted_data {category}.shape: {sorted_data.shape}')
    print(f'sorted_label {category}.shape: {sorted_label.shape}')
    return sorted_data, sorted_label


def normalize(X_train, X_test, y_train, y_test, ratio, save_path):
    print(f'y_train unique: {np.unique(y_train)}')

    ''' downsampling '''
    X_train, y_train = downsampling(X_train, y_train, ratio, phase='train')
    X_test, y_test = downsampling(X_test, y_test, ratio, phase='test')

    ''' calculate frequency for Dst Port feature in the training set, change the port feature to
    high (0), medium (1), low (2), then one hot encoder to a 3-dimensional array, then fit testing set. '''
    training_ports = X_train[:, 0]
    training_ports_counter = Counter(training_ports)
    print(f'training ports top 20: {training_ports_counter.most_common(20)}')

    high_freq_port_list = []
    medium_freq_port_list = []
    low_freq_port_list = []
    for port in training_ports_counter:
        count = training_ports_counter[port]
        if count >= 10000:
            high_freq_port_list.append(port)
        elif count >= 1000:
            medium_freq_port_list.append(port)
        else:
            low_freq_port_list.append(port)

    training_ports_transform = transform_ports_to_categorical(training_ports, high_freq_port_list,
                                                              medium_freq_port_list, low_freq_port_list)
    port_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    training_ports_transform = np.array(training_ports_transform).reshape(len(training_ports_transform), 1)
    training_ports_encoded = port_encoder.fit_transform(training_ports_transform)
    print(f'training_ports_encoded: {training_ports_encoded[0:10, :]}')
    print(f'training_ports_encoded shape: {training_ports_encoded.shape}')

    ''' One hot encoding the protocol feature, it would produce a 3-dimensional vector'''
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # ignore will encode unseen protocol as all 0
    training_protocols = X_train[:, 1].reshape(len(X_train), 1)  # when keeping Dst port, protocol is the 2nd feature
    print(f'training_protocols unique: {np.unique(training_protocols)}')
    training_protocols_encoded = encoder.fit_transform(training_protocols)

    ''' normalize other features in the training set '''
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train[:, 2:])  # MinMax the rest of the features
    print(f'training_protocols_encoded: {training_protocols_encoded.shape}')
    print(f'X_train_scale: {X_train_scale.shape}')
    X_old = np.concatenate((training_ports_encoded, training_protocols_encoded), axis=1)
    X_old = np.concatenate((X_old, X_train_scale), axis=1)
    y_old = y_train.astype('int32')

    ''' normalize Test '''
    testing_ports = X_test[:, 0]
    testing_ports_transform = transform_ports_to_categorical(testing_ports, high_freq_port_list,
                                                              medium_freq_port_list, low_freq_port_list)
    testing_ports_transform = np.array(testing_ports_transform).reshape(len(testing_ports_transform), 1)
    testing_ports_encoded = port_encoder.transform(testing_ports_transform)
    print(f'testing_ports_encoded: {testing_ports_encoded[0:10, :]}')

    test_protocols = X_test[:, 1].reshape(len(X_test), 1)
    test_protocols_encoded = encoder.transform(test_protocols)

    X_test_scale = scaler.transform(X_test[:, 2:])
    X_new_normalize = np.concatenate((testing_ports_encoded, test_protocols_encoded), axis=1)
    X_new_normalize = np.concatenate((X_new_normalize, X_test_scale), axis=1)
    X_new = X_new_normalize
    y_new = y_test.astype('int32')

    print(f'X_old: {X_old.shape}, y_old: {y_old.shape}')
    print(f'X_new: {X_new.shape}, y_new: {y_new.shape}')
    print(f'y_old labels: {Counter(y_old)}')
    print(f'y_new labels: {Counter(y_new)}')
    np.savez_compressed(save_path,
                        X_train=X_old, y_train=y_old,
                        X_test=X_new, y_test=y_new)
    print('generated data file saved')

    stats(X_old, X_new, y_old, y_new)

    # take a look at the normalized X_new feature value (without readjusting max to 1)
    stats_data_helper(X_new_normalize, 'without adjusting max to 1')


def downsampling(X_train, y_train, ratio, phase):
    # Random sampling data
    '''
    Note here for benign traffic, we random sampling from all the benign traffic
    with no consideration of date. (since we only use one day of benign data)
    '''
    for idx, family in enumerate(np.unique(y_train)):
        family_idx = np.where(y_train == family)[0]
        family_size = len(family_idx)
        filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
        X_train_family = X_train[filter_idx, :]
        y_train_family = y_train[filter_idx]
        print(f'idx: {idx}\tfamily: {family}')
        print(f'X_train_family: {X_train_family.shape}')
        print(f'y_train_family: {Counter(y_train_family)}\n\n')
        if idx == 0:
            X_train_sampling = X_train_family
            y_train_sampling = y_train_family
        else:
            X_train_sampling = np.concatenate((X_train_sampling, X_train_family), axis=0)
            y_train_sampling = np.concatenate((y_train_sampling, y_train_family), axis=0)
    return X_train_sampling, y_train_sampling


def transform_ports_to_categorical(ports, high_freq_port_list, medium_freq_port_list, low_freq_port_list):
    ports_transform = []
    for port in ports:
        if port in high_freq_port_list:
            ports_transform.append(0)
        elif port in medium_freq_port_list:
            ports_transform.append(1)
        else:
            ports_transform.append(2)
    return ports_transform


def stats_data_helper(X, data_type):
    print('==================')
    print(f'feature stats for {data_type}')
    print(f'min: {np.min(X, axis=0)}')
    print(f'avg: {np.average(X, axis=0)}')
    print(f'max: {np.max(X, axis=0)}')


def stats_label_helper(y, data_type):
    print(f'label stats for {data_type}')
    print(f'{Counter(y)}')
    print('==================')


def stats(X_old, X_new, y_old, y_new):
    stats_data_helper(X_old, 'old')
    stats_label_helper(y_old, 'old')
    stats_data_helper(X_new, 'new')
    stats_label_helper(y_new, 'new')


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    print(f'time elapsed: {end - start}')
