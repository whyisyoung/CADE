"""
data.py
~~~~~~~

Functions for caching and loading data.

"""
import random
random.seed(1)

import os, sys
import logging
import numpy as np

from timeit import default_timer as timer
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict, Counter
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import cade.utils as utils
from cade.config import config


def load_features(dataset, newfamily, folder='data/'):
    logging.info('Loading ' + dataset + ' feature vectors and labels...')
    filepath = os.path.join(folder, dataset + '.npz')
    data = np.load(filepath)
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

    logging.debug(f'before label adjusting: y_train: {Counter(y_train)}\n  y_test: {Counter(y_test)}')

    if 'drebin' in dataset:
        PERSISTENT_NEW_FAMILY = 7
    elif 'IDS' in dataset:
        PERSISTENT_NEW_FAMILY = 3
    elif 'bluehex' in dataset:
        PERSISTENT_NEW_FAMILY = newfamily
    else:
        logging.error(f'dataset {dataset} not supported')
        sys.exit(-4)

    '''transform training set to continuous labels, always use the biggest label as the unseen family'''
    le = LabelEncoder()
    y_train_prime = le.fit_transform(y_train)
    mapping = {}
    for i in range(len(y_train)):
        mapping[y_train[i]] = y_train_prime[i]  # mapping: real label -> converted label

    logging.debug(f'LabelEncoder mapping: {mapping}')

    y_test_prime = np.zeros(shape=y_test.shape, dtype=np.int32)
    for i in range(len(y_test)):
        if y_test[i] not in y_train:  # new family
            y_test_prime[i] = PERSISTENT_NEW_FAMILY
        else:
            y_test_prime[i] = mapping[y_test[i]]

    y_train_prime = np.array(y_train_prime, dtype=np.int32)
    logging.debug(f'after relabeling training: {Counter(y_train_prime)}')
    logging.debug(f'after relabeling testing: {Counter(y_test_prime)}')

    return X_train, y_train_prime, X_test, y_test_prime


def prepare_dataset(args):
    if 'drebin' in args.data:
        prepare_drebin_data(args.data, newfamily=args.newfamily_label)


def prepare_drebin_data(dataset_name, folder='data/', test_ratio=0.2, newfamily=7):
    saved_data_file = os.path.join(folder, f'{dataset_name}.npz')
    if os.path.exists(saved_data_file):
        logging.info(f'{saved_data_file} exists, no need to re-generate')
    else:
        '''Train fit test, use only 7 of top 8 families, sort by timestamp, samples do not have timestamp would be removed'''
        logging.info('Preparing Drebin malware data...')
        raw_feature_vectors_folder = config['drebin']

        intermediate_folder = os.path.join('data', dataset_name) # for saving intermediate data files.
        utils.create_folder(intermediate_folder)
        sha_sorted_by_time, label_sorted_by_time, newfamily_sha_list = sort_drebin_7family_by_time(intermediate_folder, newfamily)
        logging.debug(f'sha_sorted_by_time len: {len(sha_sorted_by_time)}')

        '''split 8 families to training and testing set by timestamp, insert the new family to the testing set'''
        train_shas, test_shas, train_labels, test_labels = split_drebin_train_and_test(sha_sorted_by_time,
                                                                                    label_sorted_by_time,
                                                                                    newfamily_sha_list,
                                                                                    test_ratio,
                                                                                    newfamily)

        '''get all the feature names in the training set'''
        train_feature_names = get_training_full_feature_names(intermediate_folder, newfamily,
                                                            raw_feature_vectors_folder, train_shas)

        '''save all the training set feature vectors'''
        saved_train_vectors = save_training_full_feature_vectors(intermediate_folder, raw_feature_vectors_folder,
                                                                train_shas, train_feature_names, train_labels, newfamily)

        '''feature selection on the training set'''
        selected_features, saved_selected_vectors_file = get_selected_features(intermediate_folder, saved_train_vectors,
                                                                            newfamily, train_feature_names)

        ''' generate the final data by saving feature vectors of both training and testing set'''
        samples = len(test_shas)
        feas = len(selected_features)
        selected_features = list(selected_features)  # numpy array does not have index method
        X_test = np.zeros((samples, feas))
        for sample_idx, sha in enumerate(test_shas):
            sys.stdin = open(f'{raw_feature_vectors_folder}/{sha}')
            lines = sys.stdin.readlines()
            for idx, l in enumerate(lines):
                if l != '\n':
                    try:
                        fea_idx = selected_features.index(l.strip())
                        X_test[sample_idx][fea_idx] = 1
                    except: # ignore unseen features.
                        pass

        y_test = np.array([int(label) for label in test_labels])
        train_data = np.load(saved_selected_vectors_file)
        X_train, y_train = train_data['X_train'], train_data['y_train']
        logging.info(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
        logging.info(f'X_test: {X_test.shape}, y_test: {y_test.shape}')
        np.savez_compressed(saved_data_file,
                            X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)
        for idx, x in enumerate(X_test):
            if np.all(x == 0):
                logging.warning(f'X_test {idx} all 0')

        logging.info('Preparing Drebin malware data finished')


def sort_drebin_7family_by_time(intermediate_folder, newfamily):
    '''
    sort the 7 families of top 8 (excluding Opfake because Opfake and FakeInstaller are confusing) by
    timestamp and saved to a new file, according to "latest_modify_time"
    also return the sha list of the new family (the left from the top 8)
    '''
    top8 = ['FakeInstaller', 'DroidKungFu', 'Plankton',
            'GinMaster', 'BaseBridge',
            'Iconosys', 'Kmin', 'FakeDoc']

    sha_family_dict = {}
    sha_timestamp_dict = {}
    newfamily_sha_list = []
    newfamily_sha_timestamp_dict = {}

    with open('data/drebin_metadata.csv', 'r') as f:
        next(f)
        for line in f:
            sha, family, latest_modify_time = line.strip().split(',')
            if family in top8:
                family_int = top8.index(family)
                if family_int == newfamily:
                    newfamily_sha_list.append(sha)
                    if latest_modify_time != 'None':
                        newfamily_sha_timestamp_dict[sha] = datetime.strptime(latest_modify_time, "%Y-%m-%d %H:%M:%S")
                        newfamily_sha_timestamp_dict = OrderedDict(sorted(newfamily_sha_timestamp_dict.items(),
                                                                        key=lambda x: x[1], reverse=False))
                else:
                    if latest_modify_time != 'None':
                        sha_family_dict[sha] = family_int
                        sha_timestamp_dict[sha] = datetime.strptime(latest_modify_time, "%Y-%m-%d %H:%M:%S")
                        sha_timestamp_dict = OrderedDict(sorted(sha_timestamp_dict.items(),
                                                                key=lambda x: x[1], reverse=False))


    sha_sorted_by_time = []
    label_sorted_by_time = []
    saved_file = os.path.join(intermediate_folder, f'drebin_new{newfamily}_sha_timestamp_family.csv')
    with open(saved_file, 'w') as f:
        f.write('sha256,timestamp,family\n')
        for sha, ts in sha_timestamp_dict.items():
            sha_sorted_by_time.append(sha)
            label_sorted_by_time.append(sha_family_dict[sha])
            f.write(f'{sha},{ts},{sha_family_dict[sha]}\n')
        for sha, ts in newfamily_sha_timestamp_dict.items():
            f.write(f'{sha},{ts},{newfamily}\n')

    return sha_sorted_by_time, label_sorted_by_time, newfamily_sha_list


def split_drebin_train_and_test(sha_sorted_by_time, label_sorted_by_time, newfamily_sha_list, test_ratio, newfamily):
    test_num = int(len(sha_sorted_by_time) * test_ratio)
    train_shas = sha_sorted_by_time[0:-test_num]
    train_labels = label_sorted_by_time[0:-test_num]
    test_shas = sha_sorted_by_time[-test_num:] + newfamily_sha_list
    test_labels = label_sorted_by_time[-test_num:] + [newfamily] * len(newfamily_sha_list)
    logging.debug(f'train_shas: {len(train_shas)}, test_shas: {len(test_shas)}')

    return train_shas, test_shas, train_labels, test_labels


def get_training_full_feature_names(intermediate_folder, newfamily, raw_feature_vectors_folder, train_shas):
    saved_train_feature_file = os.path.join(intermediate_folder, f'drebin_new{newfamily}_full_training_features.txt')
    if os.path.exists(saved_train_feature_file):
        train_feature_names = []
        with open(saved_train_feature_file, 'r') as f:
            for line in f:
                train_feature_names.append(line.strip())
    else:
        train_feature_names = set()
        for sha in train_shas:
            sys.stdin = open(f'{raw_feature_vectors_folder}/{sha}')
            lines = sys.stdin.readlines()
            for l in lines:
                if l != '\n':
                    train_feature_names.add(l.strip())

        train_feature_names = sorted(list(train_feature_names))
        logging.info(f'[drebin-new{newfamily}] # of features in training set: {len(train_feature_names)}')
        with open(saved_train_feature_file, 'w') as f:
            for fea in train_feature_names:
                f.write(fea + '\n')

    return train_feature_names


def save_training_full_feature_vectors(intermediate_folder, raw_feature_vectors_folder,
                                       train_shas, train_feature_names, train_labels, newfamily):
    saved_train_vectors = os.path.join(intermediate_folder, f'drebin_new{newfamily}_train_full_feature_vectors.npz')
    if not os.path.exists(saved_train_vectors):
        samples = len(train_shas)
        feas = len(train_feature_names)
        X = np.zeros((samples, feas))
        for sample_idx, sha in enumerate(train_shas):
            sys.stdin = open(f'{raw_feature_vectors_folder}/{sha}')
            lines = sys.stdin.readlines()
            for l in lines:
                if l != '\n':
                    fea_idx = train_feature_names.index(l.strip())
                    X[sample_idx][fea_idx] = 1

        y = np.array([int(label) for label in train_labels])
        np.savez_compressed(saved_train_vectors, X_train=X, y_train=y)

    return saved_train_vectors


def get_selected_features(intermediate_folder, saved_train_vectors, newfamily, train_feature_names):
    train_data = np.load(saved_train_vectors)
    X, y = train_data['X_train'], train_data['y_train']
    logging.debug(f'[drebin_new_{newfamily}] before feature selection X shape: {X.shape}')
    selector = VarianceThreshold(0.003)
    X_select = selector.fit_transform(X)
    logging.debug(f'[drebin_new_{newfamily}] after feature selection X_select shape: {X_select.shape}')

    selected_feature_indices = selector.get_support(indices=True)
    # logging.debug(f'selected_feature_indices: {list(selected_feature_indices)}')
    selected_features = np.array(train_feature_names)[selected_feature_indices]

    ''' save selected features and corresponding feature vectors of training set '''
    saved_selected_feature_file = os.path.join(intermediate_folder, f'drebin_new{newfamily}_train_selected_features.txt')
    if not os.path.exists(saved_selected_feature_file):
        with open(saved_selected_feature_file, 'w') as fout:
            for fea in selected_features:
                fout.write(f'{fea}\n')
    saved_selected_vectors_file = os.path.join(intermediate_folder, f'drebin_new{newfamily}_train_selected_feature_vectors.npz')
    if not os.path.exists(saved_selected_vectors_file):
        np.savez_compressed(saved_selected_vectors_file, X_train=X_select, y_train=y)

    return selected_features, saved_selected_vectors_file


def epoch_batches(X_train, y_train, batch_size, similar_samples_ratio):
    '''
        used for contrastive autoencoder split data into pairs of same label and different labels
        code was adapted from https://github.com/mmasana/OoD_Mining.
    '''
    if batch_size % 4 == 0:
        half_size = int(batch_size / 2) # the really used batch_size for each batch. Another half data is filled by similar and dissimilar samples.
    else:
        logging.error('batch_size should be a multiple of 4.')
        sys.exit(-1)

    # Divide data into batches. # TODO: ignore the last batch for now, maybe there is a better way to address this.
    batch_count = int(X_train.shape[0] / half_size)
    logging.debug(f'batch_count: {batch_count}')  # -> 118
    num_sim = int(batch_size * similar_samples_ratio)  # 64 * 0.25 = 16
    b_out_x = np.zeros([batch_count, batch_size, X_train.shape[1]])
    b_out_y = np.zeros([batch_count, batch_size], dtype=int)
    logging.debug(f'b_out_x: {b_out_x.shape}, b_out_y: {b_out_y.shape}')

    random_idx = np.random.permutation(X_train.shape[0]) # random shuffle the batches
    # split the random shuffled X_train and y_train to batch_count shares
    b_out_x[:, :half_size] = np.split(X_train[random_idx[: batch_count * half_size]], batch_count)
    b_out_y[:, :half_size] = np.split(y_train[random_idx[: batch_count * half_size]], batch_count)

    tmp = random_idx[half_size]

    # NOTE: if error here, it's because we didn't convert X_train and X_test as np.float32 when generating the npz file.
    assert np.all(X_train[tmp] == b_out_x[1, 0])  # to check if the split is correct

    # Sort data by label
    index_cls, index_no_cls = [], []
    ''' NOTE: if we want to adapt to training label non-continuing, e.g., [0,1,2,3,4,5,7], but this would cause
    b_out_y[b, m] list index out of range. So we should convert [0,1,2,3,4,5,7] to [0,1,2,3,4,5,6] in the training set.'''
    for label in range(len(np.unique(y_train))):
        index_cls.append(np.where(y_train == label)[0]) # each row shows the index of y_train where y_train == label
        index_no_cls.append(np.where(y_train != label)[0])

    index_cls_len = [len(e) for e in index_cls]
    logging.debug(f'index_cls len: {index_cls_len}')
    index_no_cls_len = [len(e) for e in index_no_cls]
    logging.debug(f'index_no_cls len: {index_no_cls_len}')

    logging.debug('generating the batches and pairs...')
    # Generate the pairs
    logging.debug(f'num_sim: {num_sim}')
    logging.debug(f'half_size: {half_size}')
    start = timer()
    for b in range(batch_count):
        # Get similar samples
        for m in range(0, num_sim):
            # random sampling without replacement, randomly pick an index from y_train
            # where y_train[index] = b_out_y[b, m]
            # NOTE: list() operation is very slow, random.sample is also slower than np.random.choice()
            # ## pair = random.sample(list(index_cls[b_out_y[b, m]]), 1) would take 80s for each b,
            # np.random.choice() and list() would lead to 130s for each b
            # using only np.random.choice() would be 0.06s for each b
            pair = np.random.choice(index_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]] # pick num_sim samples with the same label
            b_out_y[b, m + half_size] = y_train[pair[0]]
        # pick (half_size - num_sim) dissimilar samples
        for m in range(num_sim, half_size):
            # randomly pick an index from y_train where y_train[index] != b_out_y[b, m]
            pair = np.random.choice(index_no_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]
            b_out_y[b, m + half_size] = y_train[pair[0]]
        # DEBUG
        # if b == 1:
            # b_out_y[0] should looks like this (for simplicity assuming batch_size = 32, half_size = 16)
            # The first half is similar, the second half is dissimilar
            # 1, 2, 4, 8 | 2, 3, 5, 6
            # 1, 2, 4, 8 | 3, 4, 1, 7
            # logging.debug(f'b_out_x[1, 0, :20]: {b_out_x[b, 0, :20]}')
            # logging.debug(f'b_out_y[1]: {b_out_y[b]}')
    end = timer()

    logging.debug(f'split batch finished: {end - start} seconds') # ~10s
    return batch_count, b_out_x, b_out_y
