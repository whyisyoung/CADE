"""
explain_global_approximation_loose_boundary.py
~~~~~~~

Functions for explaining why a sample is a drifting.
For each closest family of the testing samples, build a global approximation model.

For this version: we use in-dist and drift samples from the training set and drift samples from testing set, and synthesized drift based on
testing drift to build a loose approximation model (does not really reflect the exact boundary of the detection module).

"""

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import sys
from functools import partial
import traceback
import logging
import re

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.metrics import accuracy_score, pairwise_distances

from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model

import cade.utils as utils
import cade.classifier as classifier
import cade.mask_exp_by_approximation as mask_exp
from cade.autoencoder import Autoencoder


def explain_drift_samples_per_instance(X_train, y_train, X_test, y_test,
                                       args,
                                       one_by_one_check_result_path,
                                       training_info_for_detect_path,
                                       cae_weights_path,
                                       saved_exp_classifier_folder,
                                       mask_file_path):
    if os.path.exists(mask_file_path):
        logging.info(f'explanation result file {mask_file_path} exists, no need to run explanation module')
    else:
        drift_samples_idx_list, drift_samples_real_labels, \
            drift_samples_closest = get_drift_samples_to_explain(one_by_one_check_result_path)

        mad_threshold = args.mad_threshold

        cae_dims = utils.get_model_dims('Contrastive AE', X_train.shape[1],
                                        args.cae_hidden, len(np.unique(y_train)))

        # load CAE encoder
        encoder_model = load_encoder(cae_dims, cae_weights_path)

        '''get all the drift samples from the testing set, separated by their closest family'''
        test_z_drift_family = get_z_drift_from_testing_set_by_family(X_test, drift_samples_idx_list, drift_samples_closest, encoder_model)

        # build global target explanation model for each family (closest one to the testing samples)
        X_in_family = build_global_exp_model_for_each_closest_family(X_train, y_train,
                                                                     test_z_drift_family, drift_samples_closest,
                                                                     training_info_for_detect_path,
                                                                     mad_threshold, saved_exp_classifier_folder,
                                                                     cae_dims, cae_weights_path)
        '''explain drift per instance '''
        masks = []
        X_drift_list = []
        for idx, sample_idx in tqdm(enumerate(drift_samples_idx_list),
                                    total=len(drift_samples_idx_list),
                                    desc='explain drift'):
            try:
                x_target = X_test[sample_idx]
                X_drift_list.append(x_target)

                closest_family = drift_samples_closest[idx]

                logging.debug(f'idx-[{idx}] closest family: {closest_family}')

                logging.debug(f'[explanation] explain single instance...')
                final_model_path = os.path.join(saved_exp_classifier_folder, f'final_model_family_{closest_family}.h5')
                approximation_mlp_model_path = os.path.join(saved_exp_classifier_folder, f'exp_mlp_family_{closest_family}.h5')

                diff = x_target - X_in_family[closest_family][-1]
                diff_idx = np.where(diff != 0)[0]

                mask = explain_instance(x_target, args.exp_lambda_1, diff_idx, final_model_path)
                masks.append(mask)
                logging.debug(f'[explanation] explain single instance finished...')

            except:
                logging.error(f'idx: {idx}, sample_idx: {sample_idx}')
                logging.error(traceback.format_exc())

        np.savez_compressed(mask_file_path, masks=masks)


def get_drift_samples_to_explain(one_by_one_check_result_path):
    pattern = re.compile('best inspection count: \d+')
    with open(one_by_one_check_result_path, 'r') as f:
        inspect_cnt = int(re.findall(pattern, f.read())[0].replace('best inspection count: ', ''))

    drift_samples_idx_list, drift_samples_real_labels, drift_samples_closest = [], [], []
    with open(one_by_one_check_result_path, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            if idx < inspect_cnt:
                line_data = line.strip().split(',')
                drift_samples_idx_list.append(int(line_data[0]))
                drift_samples_real_labels.append(int(line_data[1]))
                drift_samples_closest.append(int(line_data[2]))

    assert len(drift_samples_idx_list) == inspect_cnt
    assert len(drift_samples_closest) == inspect_cnt

    return drift_samples_idx_list, drift_samples_real_labels, drift_samples_closest


def load_encoder(cae_dims, cae_weights_path):
    K.clear_session()  # be careful with this it may clean up previous loaded models.
    ae = Autoencoder(cae_dims)
    ae_model, encoder_model = ae.build()
    encoder_model.load_weights(cae_weights_path, by_name=True)
    return encoder_model


def get_z_drift_from_testing_set_by_family(X_test, drift_samples_idx_list, drift_samples_closest, encoder_model):
    test_z_drift_family = {}

    X_test_drift = X_test[drift_samples_idx_list]
    z_test_drift = encoder_model.predict(X_test_drift)
    for family in np.unique(drift_samples_closest):
        test_z_drift_family[family] = z_test_drift[np.where(drift_samples_closest == family)[0]]
        logging.debug(f'test_z_drift_family - {family}: {test_z_drift_family[family].shape}')

    return test_z_drift_family


def build_global_exp_model_for_each_closest_family(X_train, y_train,
                                                   test_z_drift_family, drift_samples_closest,
                                                   training_info_for_detect_path,
                                                   mad_threshold, saved_exp_classifier_folder,
                                                   cae_dims, cae_weights_path):
    X_in_family= {}
    for family in np.unique(drift_samples_closest):
        '''first need to synthesize more drift samples to balance in-dist and drift'''
        z_train, z_closest_family, centroid, \
            dis_to_centroid, mad = load_training_info(training_info_for_detect_path, family)

        lower_bound = mad * mad_threshold + np.median(dis_to_centroid)
        logging.critical(f'[family-{family}] distance lower bound (to be an drift): {lower_bound}')

        X_train_family = X_train[np.where(y_train == family)[0]]
        z_in, z_drift, X_in = get_in_and_out_distribution_samples(X_train_family, z_closest_family, dis_to_centroid,
                                                                centroid, mad, mad_threshold)
        X_in_family[family] = X_in

        approximation_mlp_model_path = os.path.join(saved_exp_classifier_folder, f'exp_mlp_family_{family}.h5')

        if os.path.exists(approximation_mlp_model_path):
            logging.info(f'approximation model file {approximation_mlp_model_path} exists, no need to rerun')
        else:
            '''
                only perturb test_z_drift a little bit to synthesize more samples,
                and put them into the detection module
            '''
            test_z_drift = test_z_drift_family[family]

            cnt_syn_drift = len(z_in) - len(test_z_drift) # for IDS data, it's better not to generate more drift data because testing drift are enough.
            if cnt_syn_drift > 0:
                z_syn_in, z_syn_drift = synthesize_local_samples(test_z_drift, cnt_syn_drift, centroid, dis_to_centroid,
                                                            mad, mad_threshold, 'drift')
                if len(z_syn_in) > 0: # len(z_syn_in) == 0 can not be stacked
                    z_in = np.vstack((z_in, z_syn_in))
                if len(z_syn_drift) > 0:
                    z_drift = np.vstack((test_z_drift, z_syn_drift))
                else:
                    z_drift = test_z_drift
            else: # no need to synthesize more drift samples.
                z_drift = test_z_drift
            logging.debug(f'test_z_drift.shape: {test_z_drift.shape}')
            logging.debug(f'[family-{family}]  z_drift.shape: {z_drift.shape}')
            logging.debug(f'[family-{family}]  z_in.shape: {z_in.shape}')

            y_in = np.zeros(shape=(z_in.shape[0], ))
            y_drift = np.ones(shape=(z_drift.shape[0], ))

            ''' build a shallow classifier to distinguish in-distribution and drift samples '''
            logging.info(f'[explantion] build a global classifier for family-{family}...')
            z_weights = None  # DO not use weights at this time.
            NUM_LATENT_FEATURES = z_in.shape[1]
            num_classes = 2 # there are only two classes: in-distribution and drift.
            # NOTE: here use 8-15-2 for drebin, 3-15-2 for IDS
            mlp_dims =  [NUM_LATENT_FEATURES, 15, num_classes]
            dropout_ratio = 0 # do not use dropout here

            build_target_classifier(z_in, z_drift, y_in, y_drift, dropout_ratio,
                                    z_weights, mlp_dims, approximation_mlp_model_path)
            logging.info(f'[explantion] build a global classifier for family-{family} finished')

            ''' combine the encoder and shallow MLP classifier (approximation model) as the final model to explain '''
            final_model_path = os.path.join(saved_exp_classifier_folder, f'final_model_family_{family}.h5')
            combine_encoder_and_approximation_model(cae_dims, mlp_dims, dropout_ratio,
                                                    cae_weights_path, approximation_mlp_model_path, final_model_path)

    return X_in_family



def load_training_info(training_info_for_detect_path, closest_family):
    info = np.load(training_info_for_detect_path)
    z_train = info['z_train']
    z_family = info['z_family']
    centroids = info['centroids']
    dis_family = info['dis_family']
    mad_family = info['mad_family']

    z_closest_family = z_family[closest_family]
    centroid = centroids[closest_family]
    dis_to_centroid = dis_family[closest_family]
    mad = mad_family[closest_family]

    logging.debug(f'z_closest_family shape: {z_closest_family.shape}')
    logging.debug(f'centroid-{closest_family}: {centroid}')
    logging.critical(f'dis_to_centroid median: {np.median(dis_to_centroid)}')
    logging.critical(f'mad-{closest_family}: {mad}')

    return z_train, z_closest_family, centroid, dis_to_centroid, mad


def get_in_and_out_distribution_samples(X_train_family, z_closest_family, dis_to_centroid,
                                        centroid, mad, mad_threshold):

    # step 1: ranked by training samples' distance to the centroid
    dis_to_centroid_inds = np.array(dis_to_centroid).argsort()[::-1]  # dis descending order
    z_closest_family_sorted = z_closest_family[dis_to_centroid_inds]
    X_train_family_sorted = X_train_family[dis_to_centroid_inds]

    # step 2: only keep samples flagged as in-distribution by our detection module
    stop_idx = 0
    for idx, z in enumerate(z_closest_family_sorted):
        dis = np.linalg.norm(z - centroid)
        logging.critical(f'training set drift sample-{idx} latent distance to centroid: {dis}')
        if not detect_if_sample_is_drift(z, centroid, dis_to_centroid, mad, mad_threshold):
            stop_idx = idx
            break

    all_in_distribution = z_closest_family_sorted[stop_idx:, :]
    X_train_family_in_dist = X_train_family_sorted[stop_idx:, :]

    all_out_distribution = z_closest_family_sorted[0:stop_idx, :]

    logging.debug(f'all_in_distribution.shape: {all_in_distribution.shape}')
    logging.debug(f'all_out_distribution.shape: {all_out_distribution.shape}')
    logging.debug(f'training set drift ratio: {len(all_out_distribution) / len(z_closest_family):.3f}')
    logging.debug(f'X_train_family_in_dist.shape: {X_train_family_in_dist.shape}')

    return all_in_distribution, all_out_distribution, X_train_family_in_dist


def synthesize_local_samples(z_group, cnt_syn, centroid, dis_to_centroid, mad, mad_threshold, base_label):
    augment_times = round(cnt_syn / len(z_group)) # No. of times each sample synthesize

    sigma = mad
    logging.debug(f'noise sigma: {sigma}')

    for idx, z in enumerate(z_group):
        syn_list = []
        noise = np.random.normal(0, sigma, size=(len(z), augment_times))
        for i in range(len(z)):
            syn_list.append(z[i] + noise[i])
        z_syn = np.transpose(np.array(syn_list))

        if idx == 0:
            z_syn_total = np.array(z_syn)
        else:
            z_syn_total = np.vstack((z_syn_total, z_syn))

    logging.debug(f'z_syn_total.shape: {z_syn_total.shape}')  # (≥cnt_syn, latent_dim)

    # use the detection module to determine the synthesized samples are in-dist or drift
    z_syn_drift, z_syn_in = [], []
    for i in range(len(z_syn_total)):
        is_drift = detect_if_sample_is_drift(z_syn_total[i], centroid, dis_to_centroid, mad, mad_threshold)
        if is_drift:
            z_syn_drift.append(z_syn_total[i])
            # logging.debug(f'synthesized drift sample to centroid dis - {i}: {np.linalg.norm(z_syn_total[i] - centroid)}')
        else:
            z_syn_in.append(z_syn_total[i])

    z_syn_drift = np.array(z_syn_drift)
    z_syn_in = np.array(z_syn_in)
    logging.debug(f'z_syn_drift.shape: {z_syn_drift.shape}')
    logging.debug(f'z_syn_in.shape: {z_syn_in.shape}')
    return z_syn_in, z_syn_drift


def detect_if_sample_is_drift(z, centroid, dis_to_centroid, mad, mad_threshold):
    dis = np.linalg.norm(z - centroid)
    anomaly = np.abs(dis - np.median(dis_to_centroid)) / mad
    if anomaly > mad_threshold:
        return True
    return False


def assign_weights_based_on_dist(z_in, z_drift, z_target):
    '''
        refer LIME's code to assign sample weights
        use an exponential kernel
        weight = e^(-D(z_syn, z_target)^2 / sigma^2),
        sigma is called the kernel's width, if not specified, use sqrt(#column) * 0.75

        whether it's in-dist or drift, all assigned weights based on their distance to the target drift sample.
    '''
    z_all = np.vstack((z_in, z_drift))
    distances = pairwise_distances(z_all, z_target.reshape(1, -1), metric='euclidean').ravel()

    kernel_width = float(np.sqrt(z_all.shape[1]) * .75)

    kernel_fn = partial(kernel, kernel_width=kernel_width)  # partial: wrap the original function to have fewer arguments
    weights = kernel_fn(distances)

    # logging.debug(f'z distances in: {distances[:len(z_in)]}')
    # logging.debug(f'z distances drift: {list(distances[len(z_in):])}')
    # logging.debug(f'z weights in: {weights[:len(z_in)]}')
    # logging.debug(f'z weights drift: {list(weights[len(z_in):])}')

    return weights


def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def build_target_classifier(z_in, z_drift, y_in, y_drift,
                            dropout_ratio, z_weights, mlp_dims, model_save_path):
    mlp_classifier = classifier.MLPClassifier(dims=mlp_dims,
                                              model_save_name=model_save_path,
                                              dropout=dropout_ratio,
                                              verbose=0) # no logs

    logging.debug(f'Saving explanation MLP models to {model_save_path}...')
    retrain_flag = 1 if not os.path.exists(model_save_path) else 0

    X = np.vstack((z_in, z_drift))
    y = np.hstack((y_in, y_drift))

    logging.debug(f'X.shape: {X.shape}')
    logging.debug(f'y.shape: {y.shape}')

    epochs = 30

    val_acc = mlp_classifier.train(X, y,
                                   lr=0.01,
                                   batch_size=32,
                                   epochs=epochs,
                                   loss='binary_crossentropy',
                                   class_weight=None,
                                   sample_weight=z_weights,
                                   train_val_split=False,  # do not split train and val, predict on all the training set
                                   retrain=retrain_flag)
    K.clear_session() # to prevent load_model becomes slower and slower
    clf = load_model(model_save_path)
    logging.debug(f'[build_target_classifier] prediction in: {list(np.argmax(clf.predict(X[0:len(z_in)]), axis=1))}')
    logging.debug(f'[build_target_classifier] prediction drift: {list(np.argmax(clf.predict(X[len(z_in):]), axis=1))}')

    y_pred = clf.predict(X)
    logging.debug(f'y_pred shape: {y_pred.shape}')
    y_pred = np.argmax(clf.predict(X), axis=1)
    logging.info(f'clf predict accuracy: {accuracy_score(y, y_pred)}')


def combine_encoder_and_approximation_model(cae_dims, mlp_dims, dropout_ratio,
                                            cae_weights_path, approximation_mlp_model_path, final_model_save_path):
    act = 'relu'
    init = 'glorot_uniform'
    n_stacks = len(cae_dims) - 1

    input_ = Input(shape=(cae_dims[0],), name='input')
    x = input_
    for i in range(n_stacks-1):
        x = Dense(cae_dims[i + 1], activation=act,
                    kernel_initializer=init, name='encoder_%d' % i)(x)
    encoded = Dense(cae_dims[-1], kernel_initializer=init,
                    name='encoder_%d' % (n_stacks - 1))(x)

    clf_stacks = len(mlp_dims) - 1
    x2 = encoded
    for i in range(clf_stacks - 1):
        x2 = Dense(mlp_dims[i + 1], activation='relu', name='clf_%d' % i)(x2)
        if dropout_ratio > 0:
            x2 = Dropout(dropout_ratio, seed=42)(x2)
    data = Dense(mlp_dims[-1], activation='softmax',
                 name='clf_%d' % (clf_stacks - 1))(x2)

    final_model = Model(inputs=input_, outputs=data)
    final_model.load_weights(cae_weights_path, by_name=True)
    final_model.load_weights(approximation_mlp_model_path, by_name=True)
    final_model.save(final_model_save_path)


def explain_instance(x, lambda_1, diff_idx, final_model_path):
    OPTIMIZER = tf.train.AdamOptimizer
    INITIALIZER = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
    LR = 1e-2  # learning rate
    REGULARIZER = 'elasticnet' # a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.
    EXP_EPOCH = 250
    EXP_DISPLAY_INTERVAL = 10  # print middle result every k epochs
    EXP_LAMBDA_PATIENCE = 20
    EARLY_STOP_PATIENCE = 250

    MASK_SHAPE = (x.shape[0],)
    M1 = np.zeros(shape=(x.shape[0], ), dtype=np.float32)
    for i in diff_idx:
        M1[i] = 1
    logging.debug(f'MASK_SHAPE: {MASK_SHAPE}')

    K.clear_session()
    model = load_model(final_model_path)
    y = model.predict(x.reshape(1, -1))
    logging.debug(f'[explain_instance] y original: {y}')

    if np.argmax(y) != 1: # don't explain wrongly classified target drift samples
        mask_best = None
        logging.error(f'[explain_instance] y is predicted as 0')
    else:
        mask_best = None
        exp_test = mask_exp.OptimizeExp(input_shape=x.shape,
                                        mask_shape=MASK_SHAPE,
                                        model=model,
                                        num_class=2,
                                        optimizer=OPTIMIZER,
                                        initializer=INITIALIZER,
                                        lr=LR,
                                        regularizer=REGULARIZER,
                                        model_file=final_model_path)

        mask_best = exp_test.fit_local(X=x,
                                       y=y,
                                       epochs=EXP_EPOCH,
                                       lambda_1=lambda_1,
                                       display_interval=EXP_DISPLAY_INTERVAL,
                                       lambda_patience=EXP_LAMBDA_PATIENCE,
                                       early_stop_patience=EARLY_STOP_PATIENCE)
    return mask_best

