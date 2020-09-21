"""
explain_by_distance.py
~~~~~~~

Functions for explaining why a sample is an drift (distance-based).
Perturb the testing set drift sample to be closer to the centroid of its closest family.
If flipping a feature can make the drift sample be closer to the centroid, then we think it's important.
In this case, we can't rank the importance of a feature, just two cases: important or unimportant.

Two design options:
1. use mask only, use mask_exp_by_distance_mask_only.py
2. use mask * m1, use mask_exp_by_distance_mask_m1.py

"""

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import traceback
import logging
import re
import numpy as np
import tensorflow as tf

from timeit import default_timer as timer
from tqdm import tqdm
from keras import backend as K

import cade.utils as utils
from cade.autoencoder import Autoencoder


def explain_drift_samples_per_instance(X_train, y_train, X_test, y_test,
                                       args,
                                       one_by_one_check_result_path,
                                       training_info_for_detect_path,
                                       cae_weights_path,
                                       mask_file_path):
    if os.path.exists(mask_file_path):
        logging.info(f'explanation result file {mask_file_path} exists, no need to run explanation module')
    else:
        drift_samples_idx_list, drift_samples_real_labels, \
            drift_samples_closest = get_drift_samples_to_explain(one_by_one_check_result_path)

        mad_threshold = args.mad_threshold

        cae_dims = utils.get_model_dims('Contrastive AE', X_train.shape[1],
                                        args.cae_hidden, len(np.unique(y_train)))

        family_centroid_dict = {}
        for family in np.unique(drift_samples_closest):
            z_train, z_closest_family, centroid, \
                dis_to_centroid, mad = load_training_info(training_info_for_detect_path, family)
            distance_lowerbound = mad * mad_threshold + np.median(dis_to_centroid)
            dis_to_centroid_inds = np.array(dis_to_centroid).argsort() # distance ascending
            X_train_family = X_train[np.where(y_train == family)[0]]
            closest_to_centroid_sample = X_train_family[dis_to_centroid_inds][0]
            logging.debug(f'family-{family} closest distance to centroid: {np.min(dis_to_centroid)}')

            family_centroid_dict[family] = [centroid, distance_lowerbound, closest_to_centroid_sample]

        masks = []
        X_drift_list = []
        for idx, sample_idx in tqdm(enumerate(drift_samples_idx_list),
                                    total=len(drift_samples_idx_list),
                                    desc='explain drift'):
            try:

                x_target = X_test[sample_idx]
                X_drift_list.append(x_target)
                closest_family = drift_samples_closest[idx]

                [centroid, distance_lowerbound, closest_to_centroid_sample] = family_centroid_dict[closest_family]

                diff = x_target - closest_to_centroid_sample
                diff_idx = np.where(diff != 0)[0]

                mask = explain_instance(x_target, args.exp_method, diff_idx, centroid, closest_to_centroid_sample,
                                        distance_lowerbound, args.exp_lambda_1, cae_dims, cae_weights_path)
                masks.append(mask)
            except:
                masks.append(None)
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
    logging.debug(f'dis_to_centroid median: {np.median(dis_to_centroid)}')
    logging.debug(f'mad-{closest_family}: {mad}')

    return z_train, z_closest_family, centroid, dis_to_centroid, mad


def explain_instance(x, exp_method, diff_idx, centroid, closest_to_centroid_sample,
                     distance_lowerbound, lambda_1, cae_dims, cae_weights_path):
    OPTIMIZER = tf.train.AdamOptimizer
    INITIALIZER = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
    LR = 1e-2  # learning rate
    REGULARIZER = 'elasticnet' # a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.
    EXP_EPOCH = 250
    EXP_DISPLAY_INTERVAL = 10  # print middle result every k epochs
    EXP_LAMBDA_PATIENCE = 20
    EARLY_STOP_PATIENCE = 10
    USE_GUMBLE_TRICK = True

    MASK_SHAPE = (x.shape[0],)
    latent_dim = cae_dims[-1]

    TEMP = 0.1
    M1 = np.zeros(shape=(x.shape[0], ), dtype=np.float32)
    for i in diff_idx:
        M1[i] = 1

    logging.debug(f'MASK_SHAPE: {MASK_SHAPE}')
    logging.debug(f'latent_dim: {latent_dim}')
    logging.debug(f'distance lowerbound: {distance_lowerbound}')
    logging.debug(f'epoch: {EXP_EPOCH}')
    logging.debug(f'temperature: {TEMP}')
    logging.debug(f'use gumble trick: {USE_GUMBLE_TRICK}')

    mask_best = None
    if exp_method == 'distance_mm1':
        import cade.mask_exp_by_distance_mask_m1 as mask_exp

        K.clear_session()
        model = load_encoder(cae_dims, cae_weights_path)

        exp_test = mask_exp.OptimizeExp(batch_size=10,
                                        mask_shape=MASK_SHAPE,
                                        latent_dim=latent_dim,
                                        model=model,
                                        optimizer=OPTIMIZER,
                                        initializer=INITIALIZER,
                                        lr=LR,
                                        regularizer=REGULARIZER,
                                        temp=TEMP,
                                        normalize_choice='clip',
                                        use_concrete=USE_GUMBLE_TRICK,
                                        model_file=cae_weights_path)

        mask_best = exp_test.fit_local(X=x,
                                        m1=M1,
                                        centroid=centroid,
                                        closest_to_centroid_sample=closest_to_centroid_sample,
                                        num_sync=50,
                                        num_changed_fea=1,
                                        epochs=EXP_EPOCH,
                                        lambda_1=lambda_1,
                                        display_interval=EXP_DISPLAY_INTERVAL,
                                        exp_loss_lowerbound=distance_lowerbound,
                                        lambda_patience=EXP_LAMBDA_PATIENCE,
                                        early_stop_patience=EARLY_STOP_PATIENCE)

        logging.debug(f'M1 * mask == 1: {np.where(M1 * mask_best == 1)[0]}')

        if mask_best is not None:
            return M1 * mask_best
    return None
