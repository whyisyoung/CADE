'''
Use different explanation methods to find important features for each drifting sample,
flip/modify the found important features, compare the distance (avg +/- std) between the perturbed sample's latent representation and the centroid.

command to run:
drebin:
    python -u evaluate_explanation_by_distance.py drebin_new_7 distance_mm1 0.001 1 0.1
    python -u evaluate_explanation_by_distance.py drebin_new_7 approximation_loose 0.001 0 0.1
    nohup python -u evaluate_explanation_by_distance.py drebin_new_7 gradient 0.001 0 0.1 > logs/nohup-drebin_new_7-gradient-exp.log &
    nohup python -u evaluate_explanation_by_distance.py drebin_new_7 random 0.001 0 0.1 > logs/nohup-drebin_new_7-random-100-exp.log &

IDS:
    nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration distance_mm1 0.001 1 0.1 > logs/nohup-IDS-distance-mm1-exp.log &
    python -u evaluate_explanation_by_distance.py IDS_new_Infilteration approximation_loose 0.001 0 0.1
    nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration gradient 0.001 0 0.1 > logs/nohup-IDS-gradient-exp.log &
    random 100 times: nohup python -u evaluate_explanation_by_distance.py IDS_new_Infilteration random 0.001 0 0.1 > logs/nohup-IDS-random-exp.log &
'''

import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import os, sys
import numpy as np
import statistics
import traceback
import logging
from timeit import default_timer as timer
from keras import backend as K
from tqdm import tqdm
import tensorflow as tf

import cade.data as data
import cade.utils as utils
import cade.explain_by_distance as explain_dis
from cade.logger import init_log
from cade.autoencoder import Autoencoder

families = [
    'FakeInstaller',
    'DroidKungFu',
    'Plankton',
    'GinMaster',
    'BaseBridge',
    'Iconosys',
    'Kmin',
    'FakeDoc'
]

RANDOM_TRY = 100


def load_necessary_model_and_data(X_train, dataset, lambda_1, exp_method):
    if 'drebin' in dataset:
        cae_dims = [X_train.shape[1], 512, 128, 32, 7]
        cae_weights_path = f'models/{dataset}/cae_{X_train.shape[1]}-512-128-32-7_lr0.0001_b64_e250_m10.0_lambda0.1_weights.h5'
        feature_file = f'data/{dataset}/drebin_new{new_label}_train_selected_features.txt'
    elif 'IDS' in dataset:
        cae_dims = [X_train.shape[1], 64, 32, 16, 3]
        cae_weights_path = f'models/{dataset}/cae_83-64-32-16-3_lr0.0001_b512_e250_m10.0_lambda0.1_weights.h5'
        feature_file = f'data/IDS_83_features.txt'
    elif 'bluehex' in dataset:
        cae_dims = [X_train.shape[1], 1024, 256, 64, 5]
        cae_weights_path = f'models/{dataset}/cae_1857-1024-256-64-5_lr0.0001_b256_e250_m10.0_weights.h5'
        feature_file = '/home/liminyang/bluehex/cade_feature_names_setting5.txt' # setting 5 for option 6
    else:
        sys.exit(-1)

    K.clear_session()  # be careful with this it may clean up previous loaded models.
    ae = Autoencoder(cae_dims)
    ae_model, encoder_model = ae.build()
    encoder_model.load_weights(cae_weights_path, by_name=True)
    if 'approximation' in exp_method or 'distance' in exp_method:
        mask_list = np.load(f'reports/{dataset}/mask_{exp_method}_{lambda_1}.npz')['masks']
    else:
        mask_list = None

    features = []
    if feature_file is not None:
        with open(feature_file, 'r') as fin:
            for line in fin:
                features.append(line.strip())

    return mask_list, encoder_model, features, cae_dims, cae_weights_path


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

    return z_train, z_closest_family, centroid, dis_to_centroid, mad


def get_important_fea_and_distance(dataset, X_test, y_test, drift_samples_idx_list,
                                   drift_samples_real_labels, drift_samples_closest,
                                   family_info_dict, X_train_family_dict, closest_sample_family_dict,
                                   mask_list, encoder_model,
                                   exp_method, features, use_gumbel,
                                   save_distance_mm1_important_fea_len_file, save_result_path):

    if exp_method != 'distance_mm1':
        if os.path.exists(save_distance_mm1_important_fea_len_file):
            important_feas_len_list = read_feas_len_from_file(save_distance_mm1_important_fea_len_file)
            print(f'load important_feas_len_list len: {len(important_feas_len_list)}')
        else:
            logging.error('you need to perform distance_mm1 method to get the length of important features first')
            sys.exit(1)
    else:
        important_feas_len_list = []

    success = 0 # number of sucessful perturbations from drift to in-distribution.

    lowerbound_list = []
    logging.debug(f'len(drift_samples_idx_list): {len(drift_samples_idx_list)}')

    for idx, sample_idx, real, closest_family in tqdm(zip(range(len(drift_samples_idx_list)),
                                                          drift_samples_idx_list,
                                                          drift_samples_real_labels,
                                                          drift_samples_closest), total=len(drift_samples_idx_list)):
        x = X_test[sample_idx]
        mask = mask_list[idx]

        lowerbound = family_info_dict[closest_family][2]

        lowerbound_list.append(lowerbound)

        if idx == 0:
            X_arr = np.copy(x)
            Centroid_arr = np.copy(family_info_dict[closest_family][0])
        else:
            X_arr = np.vstack((X_arr, x))
            Centroid_arr = np.vstack((Centroid_arr, family_info_dict[closest_family][0]))

        if 'approximation' in exp_method:
            tmp = np.sum(mask)
            if not np.isnan(tmp):
                prod = x * mask
                ranked_prod_value = np.sort(prod, kind='mergesort', axis=None)[::-1]
                valid_n = len(np.where(prod > 0)[0])
                valid_n = min(valid_n, important_feas_len_list[idx])

                ranked_prod_idx = np.argsort(prod, kind='mergesort', axis=None)[::-1]
                important_feas = ranked_prod_idx[:valid_n+1]
            else:
                logging.debug(f'drift-{idx}: mask is None')
                important_feas = None
        elif exp_method == 'distance_mm1':
            if mask is not None:
                if use_gumbel:
                    # only when m = m1 = 1, it's important, we could also rank the rest of the features,
                    # but we keep m = m1 = 1 for simplicity (less features).
                    important_feas = np.where(mask == 1)[0]
                else:
                    ranked_mask = np.sort(mask, kind='mergesort', axis=None)[::-1] # bigger means more important feature
                    valid_n = len(np.where(mask == 1)[0])
                    ranked_mask_idx = np.argsort(mask, kind='mergesort', axis=None)[::-1]
                    important_feas = ranked_mask_idx[:valid_n+1]
            else:
                logging.debug(f'drift-{idx}: mask is None')
                important_feas = None

        X_test_family = X_test[np.where(y_test == closest_family)[0]]
        X_closest_family_all = np.vstack((X_train_family_dict[closest_family], X_test_family))
        X_real_family = X_test[np.where(y_test == real)[0]]

        if important_feas is not None:
            if exp_method == 'distance_mm1':
                important_feas_len_list.append(len(important_feas))

                # case study
                if 'drebin' in dataset:
                    cases = [1, 2] # idx-2: closest to Gin Master, idx-1: closest to DroidKungfu (most FakeDoc closer to DroidKungfu).
                elif 'IDS' in dataset:
                    cases = range(5) # the first 5 cases are closer to SSH, SSH, Hulk, Hulk, Hulk
                elif 'bluehex' in dataset:
                    cases = range(5)
                if idx in cases:
                    utils.create_folder('reports/explanation_case_study/')
                    with open(f'reports/explanation_case_study/{dataset}-{exp_method}-drifting-{idx}-temp-0.1.txt', 'w') as f:
                        f.write(f'feature index,sample {idx} important feature,original value,avg value in testing set(real family),avg value in training set(closest family),avg value in both train and test set(closest family),closest sample value\n')
                        for fea in important_feas:
                            f.write(f'{fea},{features[fea]},{x[fea]:e},{np.mean(X_real_family[:, fea]):e},' + \
                                    f'{np.mean(X_train_family_dict[closest_family][:, fea]):e},' + \
                                    f'{np.mean(X_closest_family_all[:, fea]):e},{closest_sample_family_dict[closest_family][fea]:e}\n')

            ''' the chosen method: perturb the important features and craft a new sample '''
            x_new = np.copy(x)
            for i in important_feas:
                ''' NOTE: flip important features:
                    for baseline 2: important features all have a feature value = 1, so there is only 1 -> 0.
                    for distance based methods: both 1-> 0 and 0->1 are possible'''
                if 'drebin' in dataset:
                    x_new[i] = 1 if x[i] == 0 else 0
                elif 'IDS' in dataset:
                    ''' use the sample (closest to centroid) feature value'''
                    perturbed_value = closest_sample_family_dict[closest_family][i]

                    x_new[i] = perturbed_value
                elif 'bluehex' in dataset:
                    perturbed_value = closest_sample_family_dict[closest_family][i]
                    x_new[i] = perturbed_value

            if idx == 0:
                X_perturb_arr = np.copy(x_new)
            else:
                X_perturb_arr = np.vstack((X_perturb_arr, x_new))

    latent_x = encoder_model.predict(X_arr)
    latent_x_perturb = encoder_model.predict(X_perturb_arr)
    original_dis = np.sqrt(np.sum(np.square(latent_x - Centroid_arr), axis=1))
    perturbed_dis = np.sqrt(np.sum(np.square(latent_x_perturb - Centroid_arr), axis=1))

    success_idx = np.where((perturbed_dis <= lowerbound_list) == True)[0]
    success = len(success_idx)


    if exp_method == 'distance_mm1':
        write_result_to_file(original_dis, 'original distance', save_result_path, 'w')
        write_result_to_file(important_feas_len_list, f'{exp_method} important feas len', save_result_path, 'a')

    write_result_to_file(perturbed_dis, f'{exp_method} perturbed distance', save_result_path, 'a')
    with open(save_result_path, 'a') as f:
        ratio = success / len(perturbed_dis)
        f.write(f'{exp_method} success idx: {success_idx}\n\n')
        print(f'{exp_method} success from drifting to in-dist: {success}, ratio: {ratio * 100:.2f}%')
        f.write(f'{exp_method} success from drifting to in-dist: {success}, ratio: {ratio * 100:.2f}%\n')

    with open(save_distance_mm1_important_fea_len_file, 'w') as f:
        logging.debug(f'important_feas_len_list len: {len(important_feas_len_list)}')
        for fea_len in important_feas_len_list:
            f.write(f'{fea_len}\n')


def preprocess_training_info(X_train, y_train, drift_samples_closest, training_info_for_detect_path):
    family_info_dict = {}
    X_train_family_dict = {}
    closest_sample_family_dict = {}

    # the load_training_info() is actually very time consuming, so just load it once for each closest family here.
    for family in np.unique(drift_samples_closest):
        z_train, z_closest_family, centroid, \
            dis_to_centroid, mad = load_training_info(training_info_for_detect_path, family)
        lowerbound = mad * 3.5 + np.median(dis_to_centroid)
        dis_to_centroid_inds = np.array(dis_to_centroid).argsort() # distance ascending
        X_train_family = X_train[np.where(y_train == family)[0]]
        closest_to_centroid_sample = X_train_family[dis_to_centroid_inds][0]

        family_info_dict[family] = [centroid, mad, lowerbound]
        X_train_family_dict[family] = X_train_family
        closest_sample_family_dict[family] = closest_to_centroid_sample

    return family_info_dict, X_train_family_dict, closest_sample_family_dict


def write_result_to_file(single_list, name, filepath, mode):
    with open(filepath, mode) as f:
        try:
            avg = np.average(single_list)
            std = np.std(single_list)
            result = f'{name}  avg: {avg:.3f}, std: {std:.3f}'
            print(result)
            f.write(result + '\n')
            f.write('=' * 80 + '\n')
        except:
            logging.error(f'{name} error\n {traceback.format_exc()}')


def get_backpropagation_important_features(dataset, X_train, X_test, y_train, y_test, drift_samples_idx_list,
                                           drift_samples_closest, family_info_dict, encoder_model, cae_dims,
                                           closest_sample_family_dict,
                                           features, cae_weights_path, important_feas_len_list, save_result_path):
    ''' G = d(f(x) - f(c)) / dx, sum G over rows (or maybe columns), then rank G to get the feature importance ranking'''
    lowerbound_list = []
    s = timer()

    ''' construct the tf nodes to calculate the gradients.
        the tensors should be put outside the for loop so that we only add these nodes
        to the graph once instead of multiple times, the latter would make the graph bigger and bigger and get slower'''
    input_tensor = encoder_model.get_input_at(0)
    centroid_tensor = tf.placeholder(tf.float32, shape=(None, cae_dims[-1]))
    latent_input = encoder_model(input_tensor)
    g = tf.gradients((latent_input - centroid_tensor), input_tensor)

    gradient_valid_important_feas_len_list = []
    for idx, sample_idx, family in tqdm(zip(range(len(drift_samples_idx_list)), drift_samples_idx_list, drift_samples_closest),
                                        total=len(drift_samples_idx_list)):

        x = X_test[sample_idx]
        centroid = family_info_dict[family][0]
        lowerbound_list.append(family_info_dict[family][2])

        start = timer()
        # original_importance could be positive or negative
        important_feas_idx, abs_importance, \
            original_importance = backpropagation_gradients(idx, x, centroid, encoder_model,
                                                            cae_weights_path, features,
                                                            input_tensor, centroid_tensor, g)
        end = timer()
        logging.debug(f'{idx} - backpropagation_gradients time: {(end - start):.3f}s')

        distance_method_important_feas_len = important_feas_len_list[idx]

        x_new = np.copy(x)
        valid_n = 0
        for i in important_feas_idx:
            if valid_n < distance_method_important_feas_len:
                if 'drebin' in dataset:
                    if original_importance[i] > 0:
                        x_new[i] = 1 if x[i] == 0 else 0
                        valid_n += 1
                elif 'IDS' in dataset:
                    ''' use the sample (closest to centroid) feature value'''
                    if original_importance[i] > 0:
                        perturbed_value = closest_sample_family_dict[family][i]
                        valid_n += 1

        gradient_valid_important_feas_len_list.append(valid_n)

        if idx == 0:
            X_perturb_arr = np.copy(x_new)
            Centroid_arr = np.copy(family_info_dict[family][0])
        else:
            X_perturb_arr = np.vstack((X_perturb_arr, x_new))
            Centroid_arr = np.vstack((Centroid_arr, family_info_dict[family][0]))

    encoder_model.load_weights(cae_weights_path, by_name=True)
    latent_x_perturb = encoder_model.predict(X_perturb_arr)
    perturbed_dis = np.sqrt(np.sum(np.square(latent_x_perturb - Centroid_arr), axis=1))
    success = len(np.where((perturbed_dis <= lowerbound_list) == True)[0])
    write_result_to_file(perturbed_dis, 'gradient perturbed distance', save_result_path, 'a')
    write_result_to_file(gradient_valid_important_feas_len_list, 'gradient valid important features len', save_result_path, 'a')
    with open(save_result_path, 'a') as f:
        ratio = success / len(perturbed_dis)
        f.write(f'baseline 2: gradient success from drifting to in-dist: {success}, ratio: {ratio * 100:.2f}%\n')

    e = timer()
    logging.debug(f'get_backpropagation_important_features time: {(e - s):.3f}s')


def backpropagation_gradients(idx, x, centroid, model, model_file, features,
                              input_tensor, centroid_tensor, g):

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        model.load_weights(model_file, by_name=True)

        feed_dict = {input_tensor: x[None], centroid_tensor: centroid[None,]}
        g_matrix = sess.run(g, feed_dict=feed_dict)[0]

        # rank by importance descending order, the output importance could be negative,
        # so we rank by their absolute value.
        g_matrix = g_matrix.reshape(-1,)
        ordered_g_abs = np.sort(np.abs(g_matrix))[::-1]

        ordered_g_abs_index = np.argsort(np.abs(g_matrix))[::-1]
        if idx == 2:
            logging.critical(f'ordered g: {list(ordered_g_abs)}')
            important_feas_list = []
            logging.debug(f'backpropagation important features for drifting-{idx}: \n####################################')
            for i in range(50):
                logging.debug(f'{features[ordered_g_abs_index[i]]}')
                important_feas_list.append(features[ordered_g_abs_index[i]])
            logging.debug('####################################')

        return ordered_g_abs_index, ordered_g_abs, g_matrix


def read_feas_len_from_file(save_distance_mm1_important_fea_len_file):
    important_feas_len_list = []
    with open(save_distance_mm1_important_fea_len_file, 'r') as f:
        for line in f:
            important_feas_len_list.append(int(line.strip()))
    return important_feas_len_list


def eval_random_select_important_feas(dataset, save_distance_mm1_important_fea_len_file,
                                      drift_samples_idx_list, drift_samples_closest,
                                      X_test, y_test, family_info_dict, closest_sample_family_dict,
                                      encoder_model, save_result_path):
    ''' baseline 3: randomly choose the same number of important features and craft a new sample'''
    if os.path.exists(save_distance_mm1_important_fea_len_file):
        s = timer()
        important_feas_len_list = read_feas_len_from_file(save_distance_mm1_important_fea_len_file)
        random_dis_array_list = []
        total_success_random = 0
        for random_cnt in tqdm(range(RANDOM_TRY)):
            lowerbound_list = []
            for idx, sample_idx, family in zip(range(len(drift_samples_idx_list)), drift_samples_idx_list, drift_samples_closest):
                x = X_test[sample_idx]

                lowerbound_list.append(family_info_dict[family][2])

                fea_len = important_feas_len_list[idx]

                x_random = np.copy(x)
                random_important_feas = np.random.choice(x.shape[0], size=fea_len, replace=False)
                for i in random_important_feas:
                    if 'drebin' in dataset:
                        x_random[i] = 1 if x[i] == 0 else 0
                    elif 'IDS' in dataset:
                        perturbed_value = closest_sample_family_dict[family][i]
                        x_random[i] = perturbed_value
                if idx == 0:
                    X_random_arr = np.copy(x_random)
                    Centroid_arr = np.copy(family_info_dict[family][0])
                else:
                    X_random_arr = np.vstack((X_random_arr, x_random))
                    Centroid_arr = np.vstack((Centroid_arr, family_info_dict[family][0]))

            latent_x_random = encoder_model.predict(X_random_arr)
            random_dis = np.sqrt(np.sum(np.square(latent_x_random - Centroid_arr), axis=1))

            success_random = len(np.where((random_dis <= lowerbound_list) == True)[0])
            total_success_random += success_random
            random_dis_array_list.append(random_dis)

        write_result_to_file(random_dis_array_list, f'random perturbed distance (n = {RANDOM_TRY})', save_result_path, 'w')

        with open(save_result_path, 'a') as f:
            total_try = len(random_dis_array_list) * len(random_dis_array_list[0])
            random_ratio = total_success_random / total_try
            print(f'random success perturbed from drifting to in-dist: {random_ratio * 100:.2f}')
            f.write(f'random success from drifting to in-dist: {total_success_random}, total_try: {total_try}, \
                     ratio: {random_ratio * 100:.2f}%\n')

        e = timer()
        logging.debug(f'eval_random_select_important_feas: {(e - s):.3f} seconds')

    else:
        logging.error('you need to perform distance_mm1 method to get the length of important features first')
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        logging.error(f'usage example: python -u evaluate_explanation_by_distance.py drebin_new_7 distance_mm1 0.001 1 0.1')
        sys.exit(-1)

    dataset = sys.argv[1]  # drebin_new_7 or IDS_new_Infilteration
    exp_method = sys.argv[2]  # distance_mm1, approximation_loose, random, gradient
    # lambda_1 for baseline methods needs to be the same as distance_mm1 to keep the same important features length
    lambda_1 = float(sys.argv[3])  # 0.001
    use_gumbel = int(sys.argv[4]) # 1 or 0, use gumbel when distance_mm1
    temp = float(sys.argv[5]) # temp for baseline methods needs to be the same as distance_mm1

    REPORT_FOLDER = f'reports/exp_evaluation/{dataset}'
    utils.create_folder(REPORT_FOLDER)
    LOG_FOLDER = f'logs/exp_evaluation/{dataset}'
    utils.create_folder(LOG_FOLDER)

    log_path = f'./{LOG_FOLDER}/{dataset}-{exp_method}-lambda-{lambda_1}-gumble-{use_gumbel}'
    if os.path.exists(log_path + '.log'):
        os.remove(log_path + '.log')
        logging.info('log file removed')

    init_log(log_path, level=logging.INFO)

    if use_gumbel == 1:
        gumble_flag = 'with'
    else:
        gumble_flag = 'without'

    save_result_path = f'{REPORT_FOLDER}/{dataset}-{exp_method}-lambda-{lambda_1}-temp-{temp}-{gumble_flag}-gumble.txt'
    # other explanation method would load this file to determine how many important features to pick
    save_distance_mm1_important_fea_len_file = os.path.join(REPORT_FOLDER, f'{dataset}-distance-mm1-important-feas-len-temp-{temp}-lambda-{lambda_1}.txt')

    if 'drebin' in dataset:
        new_label = 7
    elif 'IDS' in dataset:
        new_label = 3
    elif 'bluehex_top_5' in dataset:
        new_label = 5
    else:
        logging.error(f'dataset {dataset} not supported')
        sys.exit(-1)

    one_by_one_check_result_path = f'reports/{dataset}/dist_mlp_one_by_one_check_pr_value_m10.0_mad3.5_lambda0.1.csv'

    X_train, y_train, X_test, y_test = data.load_features(dataset, new_label)

    drift_samples_idx_list, drift_samples_real_labels, \
        drift_samples_closest = explain_dis.get_drift_samples_to_explain(one_by_one_check_result_path)

    training_info_for_detect_path = os.path.join('reports', dataset, 'intermediate', f'mlp_training_info_for_detect_m10.0_lambda0.1.npz')
    family_info_dict, X_train_family_dict, \
            closest_sample_family_dict = preprocess_training_info(X_train, y_train,
                                                                  drift_samples_closest, training_info_for_detect_path)

    s1 = timer()
    mask_list, encoder_model, features, \
            cae_dims, cae_weights_path = load_necessary_model_and_data(X_train, dataset, lambda_1, exp_method)
    e1 = timer()
    logging.debug(f'load_necessary_model_and_data time: {(e1 - s1):.2f}')

    '''main logic'''
    if 'approximation' in exp_method or 'distance' in exp_method:
        s2 = timer()
        get_important_fea_and_distance(dataset, X_test, y_test,
                                        drift_samples_idx_list,
                                        drift_samples_real_labels,
                                        drift_samples_closest,
                                        family_info_dict,
                                        X_train_family_dict,
                                        closest_sample_family_dict,
                                        mask_list, encoder_model,
                                        exp_method, features, use_gumbel,
                                        save_distance_mm1_important_fea_len_file,
                                        save_result_path)
        e2 = timer()
        logging.debug(f'get_important_fea_and_distance time: {(e2 - s2):.2f}')
    elif exp_method == 'gradient':
            ''' try Dr. Xing's mathematical baseline: backpropagate the gradients from low-d to high-d and get feature importance. '''
            if os.path.exists(save_distance_mm1_important_fea_len_file):
                important_feas_len_list = read_feas_len_from_file(save_distance_mm1_important_fea_len_file)
                get_backpropagation_important_features(dataset, X_train, X_test, y_train, y_test, drift_samples_idx_list,
                                                drift_samples_closest, family_info_dict, encoder_model, cae_dims,
                                                closest_sample_family_dict,
                                                features, cae_weights_path, important_feas_len_list, save_result_path)
            else:
                logging.error('you need to perform distance_mm1 method to get the length of important features first')
                sys.exit(1)
    elif exp_method == 'random':
            eval_random_select_important_feas(dataset, save_distance_mm1_important_fea_len_file,
                                              drift_samples_idx_list, drift_samples_closest,
                                              X_test, y_test, family_info_dict,
                                              closest_sample_family_dict, encoder_model, save_result_path)

    else:
        logging.error(f'explanation method {exp_method} not supported')
        sys.exit(-1)

