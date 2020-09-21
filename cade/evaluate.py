"""
evaluate.py
~~~~~~~

Functions for evaluating drifting detection and report classification results.

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
import logging
import copy
import traceback
import numpy as np
import pickle

from collections import Counter, OrderedDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from keras import backend as K
from keras.models import load_model

import matplotlib.pyplot as plt

import cade.utils as utils


def report_classification_results(model_path,
                                  X_new, y_new,
                                  classify_results_all_path,
                                  classify_results_simple_path):
    """Report wrongly classified samples and probabilities for classification model.

    Arguments:
        model_path {str} -- file path for the target MLP model.
        X_new {np.ndarray} -- feature vectors of new data samples.
        y_new {np.ndarray} -- groundtruth label of new data samples.
        classify_results_all_path {str} -- file path for saving the wrongly classified samples and probabilities.
        classify_results_simple_path {str} -- file path for saving all the new samples prediction, real, prob.
    """
    report_classification_results_helper(model_path,
                                         X_new, y_new,
                                         classify_results_all_path,
                                         only_wrongly_samples=False)
    report_classification_results_helper(model_path,
                                         X_new, y_new,
                                         classify_results_simple_path,
                                         only_wrongly_samples=True)


def report_classification_results_helper(model_path,
                                         X_new, y_new,
                                         report_file_path,
                                         only_wrongly_samples):
    if 'h5' in model_path:
        K.clear_session()
        clf_model = load_model(model_path)
        preds = clf_model.predict(X_new)
        y_new_pred = np.argmax(preds, axis=1)
        y_new_prob = np.max(preds, axis=1)
    elif 'pkl' in model_path:
        with open(model_path, 'rb') as f:
            clf_model = pickle.load(f)
        y_new_pred = clf_model.predict(X_new)
        y_new_prob = np.max(clf_model.predict_proba(X_new), axis=1)
    else:
        logging.error(f'saved model name {model_path} is neither h5 or pkl format')
        sys.exit(-1)

    utils.create_parent_folder(report_file_path)
    with open(report_file_path, 'w') as f:
        f.write('sample_idx,real_label,pred_label,pred_prob\n')
        for idx, real_label in tqdm(enumerate(y_new), desc='MLP classified'):
            if only_wrongly_samples:
                if y_new_pred[idx] != real_label:
                    f.write(f'{idx},{real_label},{y_new_pred[idx]},{y_new_prob[idx]}\n')
            else:
                f.write(f'{idx},{real_label},{y_new_pred[idx]},{y_new_prob[idx]}\n')
    if only_wrongly_samples:
        logging.info('Reported wrongly classified samples.')
    else:
        logging.info('Reported the classification for all new samples.')


def combine_classify_and_detect_result(classify_results_all_path, detect_results_all_path, combined_report_path):
    '''combine classification and detect results as a final result'''
    if not os.path.exists(combined_report_path):
        with open(combined_report_path, 'w') as fout:
            fout.write(f'sample_idx,real_label,pred_label,closest_label,is_drift,pred_prob,min_distance,min_anomaly_score\n')
            with open(classify_results_all_path, 'r') as fin1:
                next(fin1)
                with open(detect_results_all_path, 'r') as fin2:
                    next(fin2)
                    for line1, line2 in zip(fin1, fin2):
                        idx, real, pred, pred_prob = line1.strip().split(',')
                        idx, is_drift, closest, real, pred, min_dis, min_score = line2.strip().split(',')
                        fout.write(f'{idx},{real},{pred},{closest},{is_drift},{pred_prob},{min_dis},{min_score}\n')


def evaluate_newfamily_as_drift_by_distance(dataset_name, newfamily, combined_report_path, mad_threshold,
                                            save_ordered_dis_path, dist_effort_pr_value_fig_path,
                                            dist_one_by_one_check_result_path):
    if 'drebin' in dataset_name:
        newfamily = 7  # since we adjust all the new family to label 7, no matter it is 0~7.
    elif 'IDS' in dataset_name:
        newfamily = 3

    total_new_family = 0
    sample_result_dict = {}
    y_closest = []
    y_real = []
    y_pred = []
    with open(combined_report_path, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            sample_idx, real, pred, closest, is_drift, prob, min_dis, min_score = read_combined_report_line(line)
            y_closest.append(closest)
            y_real.append(real)
            y_pred.append(pred)

            if real == newfamily:
                total_new_family += 1
            if min_score > mad_threshold:
                sample_result_dict[sample_idx] = [real, pred, closest, min_dis, min_score]

    ordered_sample_result_dict = OrderedDict(sorted(sample_result_dict.items(),
                                            key=lambda x: x[1][3],
                                            reverse=True))
    with open(save_ordered_dis_path, 'w') as f:
        f.write('sample_idx,real_label,min_dis\n')
        for k, v in ordered_sample_result_dict.items():
            f.write(f'{k},{v[0]},{v[3]}\n')

    plot_inspection_effort_pr_value_by_dist(ordered_sample_result_dict, newfamily, total_new_family,
                                            dist_effort_pr_value_fig_path, dist_one_by_one_check_result_path)

    '''get the drift closest family confusion matrix'''
    acc_classifier = accuracy_score(y_real, y_pred)
    acc_closest = accuracy_score(y_real, y_closest)
    cm = confusion_matrix(y_real, y_closest)

    append_accuracy_result_to_final_report(acc_classifier, acc_closest, dist_one_by_one_check_result_path)

    logging.debug(f'use drift closest family as prediction accuracy:\n {acc_closest}')
    logging.debug(f'use drift closest family as prediction confusion matrix:\n {cm}')


def plot_inspection_effort_pr_value_by_dist(sorted_samples, newfamily, total_new_family, fig_path, pr_result_path):
    TP, FP = 0, 0
    precision_list, recall_list = [], []
    inspection_cnt_list = range(1, len(sorted_samples) + 1)

    with open(pr_result_path, 'w') as f:
        f.write('sample_idx,real,closest,TP,FP,precision,recall\n')
        for sample_idx, values in sorted_samples.items():
            if values[0] == newfamily:
                TP += 1
            else:
                FP += 1

            precision = TP / (TP + FP)
            recall = TP / total_new_family
            precision_list.append(precision)
            recall_list.append(recall)
            f.write(f'{sample_idx},{values[0]},{values[2]},{TP},{FP},{precision},{recall}\n')

        best_inspection_cnt, best_precision, best_recall, best_f1 = get_best_result(precision_list, recall_list)
        best_inspection_percent = best_inspection_cnt / len(precision_list)
        f.write(f'\n\nTotal: {len(sorted_samples)}\n')
        f.write(f'best inspection count: {best_inspection_cnt}, percent: {best_inspection_percent}\n')
        f.write(f'best performance -- precision: {best_precision * 100:.2f}%, recall: {best_recall * 100:.2f}%\
                f1: {best_f1 * 100:.2f}%\n')

    annotation_text = f'inspect {best_inspection_cnt} samples\nP:{best_precision * 100:.2f}%, R:{best_recall * 100:.2f}%\nF1:{best_f1*100:.2f}%'

    fig, ax = plt.subplots()
    ax.plot(inspection_cnt_list, precision_list, label='precision', color='g')
    ax.plot(inspection_cnt_list, recall_list, label='recall', color='b')
    ax.annotate(annotation_text, (best_inspection_cnt, best_precision), fontsize=6, color='red')
    ax.set_title('Precision Recall value as the change of inspection efforts', fontsize=12)
    ax.set_xticks(np.around(np.linspace(0, len(inspection_cnt_list), 10), decimals=0))
    ax.set_xlabel('Inspection Effort (# of Samples)', fontsize=16)
    ax.set_ylabel('Rate', fontsize=16)
    ax.legend(loc='best')
    fig.savefig(fig_path, dpi=200)
    plt.clf()


def append_accuracy_result_to_final_report(acc_classifier, acc_closest, dist_one_by_one_check_result_path):
    with open(dist_one_by_one_check_result_path, 'a') as f:
        f.write('\n====================================\n')
        f.write(f'classifier acc on testing set: {acc_classifier}\n')
        f.write(f'use drift closest family as prediction accuracy on testing set: {acc_closest}\n')


def get_best_result(precision_list, recall_list):
    best_inspection_cnt = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    for i in range(len(precision_list)):
        try:
            f1 = 2 * precision_list[i] * recall_list[i] / (precision_list[i] + recall_list[i])
            if f1 > best_f1:
                best_f1 = f1
                best_inspection_cnt = i + 1
                best_precision = precision_list[i]
                best_recall = recall_list[i]
        except:
            logging.debug(f'list-{i}\n {traceback.format_exc()}')
            continue
    return best_inspection_cnt, best_precision, best_recall, best_f1


def read_combined_report_line(line):
    sample_idx, real, pred, closest, is_drift, prob, min_dis, min_score = line.strip().split(',')
    real = int(real)
    pred = int(pred)
    closest = int(closest)
    prob = float(prob)
    min_dis = float(min_dis)
    min_score = float(min_score)
    return sample_idx, real, pred, closest, is_drift, prob, min_dis, min_score
