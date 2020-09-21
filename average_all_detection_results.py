import os
import sys
import re
import numpy as np

from collections import Counter

import cade.data as data


def main(dataset, use_pure_ae, families_cnt, last_label, margin, mad, cae_lambda):
    if use_pure_ae == 0:
        REPORT_FOLDER = 'reports'
    else:
        REPORT_FOLDER = 'pure_ae_reports'

    if dataset == 'drebin':
        families = range(families_cnt)
    else:
        families = range(1, families_cnt)
        name_dict = {1: 'SSH', 2: 'Hulk', 3: 'Infilteration'}

    p1 = re.compile('precision: \d+\.\d+')
    p2 = re.compile('recall: \d+\.\d+')
    p3 = re.compile('f1: \d+\.\d+')
    p4 = re.compile('best inspection count: \d+')

    precision_list, recall_list, f1_list, inspect_cnt_list = [], [], [], []
    involved_families_list = []
    normalized_inspect_cnt_list = []

    for i in families:
        '''calc how many new family samples in the testing set'''
        if dataset == 'drebin':
            single_dataset = f'drebin_new_{i}'
            name = i
        else:
            single_dataset = f'IDS_new_{name_dict[i]}'
            name = name_dict[i]

        X_train, y_train, X_test, y_test = data.load_features(single_dataset, i)

        total_new_family = len(np.where(y_test == last_label)[0])

        '''record results for each family'''
        result_path = os.path.join(f'{REPORT_FOLDER}', single_dataset, f'dist_mlp_one_by_one_check_pr_value_m{margin}_mad{mad}_lambda{cae_lambda}.csv')
        with open(result_path, 'r') as f:
            content = f.read()
        precision = float(re.findall(p1, content)[0].replace('precision: ', '')) / 100
        recall = float(re.findall(p2, content)[0].replace('recall: ', '')) / 100
        f1 = float(re.findall(p3, content)[0].replace('f1: ', '')) / 100
        inspect_cnt = int(re.findall(p4, content)[0].replace('best inspection count: ', ''))
        print(f'family {name}: precision: {precision * 100}%, recall: {recall * 100}%, f1: {f1 * 100}%, inspect: {inspect_cnt}')
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        inspect_cnt_list.append(inspect_cnt)
        normalized_inspect_cnt_list.append(inspect_cnt / total_new_family)

        # check the involved families in the drifting samples when we get the best results
        involved_families = []
        with open(result_path, 'r') as f:
            next(f)
            for idx, line in enumerate(f):
                if idx < inspect_cnt:
                    line = line.strip().split(',')
                    real = line[1]
                    involved_families.append(real)

        involved_families_list.append(involved_families)

    print('============================================')
    print('avg +/- std (final result in Table 3): ')
    print(f'precision: {np.average(precision_list) * 100:.2f}% +/- {np.std(precision_list):.2f}')
    print(f'recall: {np.average(recall_list) * 100:.2f}% +/- {np.std(recall_list):.2f}')
    print(f'f1: {np.average(f1_list) * 100:.2f}% +/- {np.std(f1_list):.2f}')
    print(f'inspect_cnt: {np.average(inspect_cnt_list):.2f} +/- {np.std(inspect_cnt_list):.2f}')
    print(f'normalized inspect_cnt: {np.average(normalized_inspect_cnt_list):.2f} ' +
          f'+/- {np.std(normalized_inspect_cnt_list):.2f}')
    print('============================================')

    saved_report_folder = f'{REPORT_FOLDER}/average_{dataset}'
    os.makedirs(saved_report_folder, exist_ok=True)
    with open(f'{saved_report_folder}/average_{dataset}_result_margin{margin}_mad{mad}_lambda{cae_lambda}.txt', 'w') as f:
        f.write('family_idx,precision,recall,f1,insepct_cnt,normalized_inspect_cnt\n')
        for i in range(len(precision_list)):
            if dataset == 'drebin':
                name = i
            else:
                name = name_dict[i+1]
            f.write(f'{name},{precision_list[i]:.4f},{recall_list[i]:.4f},{f1_list[i]:.4f},' + \
                    f'{inspect_cnt_list[i]:.2f},{normalized_inspect_cnt_list[i]:.2f}\n')

        f.write('============================================\n')
        f.write('avg +/- std (final result in Table 3): \n')
        f.write(f'precision: {np.average(precision_list) * 100:.2f}% +/- {np.std(precision_list):.2f}\n')
        f.write(f'recall: {np.average(recall_list) * 100:.2f}% +/- {np.std(recall_list):.2f}\n')
        f.write(f'f1: {np.average(f1_list) * 100:.2f}% +/- {np.std(f1_list):.2f} \n')
        f.write(f'inspect_cnt: {np.average(inspect_cnt_list):.2f} +/- {np.std(inspect_cnt_list):.2f}\n')
        f.write(f'normalized inspect_cnt: {np.average(normalized_inspect_cnt_list):.2f} ' +
                f'+/- {np.std(normalized_inspect_cnt_list):.2f}\n')

        f.write('============================================\n')
        for i in range(len(involved_families_list)):
            if dataset == 'drebin':
                name = i
            else:
                name = name_dict[i+1]
            f.write(f'family {name}:\t families detected as drifting: {Counter(involved_families_list[i])}\n')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'usage: "python -u average_all_detection_results.py drebin 0", ' +
               'where 0 for CADE, 1 for vanilla autoencoder. You may also specify to use drebin or IDS.')
        sys.exit(-1)

    dataset = sys.argv[1]
    use_pure_ae = int(sys.argv[2])

    if dataset == 'drebin':
        families_cnt = 8
        last_label = 7
    elif dataset == 'IDS':
        families_cnt = 4
        last_label = 3
    else:
        print('dataset could only be "drebin" or "IDS"')
        sys.exit(-1)

    if use_pure_ae:
        mad = 0.0
    else:
        mad = 3.5

    margin = 10.0
    cae_lambda = 0.1

    main(dataset, use_pure_ae, families_cnt, last_label, margin, mad, cae_lambda)
