import os
import random
import argparse
import numpy as np
import pandas as pd
import fnmatch
import scipy.stats as st
import statistics
import copy
import math
import csv

def get_args():
    parser = argparse.ArgumentParser('argument for feature dominance')
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--feature_enhancement_filter', type=float, default=0.8)
    parser.add_argument('--feature_enhancement', type=float, default=1.5)
    parser.add_argument('--other_loss_coefficient', type=float, default=2)
    return parser.parse_args() 

args = get_args()

dataset = args.dataset_name
if dataset in ['mnist', 'cifar10', 'imagenet_subset', 'gtsrb']:
    datasets = [f'composite_{dataset}_agnostic', f'composite_{dataset}_specific',f'input_{dataset}_all', f'input_{dataset}_one', f'Narcissus_{dataset}', f'dfb_{dataset}', f'{dataset}_adaptive',
                f'{dataset}_adaptive_feature', f'{dataset}_input_adaptive', f'{dataset}_adaptive_specific', f'{dataset}_adaptive_feature_specific', f'{dataset}_input_adaptive_all']
if 'new_BadEncoder_cifar10_svhn' in dataset:
    datasets = ['new_DRUPE_cifar10_svhn']
elif 'new_BadEncoder_cifar10_gtsrb' in dataset:
    datasets = ['new_DRUPE_cifar10_gtsrb']
root = '/BARBIE/dominance/RCS_' + str(args.feature_enhancement) + '_' + str(args.feature_enhancement_filter) + '_' + str(args.other_loss_coefficient)
enhancement_root = os.path.join(root, dataset) 

def config_dataset(dataset):
    if len(dataset.split('_')) > 2:
        if (dataset.split('_')[1]=='new'):
            dataset = dataset.split('_')[3]
        else:
                if 'composite' in dataset or 'input' in dataset or 'dfb' in dataset or 'Narcissus' in dataset:
                    dataset = dataset.split('_')[1]
                else:
                    dataset = dataset.split('_')[2]
    if 'mnist' in dataset:
        num_classes = 10
    elif 'cifar10' in dataset:
        num_classes = 10
    elif 'gtsrb' in dataset:
        num_classes = 43
    elif 'imagenet' in dataset:
        num_classes = 10
    elif 'stl10' in dataset:
        num_classes = 10
    elif 'svhn' in dataset:
        num_classes =10
    return num_classes

rank_number = 5

benign_pattern = 'benign*.csv'
benign_agnostic_pattern = 'benign_agnostic_*.csv'
benign_specific_pattern = 'benign_specific_*.csv'
agnostic_pattern = 'agnostic*.csv'
specific_pattern = 'specific*.csv'
poisoned_pattern = 'poisoned*.csv'
n_cls = config_dataset(dataset)
backdoor_list = ['agnostic', 'specific']
triggers_list = ['patched', 'blending', 'filter']

def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    data = data[:, 1:]

    return data

def record_to_csv(data_array, csv_name, root_restorage):
    """
        model_kind = 0 代表 benign model
        model_kind = 1 代表 agnostic backdoor
        model_kind = 2 代表 specific backdoor
    """
    save_csv_path = os.path.join(root_restorage, csv_name)
    data = pd.DataFrame(data_array)
    data.to_csv(save_csv_path)

def boundary_decision(root, benign_part_models):
    count = 0
    for benign_model in benign_part_models:
        benign_model_path = os.path.join(root, benign_model)
        if os.path.exists(benign_model_path):
            if fnmatch.fnmatch(benign_model, benign_pattern):
                data_record = load_csv(benign_model_path) 
                if ('cifar10' in dataset or 'new' in dataset) and (np.max(data_record) > 100):
                    continue 
                _n_cls = n_cls         
                data_row_mean = np.zeros(_n_cls)
                data_column_mean = np.zeros(_n_cls)
                data_meaning = 0
                data_meaning_count =0 
                for row_i in range(_n_cls):
                    for column_i in range(_n_cls):
                        if row_i != column_i:
                                data_row_mean[row_i] += data_record[row_i, column_i]
                                data_column_mean[row_i] += data_record[column_i,row_i]
                                data_meaning += data_record[row_i, column_i]
                                data_meaning_count += 1
                data_row_mean /= (_n_cls - 1)
                data_column_mean /= (_n_cls -1)
                data_meaning /= data_meaning_count
                data_standard_deviation = 0
                data_standard_deviation_count = 0
                for row_i in range(_n_cls):
                    for column_i in range(_n_cls):
                        if row_i != column_i:
                                data_standard_deviation += (data_record[row_i][column_i] - data_meaning) ** 2
                                data_standard_deviation_count += 1
                data_standard_deviation /= data_standard_deviation_count
                data_standard_deviation = data_standard_deviation ** 0.5
                _data_record = data_record
                for row_i in range(_n_cls):
                    _data_record[row_i, row_i] = np.sum(data_record[row_i]) / (_n_cls-1) 

                judge_index = [np.max(data_row_mean - data_column_mean), np.max(data_column_mean - data_row_mean), np.max(data_row_mean), np.max(data_column_mean), np.min(data_row_mean), np.min(data_column_mean), np.max(_data_record), np.min(_data_record), data_meaning, data_standard_deviation, data_standard_deviation/data_meaning, statistics.mode(_data_record.flatten().tolist()), st.skew(_data_record.flatten()), st.kurtosis(_data_record.flatten()),np.max(_data_record)-np.min(_data_record)]

                if count == 0:
                    benign_boundary = []
                    mean_benign_boundary = []
                    mean_benign_boundary_count = []
                    for boundary_i in range(len(judge_index)*2):
                        benign_boundary.append(judge_index[int(boundary_i/2)])
                        if boundary_i % 2 == 0:
                            mean_benign_boundary.append(judge_index[int(boundary_i/2)])
                            mean_benign_boundary_count.append(1)
                    all_data_record = [[] for i in range(len(judge_index))]
                    print(benign_boundary)
                else:
                    for boundary_i in range(len(judge_index)):
                        if judge_index[boundary_i] > benign_boundary[boundary_i*2]:
                            benign_boundary[boundary_i*2] = judge_index[boundary_i]
                        if judge_index[boundary_i] < benign_boundary[boundary_i*2+1]:
                            benign_boundary[boundary_i*2+1] = judge_index[boundary_i]
                        mean_benign_boundary[boundary_i] += judge_index[boundary_i]
                        mean_benign_boundary_count[boundary_i] += 1
                for boundary_i in range(len(judge_index)):
                    all_data_record[boundary_i].append(judge_index[boundary_i])
                count += 1
    for boundary_i in range(len(mean_benign_boundary)):
        mean_benign_boundary[boundary_i] /= mean_benign_boundary_count[boundary_i]

    all_data_std = [0 for i in range(len(judge_index))]
    for boundary_i in range(len(judge_index)):
        all_data_std[boundary_i] = np.std(all_data_record[boundary_i])
    if np.sum(all_data_std) != 0:
        all_data_std /= np.sum(all_data_std)
    return benign_boundary, mean_benign_boundary, all_data_std

def index_range_enlarge(indexs, index_weights, multi):
    print(indexs)
    enlarge_indexs = indexs
    print('')
    print('Weights:')
    print(index_weights)
    index_weights = index_weights + 1e-6
    index_weights = index_weights / np.min(index_weights)
    index_weights = np.log(index_weights)
    index_weights = (index_weights + sorted(set(index_weights))[1])
    index_weights = index_weights/ np.max(index_weights) * multi + 1
    print('')
    print(index_weights)
    print('')
    for index_i in range(int(len(indexs)/2)):
        if enlarge_indexs[index_i*2] > 0:
            enlarge_indexs[index_i*2] = enlarge_indexs[index_i*2]*index_weights[index_i]
        elif enlarge_indexs[index_i*2] < 0:
            enlarge_indexs[index_i*2] = enlarge_indexs[index_i*2]/index_weights[index_i]
        if enlarge_indexs[index_i*2+1] > 0:
            enlarge_indexs[index_i*2+1] =enlarge_indexs[index_i*2+1]/index_weights[index_i]
        elif enlarge_indexs[index_i*2+1] < 0:
            enlarge_indexs[index_i*2+1] = enlarge_indexs[index_i*2+1]*index_weights[index_i]
    print(enlarge_indexs)
    return enlarge_indexs

if dataset == 'mnist':
    # MNIST
    multi = 1
elif dataset == 'cifar10':
    # CIFAR10
    multi = 0.75
elif dataset == 'gtsrb':
    # GTSRB
    multi = 0.45
elif dataset == 'imagenet_subset':
    # Imagenette
    multi = 0.35

for round_count in range(5):
    print()
    print()
    print('=========================================================')
    print('Round: ', round_count)
    benign_range = {'up_mean': -1000, 'down_mean': 1000, 'up_std':-1000, 'down_std': 1000, 'up_diff': -1000, 'down_diff': 1000, 'up_index': -1000, 'down_index': 1000}

    all_count = 0

    data_record_files = os.listdir(enhancement_root)
    judge_results_list = ['TP', 'TN', 'FP', 'FN']
    true_specific_counts = {}
    true_agnostic_counts = {}
    true_poisoned_count = 0
    judge_agnostic_counts = {}
    judge_specific_counts = {}
    judge_poisoned_counts = {}
    for trigger in triggers_list:
        true_specific_counts[trigger] = 0
        true_agnostic_counts[trigger] = 0
        judge_agnostic_counts[trigger] = {}
        judge_specific_counts[trigger] = {}
        for judge_result in judge_results_list:
            judge_agnostic_counts[trigger][judge_result] = 0
            judge_specific_counts[trigger][judge_result] = 0
            judge_poisoned_counts[judge_result] = 0
    judge_false = []
    data_counts = {'specific': 0, 'agnostic': 0, 'benign': 0, 'poisoned': 0}
    data_meanings = {'specific': 0, 'agnostic': 0, 'benign': 0, 'poisoned': 0}
    data_deviations = {'specific': 0, 'agnostic': 0, 'benign': 0, 'poisoned': 0}
    data_ranges = {'specific': 0, 'agnostic': 0, 'benign': 0, 'poisoned': 0}
    _data_index_meanings = {'specific': 0, 'agnostic': 0, 'benign': 0, 'poisoned': 0}
    _data_index_maxs = {'specific': 0, 'agnostic': 0, 'benign': 0, 'poisoned': 0}
    _data_index_mins = {'specific': 10000000, 'agnostic': 100000000, 'benign': 100000000, 'poisoned': 100000000}
    benign_all_count = 0
    benign_all_models = []
    for data_record_file in data_record_files:
        if fnmatch.fnmatch(data_record_file, benign_pattern):
            benign_all_count += 1
            benign_all_models.append(data_record_file)
    benign_part_models = random.sample(benign_all_models, int(benign_all_count*0.3))
    benign_boundary, mean_benign_boundary, index_weights = boundary_decision(enhancement_root, benign_part_models)
    benign_boundary = index_range_enlarge(benign_boundary, index_weights, multi)
    print('')
    print(mean_benign_boundary)
    benign_part_count = 0
    range_index_all = []
    poisoned_range_index_all = []
    for data_record_file in data_record_files:
        if fnmatch.fnmatch(data_record_file, '*.csv'): 
            enhancement_data_record_path = os.path.join(enhancement_root, data_record_file)
            if os.path.exists(enhancement_data_record_path):
                try:
                    data_record = load_csv(enhancement_data_record_path)
                except:
                    continue
                if fnmatch.fnmatch(data_record_file, specific_pattern):
                    for trigger in triggers_list:
                        if trigger in data_record_file:
                            true_specific_counts[trigger] += 1
                elif fnmatch.fnmatch(data_record_file, agnostic_pattern):
                    for trigger in triggers_list:
                        if trigger in data_record_file:
                            true_agnostic_counts[trigger] += 1
                elif fnmatch.fnmatch(data_record_file, poisoned_pattern):
                    true_poisoned_count += 1
                elif fnmatch.fnmatch(data_record_file, benign_pattern):
                    if data_record_file in benign_part_models:
                        continue
                if np.max(data_record) > 100:
                    print(np.max(data_record))
                    print(np.argmax(data_record))
                    print(data_record_file)
                _n_cls = n_cls         
                data_row_mean = np.zeros(_n_cls)
                data_column_mean = np.zeros(_n_cls)
                data_meaning = 0
                data_meaning_count =0 
                for row_i in range(_n_cls):
                    for column_i in range(_n_cls):
                        if row_i != column_i:
                                data_row_mean[row_i] += data_record[row_i, column_i]
                                data_column_mean[row_i] += data_record[column_i,row_i]
                                data_meaning += data_record[row_i, column_i]
                                data_meaning_count += 1
                data_row_mean /= (_n_cls - 1)
                data_column_mean /= (_n_cls -1)
                data_meaning /= data_meaning_count
                data_standard_deviation = 0
                data_standard_deviation_count = 0
                for row_i in range(_n_cls):
                    for column_i in range(_n_cls):
                        if row_i != column_i:
                                data_standard_deviation += (data_record[row_i][column_i] - data_meaning) ** 2
                                data_standard_deviation_count += 1
                data_standard_deviation /= data_standard_deviation_count
                data_standard_deviation = data_standard_deviation ** 0.5
                _data_record = data_record
                for row_i in range(_n_cls):
                    _data_record[row_i, row_i] = np.sum(data_record[row_i]) / (_n_cls-1) 

                _range_index = st.kurtosis(_data_record.flatten())
                judge_index = [np.max(data_row_mean - data_column_mean), np.max(data_column_mean - data_row_mean), np.max(data_row_mean), np.max(data_column_mean), np.min(data_row_mean), np.min(data_column_mean), np.max(_data_record), np.min(_data_record), data_meaning, data_standard_deviation, data_standard_deviation/data_meaning, statistics.mode(_data_record.flatten().tolist()), st.skew(_data_record.flatten()), st.kurtosis(_data_record.flatten()) ,np.max(_data_record)-np.min(_data_record)]
                if fnmatch.fnmatch(data_record_file, benign_pattern):
                    if len(range_index_all) == 0:
                        for range_i in range(len(judge_index)):
                            range_index_all.append(judge_index[range_i])
                            range_index_all.append(judge_index[range_i])
                if fnmatch.fnmatch(data_record_file, poisoned_pattern)  or fnmatch.fnmatch(data_record_file, agnostic_pattern) or fnmatch.fnmatch(data_record_file, specific_pattern):
                    if len(poisoned_range_index_all) == 0:
                        for range_i in range(len(judge_index)):
                            poisoned_range_index_all.append(judge_index[range_i])
                            poisoned_range_index_all.append(judge_index[range_i])
                data_index = np.zeros((_n_cls, _n_cls))
                _data_index = np.zeros((_n_cls, _n_cls))
                for row_i in range(_n_cls):
                    for column_i in range(_n_cls):
                        if row_i != column_i:
                            data_index[row_i, column_i] = data_record[row_i, column_i] / data_row_mean[row_i]
                            _data_index[row_i, column_i] = data_index[row_i, column_i]
                        
                _data_index_meaning = np.mean(_data_index)
                for row_i in range(_n_cls):
                    _data_index[row_i][row_i] = data_row_mean[row_i]

                judge_flag = 0
                _judge_index =0 
                for i in range(len(judge_index)):
                    if judge_index[i] > benign_boundary[2*i]:
                        judge_flag = 1
                    elif judge_index[i] < benign_boundary[2*i+1]:
                        judge_flag = 1
                if fnmatch.fnmatch(data_record_file, agnostic_pattern) or fnmatch.fnmatch(data_record_file, benign_pattern):
                    for trigger in triggers_list:
                        if (trigger in data_record_file) or fnmatch.fnmatch(data_record_file, benign_pattern):
                            if judge_flag:
                                if fnmatch.fnmatch(data_record_file, agnostic_pattern):
                                    judge_agnostic_counts[trigger]['TP'] += 1
                                else:
                                    judge_agnostic_counts[trigger]['FP'] += 1
                                    judge_false.append(data_record_file)
                            else:
                                if fnmatch.fnmatch(data_record_file, agnostic_pattern):
                                    judge_agnostic_counts[trigger]['FN'] += 1
                                    judge_false.append(data_record_file)
                                else:
                                    judge_agnostic_counts[trigger]['TN'] += 1

                if fnmatch.fnmatch(data_record_file, specific_pattern) or fnmatch.fnmatch(data_record_file, benign_pattern):
                    for trigger in triggers_list:
                        if (trigger in data_record_file) or fnmatch.fnmatch(data_record_file, benign_pattern):
                            if judge_flag:
                                if fnmatch.fnmatch(data_record_file, specific_pattern):
                                    judge_specific_counts[trigger]['TP'] += 1
                                else:
                                    judge_specific_counts[trigger]['FP'] += 1
                                    judge_false.append(data_record_file)
                            else:
                                if fnmatch.fnmatch(data_record_file, specific_pattern):
                                    judge_specific_counts[trigger]['FN'] += 1
                                    judge_false.append(data_record_file)
                                else:
                                    judge_specific_counts[trigger]['TN'] += 1

                if fnmatch.fnmatch(data_record_file, poisoned_pattern) or fnmatch.fnmatch(data_record_file, benign_pattern):
                    if judge_flag:
                        if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                            judge_poisoned_counts['TP'] += 1
                        else:
                            judge_poisoned_counts['FP'] += 1
                            judge_false.append(data_record_file)
                    else:
                        if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                            judge_poisoned_counts['FN'] += 1
                            judge_false.append(data_record_file)
                        else:
                            judge_poisoned_counts['TN'] += 1

                if np.max(data_record) > 100:
                    continue
                if fnmatch.fnmatch(data_record_file, benign_pattern):
                    for range_i in range(len(judge_index)):
                        if judge_index[range_i] > range_index_all[range_i*2]:
                            range_index_all[range_i*2] = judge_index[range_i]
                        elif judge_index[range_i] < range_index_all[range_i*2+1]:
                            range_index_all[range_i*2+1] = judge_index[range_i]
                
                if fnmatch.fnmatch(data_record_file, poisoned_pattern) or fnmatch.fnmatch(data_record_file, agnostic_pattern) or fnmatch.fnmatch(data_record_file, specific_pattern):
                    for range_i in range(len(judge_index)):
                        if judge_index[range_i] > poisoned_range_index_all[range_i*2]:
                            poisoned_range_index_all[range_i*2] = judge_index[range_i]
                        elif judge_index[range_i] < poisoned_range_index_all[range_i*2+1]:
                            poisoned_range_index_all[range_i*2+1] = judge_index[range_i]                
    if round_count == 0:
        all_results = {}
    for trigger in triggers_list:
        print()
        print(f'For the results of {trigger}: ')
        if true_agnostic_counts[trigger] != 0:
            print()
            print('Agnostic:')
            print('-TPR:')
            if judge_agnostic_counts[trigger]['TP']+judge_agnostic_counts[trigger]['FN'] != 0:
                print(judge_agnostic_counts[trigger]['TP']/(judge_agnostic_counts[trigger]['TP']+judge_agnostic_counts[trigger]['FN']))
                if round_count == 0:
                    all_results[trigger+'_agnostic'] = {}
                    all_results[trigger+'_agnostic']['TPR'] = []
                all_results[trigger+'_agnostic']['TPR'].append(judge_agnostic_counts[trigger]['TP']/(judge_agnostic_counts[trigger]['TP']+judge_agnostic_counts[trigger]['FN']))
                    
            print('-FPR:')
            if judge_agnostic_counts[trigger]['FP']+judge_agnostic_counts[trigger]['TN'] != 0:
                print(judge_agnostic_counts[trigger]['FP']/(judge_agnostic_counts[trigger]['FP']+judge_agnostic_counts[trigger]['TN']))
                if round_count == 0:
                    all_results[trigger+'_agnostic']['FPR'] = []
                all_results[trigger+'_agnostic']['FPR'].append(judge_agnostic_counts[trigger]['FP']/(judge_agnostic_counts[trigger]['FP']+judge_agnostic_counts[trigger]['TN']))
        
        if true_specific_counts[trigger] != 0:
            print()
            print('Specific:')
            print('-TPR:')
            if judge_specific_counts[trigger]['TP']+judge_specific_counts[trigger]['FN'] != 0:
                print(judge_specific_counts[trigger]['TP']/(judge_specific_counts[trigger]['TP']+judge_specific_counts[trigger]['FN']))
                if round_count == 0:
                    all_results[trigger+'_specific'] = {}
                    all_results[trigger+'_specific']['TPR'] = []
                all_results[trigger+'_specific']['TPR'].append(judge_specific_counts[trigger]['TP']/(judge_specific_counts[trigger]['TP']+judge_specific_counts[trigger]['FN']))
            print('-FPR:')
            if judge_specific_counts[trigger]['FP']+judge_specific_counts[trigger]['TN'] != 0:
                print(judge_specific_counts[trigger]['FP']/(judge_specific_counts[trigger]['FP']+judge_specific_counts[trigger]['TN']))
                if round_count == 0:
                    all_results[trigger+'_specific']['FPR'] = []
                all_results[trigger+'_specific']['FPR'].append(judge_specific_counts[trigger]['FP']/(judge_specific_counts[trigger]['FP']+judge_specific_counts[trigger]['TN']))

    if true_poisoned_count != 0:
        print()
        print('Poisoned:')
        print('-TPR:')
        if judge_poisoned_counts['TP']+judge_poisoned_counts['FN'] != 0:
            print(judge_poisoned_counts['TP']/(judge_poisoned_counts['TP']+judge_poisoned_counts['FN']))
            if round_count == 0:
                all_results[dataset] = {}
                all_results[dataset]['TPR'] = []
            all_results[dataset]['TPR'].append(judge_poisoned_counts['TP']/(judge_poisoned_counts['TP']+judge_poisoned_counts['FN']))
        print('-FPR:')
        if judge_poisoned_counts['FP']+judge_poisoned_counts['TN'] != 0:
            print(judge_poisoned_counts['FP']/(judge_poisoned_counts['FP']+judge_poisoned_counts['TN']))
            if round_count == 0:
                all_results[dataset]['FPR'] = []
            all_results[dataset]['FPR'].append(judge_poisoned_counts['FP']/(judge_poisoned_counts['FP']+judge_poisoned_counts['TN']))

    if dataset in ['mnist', 'cifar10', 'gtsrb', 'imagenet_subset', 'new_BadEncoder_cifar10_svhn', 'new_BadEncoder_cifar10_gtsrb']:
        for _dataset in datasets:
            _enhancement_root = os.path.join(root, _dataset)
            try:
                data_record_files = os.listdir(_enhancement_root)
            except:
                continue
            true_poisoned_count = 0
            judge_poisoned_counts = {}
            for judge_result in judge_results_list:
                judge_poisoned_counts[judge_result] = 0
            judge_false = []
            for data_record_file in data_record_files:
                if fnmatch.fnmatch(data_record_file, '*.csv'): 
                    enhancement_data_record_path = os.path.join(_enhancement_root, data_record_file)
                    if os.path.exists(enhancement_data_record_path):
                        try:
                            data_record = load_csv(enhancement_data_record_path)
                        except:
                            continue
                        if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                            true_poisoned_count += 1
                        elif fnmatch.fnmatch(data_record_file, benign_pattern):
                            if data_record_file in benign_part_models:
                                continue
                        _n_cls = n_cls         
                        data_row_mean = np.zeros(_n_cls)
                        data_column_mean = np.zeros(_n_cls)
                        data_meaning = 0
                        data_meaning_count =0 
                        for row_i in range(_n_cls):
                            for column_i in range(_n_cls):
                                if row_i != column_i:
                                        data_row_mean[row_i] += data_record[row_i, column_i]
                                        data_column_mean[row_i] += data_record[column_i,row_i]
                                        data_meaning += data_record[row_i, column_i]
                                        data_meaning_count += 1
                        data_row_mean /= (_n_cls - 1)
                        data_column_mean /= (_n_cls -1)
                        data_meaning /= data_meaning_count
                        data_standard_deviation = 0
                        data_standard_deviation_count = 0
                        for row_i in range(_n_cls):
                            for column_i in range(_n_cls):
                                if row_i != column_i:
                                        data_standard_deviation += (data_record[row_i][column_i] - data_meaning) ** 2
                                        data_standard_deviation_count += 1
                        data_standard_deviation /= data_standard_deviation_count
                        data_standard_deviation = data_standard_deviation ** 0.5
                        _data_record = data_record
                        for row_i in range(_n_cls):
                            _data_record[row_i, row_i] = np.sum(data_record[row_i]) / (_n_cls-1) 

                        _range_index = st.kurtosis(_data_record.flatten())
                        judge_index = [np.max(data_row_mean - data_column_mean), np.max(data_column_mean - data_row_mean), np.max(data_row_mean), np.max(data_column_mean), np.min(data_row_mean), np.min(data_column_mean), np.max(_data_record), np.min(_data_record), data_meaning, data_standard_deviation, data_standard_deviation/data_meaning, statistics.mode(_data_record.flatten().tolist()), st.skew(_data_record.flatten()), st.kurtosis(_data_record.flatten()) ,np.max(_data_record)-np.min(_data_record)]
                        if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                            if len(poisoned_range_index_all) == 0:
                                for range_i in range(len(judge_index)):
                                    poisoned_range_index_all.append(judge_index[range_i])
                                    poisoned_range_index_all.append(judge_index[range_i])
                        data_index = np.zeros((_n_cls, _n_cls))
                        _data_index = np.zeros((_n_cls, _n_cls))
                        for row_i in range(_n_cls):
                            for column_i in range(_n_cls):
                                if row_i != column_i:
                                    data_index[row_i, column_i] = data_record[row_i, column_i] / data_row_mean[row_i]
                                    _data_index[row_i, column_i] = data_index[row_i, column_i]
                                
                        _data_index_meaning = np.mean(_data_index)
                        for row_i in range(_n_cls):
                            _data_index[row_i][row_i] = data_row_mean[row_i]

                        ### judge ###
                        judge_flag = 0
                        _judge_index =0 
                        for i in range(len(judge_index)):
                            if judge_index[i] > benign_boundary[2*i]:
                                judge_flag = 1
                            elif judge_index[i] < benign_boundary[2*i+1]:
                                judge_flag = 1

                        if fnmatch.fnmatch(data_record_file, poisoned_pattern) or fnmatch.fnmatch(data_record_file, benign_pattern):
                            if judge_flag:
                                if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                                    judge_poisoned_counts['TP'] += 1
                                else:
                                    judge_poisoned_counts['FP'] += 1
                                    judge_false.append(data_record_file)
                            else:
                                if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                                    judge_poisoned_counts['FN'] += 1
                                    judge_false.append(data_record_file)
                                else:
                                    judge_poisoned_counts['TN'] += 1

                        _range_index = st.kurtosis(_data_record.flatten())
                        
                        if fnmatch.fnmatch(data_record_file, poisoned_pattern):
                            for range_i in range(len(judge_index)):
                                if judge_index[range_i] > poisoned_range_index_all[range_i*2]:
                                    poisoned_range_index_all[range_i*2] = judge_index[range_i]
                                elif judge_index[range_i] < poisoned_range_index_all[range_i*2+1]:
                                    poisoned_range_index_all[range_i*2+1] = judge_index[range_i]                

            print('For the results of ', _dataset)
            if true_poisoned_count != 0:
                print()
                print('Poisoned:')
                print('-TPR:')
                if judge_poisoned_counts['TP']+judge_poisoned_counts['FN'] != 0:
                    print(judge_poisoned_counts['TP']/(judge_poisoned_counts['TP']+judge_poisoned_counts['FN']))
                    if round_count == 0:
                        all_results[_dataset] = {}
                        all_results[_dataset]['TPR'] = []
                    all_results[_dataset]['TPR'].append(judge_poisoned_counts['TP']/(judge_poisoned_counts['TP']+judge_poisoned_counts['FN']))
                print('-FPR:')
                if judge_poisoned_counts['FP']+judge_poisoned_counts['TN'] != 0:
                    print(judge_poisoned_counts['FP']/(judge_poisoned_counts['FP']+judge_poisoned_counts['TN']))
                    if round_count == 0:
                        all_results[_dataset]['FPR'] = []
                    all_results[_dataset]['FPR'].append(judge_poisoned_counts['FP']/(judge_poisoned_counts['FP']+judge_poisoned_counts['TN']))

    print('Benign:')
    print(range_index_all)
    print('Poisoned:')
    print(poisoned_range_index_all)
    print('Judge:')
    print(benign_boundary)

print('---Best')
max_index = -1
min_index = -1
for key in all_results.keys():
    print('-for results for ', key)
    if max_index == -1:
        max_index = np.argmax(all_results[key]['TPR'])
        print(max_index)
    print('TPR: {:.5f}'.format(all_results[key]['TPR'][max_index]))
    try:
        print('FPR: {:.5f}'.format(all_results[key]['FPR'][max_index]))
    except:
        continue
    print()
    if min_index == -1:
        min_index = np.argmin(all_results[key]['TPR'])
        print(min_index)
    print('TPR: {:.5f}'.format(all_results[key]['TPR'][min_index]))
    try:
        print('FPR: {:.5f}'.format(all_results[key]['FPR'][min_index]))
    except:
        continue
    print()  

print()
print()
print('---Average')
for key in all_results.keys():
    print('-for results for ', key)
    print('TPR: {:.5f}'.format(np.mean(all_results[key]['TPR'])))
    try:
        print('FPR: {:.5f}'.format(np.mean(all_results[key]['FPR'])))
    except:
        continue
    print()
