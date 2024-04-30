import os
import ast
import csv
import copy
import math
import torch
import fnmatch
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from model_inversion import get_features, model_segmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser('argument for feature dominance')
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--feature_repetition', type=int, default=1)
    parser.add_argument('--feature_enhancement_filter', type=float, default=1.0)
    parser.add_argument('--feature_enhancement', type=float, default=0.1)
    parser.add_argument('--other_loss_coefficient', type=float, default=0.5)
    parser.add_argument('--feature_influence_threshold', type=float, default=1e-3)
    parser.add_argument('--dominance_step_up', type=float, default=4)
    parser.add_argument('--dominance_step_down', type=float, default=5)
    parser.add_argument('--metric', type=str, default='softmax_score', choices=['logit', 'softmax_score'])
    parser.add_argument('--use_transpose_correction', type=ast.literal_eval, default=False,
                        help='mul the correction factor -- (a,b)/(b,a) if (a,b) is larger')
    parser.add_argument('--root', type=str, 
                        default='/BARBIE/saved_models')
    return parser.parse_args() 

args = get_args()

root = args.root
save_csv_path = '/BARBIE/dominance/RCS_' + str(args.feature_enhancement) + '_' + str(args.feature_enhancement_filter) + '_' + str(args.other_loss_coefficient)

print()
print('Run feature_dominance.py!')
print()
print('Parameter Configuration: ')
print(f'Repetitions of feature reconstruction: {args.feature_repetition}')
print(f'Coefficient of other loss influence in criterion: {args.other_loss_coefficient}')
print(f'Threshold of feature influence: {args.feature_influence_threshold}')
print()

datasets = [args.dataset_name]

dataset_re_ag_dict = {'imagenet_subset': 10, 'cifar10': 20, 'gtsrb': 5, 'mnist': 20}
dataset_re_sp_dict = {'imagenet_subset': 3, 'cifar10': 8, 'gtsrb': 4, 'mnist': 8}

dataset_arch_dict = {'imagenet_subset': 'resnet50', 'cifar10': 'vgg16', 'gtsrb': 'google_net', 'mnist': 'simple_cnn',
                     'dfb_imagenet_subset': 'resnet50','dfb_cifar10': 'vgg16', 'dfb_gtsrb': 'google_net', 'dfb_mnist': 'simple_cnn',
                     'Narcissus_imagenet_subset': 'resnet50', 'Narcissus_cifar10': 'vgg16', 
                     'Narcissus_gtsrb': 'google_net','Narcissus_mnist': 'simple_cnn',
                     'input_mnist_all': 'simple_cnn', 'input_cifar10_all': 'vgg16', 'input_gtsrb_all': 'google_net', 'input_imagenet_subset_all': 'resnet50',
                     'input_mnist_one': 'simple_cnn', 'input_cifar10_one': 'vgg16', 'input_gtsrb_one': 'google_net', 'input_imagenet_subset_one': 'resnet50',
                     'composite_mnist_agnostic': 'simple_cnn', 'composite_cifar10_agnostic': 'vgg16', 'composite_gtsrb_agnostic': 'google_net', 'composite_imagenet_subset_agnostic': 'resnet50',
                     'composite_mnist_specific': 'simple_cnn', 'composite_cifar10_specific': 'vgg16', 'composite_gtsrb_specific': 'google_net', 'composite_imagenet_subset_specific': 'resnet50'}
segment_layer_dict = {'resnet50': 2, 'vgg16': 2, 'google_net': 2, 'simple_cnn': 1}
dataset_ncls_dict = {'imagenet_subset': 10, 'cifar10': 10, 'gtsrb': 43, 'mnist': 10,
                     'dfb_imagenet_subset': 10,'dfb_cifar10': 10, 'dfb_gtsrb': 43, 'dfb_mnist': 10,
                     'Narcissus_imagenet_subset': 10, 'Narcissus_cifar10': 10, 
                     'Narcissus_gtsrb': 43,'Narcissus_mnist': 10,
                     'input_mnist_all': 10, 'input_cifar10_all': 10, 'input_gtsrb_all': 43, 'input_imagenet_subset_all': 10,
                     'input_mnist_one': 10, 'input_cifar10_one': 10, 'input_gtsrb_one': 43, 'input_imagenet_subset_one': 10,
                     'composite_mnist_agnostic': 10, 'composite_cifar10_agnostic': 10, 'composite_gtsrb_agnostic': 43, 'composite_imagenet_subset_agnostic': 10,
                     'composite_mnist_specific': 10, 'composite_cifar10_specific': 10, 'composite_gtsrb_specific': 43, 'composite_imagenet_subset_specific': 10}
dataset_size_dict = {'imagenet_subset': 224, 'cifar10': 32, 'gtsrb': 32, 'mnist': 28,
                     'dfb_imagenet_subset': 224,'dfb_cifar10': 32, 'dfb_gtsrb': 32, 'dfb_mnist': 28,
                     'Narcissus_imagenet_subset': 224, 'Narcissus_cifar10': 32, 
                     'Narcissus_gtsrb': 32,'Narcissus_mnist': 28,
                     'input_mnist_all': 28, 'input_cifar10_all': 32, 'input_gtsrb_all': 32, 'input_imagenet_subset_all': 224,
                     'input_mnist_one': 28, 'input_cifar10_one': 32, 'input_gtsrb_one': 32, 'input_imagenet_subset_one': 224,
                     'composite_mnist_agnostic': 28, 'composite_cifar10_agnostic': 32, 'composite_gtsrb_agnostic': 32, 'composite_imagenet_subset_agnostic': 224,
                     'composite_mnist_specific': 28, 'composite_cifar10_specific': 32, 'composite_gtsrb_specific': 32, 'composite_imagenet_subset_specific': 224}
dataset_specific_backdoor_targeted_classes_dict = {'imagenet_subset': [0, 6, 7, 9],
                                                   'cifar10': [1, 2, 3],
                                                   'gtsrb': [7, 8],
                                                   'mnist': [6, 7, 8]}                                        
backdoor_types = ['agnostic', 'specific']
trigger_types = ['blending_img', 'filter_img', 'patched_img']
saved_model_name_pattern = '*.pth'

def feature_influence(features, influence_feature_slices, affected_feature_slices, model, n_cls):
    features_dominance = np.zeros((n_cls,n_cls))
    dominance_repetition = np.zeros((n_cls,n_cls))
    with torch.no_grad():
        for num_i in range(len(influence_feature_slices[0])):
            for inflence_idx in range(n_cls):
                feature_influence = influence_feature_slices[inflence_idx][num_i]
                for affected_idx in range(n_cls):
                    if affected_idx != inflence_idx:
                        whole_feature_affected = features[affected_idx][num_i]
                        feature_affected = affected_feature_slices[affected_idx][num_i]
                        model_output = model(whole_feature_affected)
                        model_pred = model_output.max(-1)[1]
                    
                        feature_dominance = 0
                        dominance_step = 0.25
                        dominance_up_times = 0
                        test_pred_label = [0 for i in range(n_cls)]
                        while dominance_step > args.feature_influence_threshold and dominance_step < 1e5:
                            influence_coefficient = feature_dominance + dominance_step
                            feature_input = (1-influence_coefficient) * feature_affected + influence_coefficient * feature_influence
                            whole_feature_input = whole_feature_affected - feature_affected + feature_input
                            model_output = model(whole_feature_input)
                            model_pred = model_output.max(-1)[1]
                            if model_pred[0] != affected_idx:
                                dominance_step /= args.dominance_step_down
                                test_pred_label[int(model_pred[0])] += 1
                            else:
                                feature_dominance += dominance_step
                                dominance_up_times += 1
                            if dominance_up_times >= (args.dominance_step_up/2):
                                dominance_step *= args.dominance_step_up
                                dominance_up_times = 0
                        features_dominance[inflence_idx, affected_idx] += feature_dominance
                        dominance_repetition[inflence_idx, affected_idx] += 1
    for inflence_idx in range(n_cls):
        for affected_idx in range(n_cls):
            if dominance_repetition[inflence_idx, affected_idx] != 0:
                features_dominance[inflence_idx, affected_idx] = features_dominance[inflence_idx, affected_idx] / dominance_repetition[inflence_idx, affected_idx]

    return features_dominance

def feature_enhance(features, _features, args):
    if args.feature_enhancement != 0:
        diff_feature = torch.zeros_like(features[0][0])
        diff_feature_mean = 0 
        diff_feature_nonzero_count = 0
        diff_feature_count = 0
        for features_id in range(len(features)):
            for feature_id in range(len(features[features_id])):
                diff_feature += features[features_id][feature_id] - _features[features_id][feature_id]
                diff_feature_mean += torch.sum(diff_feature)
                diff_feature_nonzero_count += torch.count_nonzero(diff_feature).item()
                diff_feature_count += 1
        diff_feature = args.feature_enhancement * diff_feature/diff_feature_count
        diff_feature_mean = args.feature_enhancement * diff_feature_mean/diff_feature_nonzero_count
        up = torch.ones_like(diff_feature)
        down = torch.zeros_like(diff_feature)
        diff_feature_filter = torch.where(diff_feature > args.feature_enhancement_filter*diff_feature_mean, up, down)
        diff_feature = torch.mul(diff_feature, diff_feature_filter)
        for features_id in range(len(features)):
            for feature_id in range(len(features[features_id])):
                coefficient = 1 - torch.sum(diff_feature) / torch.sum(features[features_id][feature_id])
                features[features_id][feature_id] = coefficient * features[features_id][feature_id] + diff_feature
    return features

def record_to_csv(data_array, dataset, saved_model_name, model_kind, target=-1, source=-1):
    """
        model_kind = 0 -- benign models
        model_kind = 1 -- models backdoored with a source-agnostic backdoor
        model_kind = 2 -- models backdoored with a source-specific backdoor
        model_kind = 3 -- models backdoored with other backdoors
    """
    global benign_csv_count, agnostic_csv_count, specific_csv_count
    dataset_save_csv_path = os.path.join(save_csv_path, dataset)
    if not os.path.exists(dataset_save_csv_path):
        os.makedirs(dataset_save_csv_path)
    if model_kind == 0:
        csv_name = 'benign_' + saved_model_name + '_' + str(benign_csv_count) + '.csv'
        csv_path = os.path.join(dataset_save_csv_path, csv_name)
        data = pd.DataFrame(data_array)
        data.to_csv(csv_path)
    elif model_kind == 1:
        agnostic_csv_count += 1
        csv_name = 'agnostic_' + saved_model_name + '_' + str(agnostic_csv_count) + '_' + str(target) + '.csv'
        csv_path = os.path.join(dataset_save_csv_path, csv_name)
        data = pd.DataFrame(data_array)
        data.to_csv(csv_path)
    elif model_kind == 2:
        specific_csv_count += 1
        csv_name = 'specific_' + saved_model_name + '_' + str(specific_csv_count) + '_' + str(target) + '_' + str(source) + '.csv'
        csv_path = os.path.join(dataset_save_csv_path, csv_name)
        data = pd.DataFrame(data_array)
        data.to_csv(csv_path) 
    elif model_kind ==3:
        csv_name = 'poisoned_' + saved_model_name + '_' + str(benign_csv_count) + '.csv'
        csv_path = os.path.join(dataset_save_csv_path, csv_name)
        data = pd.DataFrame(data_array)
        data.to_csv(csv_path)               

benign_csv_count = 0
agnostic_csv_count = 0
specific_csv_count = 0
for dataset in datasets:
    print()
    print(f'Dataset: {dataset}\n')
    model_arch = dataset_arch_dict[dataset]
    segment_layer_position = segment_layer_dict[model_arch]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]

    begin_count = 0
    benign_csv_count = 0
    for benign_model_id in range(200):
        benign_model_id += 1
        saved_model_file = f'{root}/{dataset}_models/{dataset}_{model_arch}_{benign_model_id}'
        saved_model_name = f'{dataset}_{model_arch}_{benign_model_id}'
        model_kind = 0          
        if dataset == 'dfb_mnist':
            saved_model_file = '/Data-free_Backdoor-main/saved_models/mnist'
            saved_model_name = 'dfb_mnist'
            model_kind = 3
        elif dataset == 'dfb_cifar10':
            saved_model_file = '/Data-free_Backdoor-main/saved_models/cifar10'
            saved_model_name = 'dfb_cifar10'
            model_kind = 3
        elif dataset == 'dfb_gtsrb':
            saved_model_file = '/Data-free_Backdoor-main/saved_models/gtsrb'
            saved_model_name = 'dfb_gtsrb'
            model_kind = 3
        elif dataset == 'dfb_imagenet_subset':
            saved_model_file = '/Data-free_Backdoor-main/saved_models/imagenet_subset'
            saved_model_name = 'dfb_imagenet_subset'
            model_kind = 3
        elif dataset == 'Narcissus_cifar10':
            saved_model_file = '/Narcissus-main/saved_models/cifar10'
            saved_model_name = 'Narcissus_cifar10'
            model_kind = 3
        elif dataset == 'Narcissus_mnist':
            saved_model_file = '/Narcissus-main/saved_models/mnist'
            saved_model_name = 'Narcissus_mnist'
            model_kind = 3
        elif dataset == 'Narcissus_gtsrb':
            saved_model_file = '/Narcissus-main/saved_models/gtsrb'
            saved_model_name = 'Narcissus_gtsrb'
            model_kind = 3
        elif dataset == 'Narcissus_imagenet_subset':
            saved_model_file = '/Narcissus-main/saved_models/imagenet_subset'
            saved_model_name = 'Narcissus_imagenet_subset'
            model_kind = 3
        elif dataset == 'input_mnist_one':
            saved_model_file = '/Beatrix-master/checkpoints/mnist/all2one'
            saved_model_name = 'input_mnist_one'
            model_kind = 3
        elif dataset == 'input_mnist_all':
            saved_model_file = '/Beatrix-master/checkpoints/mnist/all2all'
            saved_model_name = 'input_mnist_all'
            model_kind = 3
        elif dataset == 'input_gtsrb_one':
            saved_model_file = '/Beatrix-master/checkpoints/gtsrb/all2one'
            saved_model_name = 'input_gtsrb_one'
            model_kind = 3
        elif dataset == 'input_gtsrb_all':
            saved_model_file = '/Beatrix-master/checkpoints/gtsrb/all2all'
            saved_model_name = 'input_gtsrb_all'
            model_kind = 3
        elif dataset == 'input_cifar10_one':
            saved_model_file = '/Beatrix-master/checkpoints/cifar10/all2one'
            saved_model_name = 'input_cifar10_one'
            model_kind = 3
        elif dataset == 'input_cifar10_all':
            saved_model_file = '/Beatrix-master/checkpoints/cifar10/all2all'
            saved_model_name = 'input_cifar10_all'
            model_kind = 3
        elif dataset == 'input_imagenet_subset_one':
            saved_model_file = '/Beatrix-master/checkpoints/imagenet_subset/all2one'
            saved_model_name = 'input_imagenet_subset_one'
            model_kind = 3
        elif dataset == 'input_imagenet_subset_all':
            saved_model_file = '/Beatrix-master/checkpoints/imagenet_subset/all2all'
            saved_model_name = 'input_imagenet_subset_all'
            model_kind = 3
        elif dataset == 'composite_mnist_agnostic':
            saved_model_file = '/composite-attack-master/saved_models/mnist/class-agnostic'
            saved_model_name = 'composite_mnist_agnostic'
            model_kind = 3
        elif dataset == 'composite_mnist_specific':
            saved_model_file = '/composite-attack-master/saved_models/mnist/class-specific'
            saved_model_name = 'composite_mnist_specific'
            model_kind = 3
        elif dataset == 'composite_cifar10_agnostic':
            saved_model_file = '/composite-attack-master/saved_models/cifar10/class-agnostic'
            saved_model_name = 'composite_cifar10_agnostic'
            model_kind = 3
        elif dataset == 'composite_cifar10_specific':
            saved_model_file = '/composite-attack-master/saved_models/cifar10/class-specific'
            saved_model_name = 'composite_cifar10_specific'
            model_kind = 3
        elif dataset == 'composite_gtsrb_agnostic':
            saved_model_file = '/composite-attack-master/saved_models/gtsrb/class-agnostic'
            saved_model_name = 'composite_gtsrb_agnostic'
            model_kind = 3
            existing_num = 38
        elif dataset == 'composite_gtsrb_specific':
            saved_model_file = '/composite-attack-master/saved_models/gtsrb/class-specific'
            saved_model_name = 'composite_gtsrb_specific'
            model_kind = 3
            existing_num = 39
        elif dataset == 'composite_imagenet_subset_agnostic':
            saved_model_file = '/composite-attack-master/saved_models/imagenet_subset/class-agnostic'
            saved_model_name = 'composite_imagenet_subset_agnostic'
            model_kind = 3
        elif dataset == 'composite_imagenet_subset_specific':
            saved_model_file = '/composite-attack-master/saved_models/imagenet_subset/class-specific'
            saved_model_name = 'composite_imagenet_subset_specific'
            model_kind = 3
        if os.path.exists(saved_model_file):
            saved_models = os.listdir(saved_model_file)
            for saved_model in sorted(saved_models):
                if fnmatch.fnmatch(saved_model, saved_model_name_pattern):
                    benign_csv_count += 1
                    saved_model_path = os.path.join(saved_model_file, saved_model)
                    features, preds, model_latter = get_features(saved_model_path, model_arch, segment_layer_position, _n_cls, _size, args, other_loss_flag=1)    
                    _features, _preds, _model_latter = get_features(saved_model_path, model_arch, segment_layer_position, _n_cls, _size, args, other_loss_flag=0)
                    if args.feature_enhancement:
                        influence_features = feature_enhance(_features, features, args)
                    else:
                        influence_features = _features
                    affected_features = features
                    features_dominance = feature_influence(affected_features, influence_features, affected_features, model_latter, _n_cls)
                    record_to_csv(features_dominance, dataset, saved_model_name, model_kind=model_kind)
                    begin_count += 1
        if dataset not in ['mnist', 'cifar10', 'imagenet_subset', 'gtsrb']:
            break

for dataset in datasets:
    if dataset not in ['mnist', 'cifar10', 'imagenet_subset', 'gtsrb']:
        continue
    print()
    print(f'Dataset: {dataset}\n')
    REPEAT_ROUNDS_AGNOSTIC = dataset_re_ag_dict[dataset]
    REPEAT_ROUNDS_SPECIFIC = dataset_re_sp_dict[dataset]

    model_arch = dataset_arch_dict[dataset]
    segment_layer_position = segment_layer_dict[model_arch]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]

    poisoned_dataset = f'poisoned_{dataset}'

    for trigger_type in trigger_types:
        agnostic_csv_count = 0
        for targeted_class in range(_n_cls):
            target_count_id = 0
            saved_agnostic_poisoned_model_file = \
                f'{root}/{poisoned_dataset}_models/' \
                f'{poisoned_dataset}_{model_arch}' \
                f'_class-agnostic_targeted={targeted_class}' \
                f'_{trigger_type}-trigger' 
            saved_model_name = f'{poisoned_dataset}_{model_arch}' \
                f'_class-agnostic_targeted={targeted_class}' \
                f'_{trigger_type}-trigger' 
            if os.path.exists(saved_agnostic_poisoned_model_file):
                saved_models = os.listdir(saved_agnostic_poisoned_model_file)
                for saved_model in sorted(saved_models):
                    if fnmatch.fnmatch(saved_model, saved_model_name_pattern):
                        saved_model_path = os.path.join(saved_agnostic_poisoned_model_file, saved_model)
                        features, preds, model_latter = get_features(saved_model_path, model_arch, segment_layer_position, _n_cls, _size, args, other_loss_flag=1)             
                        if args.other_loss_coefficient == 0:
                            _features = copy.deepcopy(features)
                        else:
                            _features, _preds, _model_latter = get_features(saved_model_path, model_arch, segment_layer_position, _n_cls, _size, args, other_loss_flag=0)
                        if args.feature_enhancement:
                            influence_features = feature_enhance(_features, features, args)
                        else:
                            influence_features = _features
                        affected_features = features
                        features_dominance = feature_influence(affected_features, influence_features, affected_features, model_latter, _n_cls)
                        record_to_csv(features_dominance, dataset, saved_model_name, model_kind=1, target=targeted_class)
                        target_count_id += 1


        _specific_backdoor_targeted_classes = dataset_specific_backdoor_targeted_classes_dict[dataset]
        for _specific_backdoor_targeted_class in _specific_backdoor_targeted_classes:
            specific_csv_count = 0
            for _source_class in range(_n_cls):
                if _source_class != _specific_backdoor_targeted_class:
                    source_target_count_id = 0
                    saved_specific_poisoned_model_file = \
                        f'{root}/{poisoned_dataset}_models/' \
                        f'{poisoned_dataset}_{model_arch}' \
                        f'_class-specific_targeted={_specific_backdoor_targeted_class}_sources=[{_source_class}]' \
                        f'_{trigger_type}-trigger' 
                    saved_model_name = f'{poisoned_dataset}_{model_arch}' \
                        f'_class-specific_targeted={_specific_backdoor_targeted_class}_sources=[{_source_class}]' \
                        f'_{trigger_type}-trigger' 
                    if os.path.exists(saved_specific_poisoned_model_file):
                        saved_models = os.listdir(saved_specific_poisoned_model_file)
                        for saved_model in sorted(saved_models):
                            if fnmatch.fnmatch(saved_model, saved_model_name_pattern):
                                saved_model_path = os.path.join(saved_specific_poisoned_model_file, saved_model)
                                features, preds, model_latter = get_features(saved_model_path, model_arch, segment_layer_position, _n_cls, _size, args, other_loss_flag=1)
                                if args.other_loss_coefficient == 0:
                                    _features = copy.deepcopy(features)
                                else:
                                    _features, _preds, _model_latter = get_features(saved_model_path, model_arch, segment_layer_position, _n_cls, _size, args, other_loss_flag=0)
                                if args.feature_enhancement:
                                    influence_features = feature_enhance(_features, features, args)
                                else:
                                    influence_features = _features
                                affected_features = features
                                features_dominance = feature_influence(affected_features, influence_features, affected_features, model_latter, _n_cls)
                                record_to_csv(features_dominance, dataset, saved_model_name, model_kind=2, target=_specific_backdoor_targeted_class, source=_source_class)
                                source_target_count_id += 1


