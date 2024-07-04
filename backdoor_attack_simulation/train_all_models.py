'''
Use this script to train backdoored and benign models.
'''
import sys
import os
from backdoor_inspection_new import *

root = '/BARBIE/saved_models'
opt = parse_option()

# generate paths of saved model files
datasets = ['mnist', 'gtsrb', 'imagenet_subset', 'cifar10']

dataset_re_ag_dict = {'imagenet_subset': 20, 'cifar10': 20, 'gtsrb': 5, 'mnist': 20}
dataset_re_sp_dict = {'imagenet_subset': 6, 'cifar10': 8, 'gtsrb': 3, 'mnist': 8}

dataset_arch_dict = {'imagenet_subset': 'resnet50', 'cifar10': 'vgg16', 'gtsrb': 'google_net', 'mnist': 'simple_cnn'}
dataset_ncls_dict = {'imagenet_subset': 10, 'cifar10': 10, 'gtsrb': 43, 'mnist': 10}
dataset_size_dict = {'imagenet_subset': 224, 'cifar10': 32, 'gtsrb': 32, 'mnist': 28}
dataset_specific_backdoor_targeted_classes_dict = {'imagenet_subset': [0, 6, 7, 9],
                                                   'cifar10': [1, 2, 3],
                                                   'gtsrb': [7, 8],
                                                   'mnist': [6, 7, 8]}

dataset_root_dict = {'cifar10': '/BARBIE/datasets/CIFAR10',
                     'gtsrb': '/BARBIE/datasets/GTSRB',
                     'imagenet_subset': '/BARBIE/datasets/imagenette2-160',
                     'mnist': '/BARBIE/datasets/'
                     }

dataset_example_path_dict = \
    {'cifar10': '/BARBIE/saved_models/cifar10_models/cifar10_vgg16/last.pth',
     'gtsrb': '/BARBIE/saved_models/gtsrb_models/gtsrb_google_net/last.pth',}

poison_ratio_specific_dict = {'imagenet_subset': 0.025, 'cifar10': 0.05, 'gtsrb': 0.0125, 'mnist': 0.05}

trigger_types = ['patched_img', 'blending_img', 'filter_img']

dataset = 'cifar10'
model_arch = dataset_arch_dict[dataset]
data_folder = dataset_root_dict[dataset]
clean_exp_path = dataset_example_path_dict[dataset]

# train benign models
for dataset in datasets:
    model_arch = dataset_arch_dict[dataset]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]
    _root = dataset_root_dict[dataset]

    existing_benign_model_num = 0
    if not os.path.exists(f'{root}/{dataset}_models/'):
        os.makedirs(os.path.abspath(f'{root}/{dataset}_models/'))
    for saved_model_dir in os.listdir(f'{root}/{dataset}_models/'):
        saved_model_file = f'{root}/{dataset}_models/{saved_model_dir}/last.pth'
        if os.path.exists(saved_model_file):
            existing_benign_model_num += 1

    set_benign_model_num = 200
    while existing_benign_model_num < set_benign_model_num:
        command_str = f'python /BARBIE/backdoor_attack_simulation/model_training.py ' \
                      f'--dataset {dataset} ' \
                      f'--model {model_arch} ' \
                      f'--data_folder {_root} ' 
        train_process_status = os.system(command_str)
        print(f'Training status exited with status: {train_process_status}.\n')
        existing_benign_model_num += 1

# train poisoned models
for dataset in datasets:
    REPEAT_ROUNDS_AGNOSTIC = dataset_re_ag_dict[dataset]
    REPEAT_ROUNDS_SPECIFIC = dataset_re_sp_dict[dataset]
    model_arch = dataset_arch_dict[dataset]
    _n_cls = dataset_ncls_dict[dataset]
    _size = dataset_size_dict[dataset]
    _root = dataset_root_dict[dataset]

    _poison_ratio_specific = poison_ratio_specific_dict[dataset]

    poisoned_dataset = f'poisoned_{dataset}'
    for trigger_type in trigger_types:
        # class agnostic backdoor
        for repeat_round_id in range(REPEAT_ROUNDS_AGNOSTIC):
            for targeted_class in range(_n_cls):
                command_str = f'python /BARBIE/backdoor_attack_simulation/model_training.py ' \
                              f'--dataset {poisoned_dataset} ' \
                              f'--model {model_arch} ' \
                              f'--data_folder {_root} ' \
                              f'--targeted_class {targeted_class} ' \
                              f'--trigger_type {trigger_type} ' \
                              f'--my_marker {repeat_round_id}'
                train_process_status = os.system(command_str)
                print(f'Training status exited with status: {train_process_status}.\n')

        # class specific backdoor
        for repeat_round_id in range(REPEAT_ROUNDS_SPECIFIC):
            _specific_backdoor_targeted_classes = dataset_specific_backdoor_targeted_classes_dict[dataset]
            for _specific_backdoor_targeted_class in _specific_backdoor_targeted_classes:
                for _source_class in range(_n_cls):
                    if _source_class != _specific_backdoor_targeted_class:
                        command_str = f'python /BARBIE/backdoor_attack_simulation/model_training.py ' \
                                      f'--dataset {poisoned_dataset} ' \
                                      f'--model {model_arch} ' \
                                      f'--data_folder {_root} ' \
                                      f'--targeted_class {_specific_backdoor_targeted_class} ' \
                                      f'--source_classes {_source_class} ' \
                                      f'--poison_ratio {_poison_ratio_specific} ' \
                                      f'--trigger_type {trigger_type} ' \
                                      f'--my_marker {repeat_round_id}'
                        train_process_status = os.system(command_str)
                        print(f'Training status exited with status: {train_process_status}.\n')
