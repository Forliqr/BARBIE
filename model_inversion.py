import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import Bottleneck, BasicBlock
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import seaborn as sns

from networks import partial_models_adaptive
from networks.BadEncoderOriginalModels.simclr_model import SimCLR, SimCLRBase
from networks.BadEncoderOriginalModels.nn_classifier import NeuralNet
from networks.BadEncoderOriginalModels import bad_encoder_full_model_partial

def model_segmentation(saved_model_path, model_arch, segment_layer_position, n_cls, input_size):
    print()
    print('Model Path: ')
    print(saved_model_path)
    print()
    print(f'The position of model segmentation: ', segment_layer_position)
    if 'resnet' in model_arch:
        if '50' in model_arch:
            layer_setting = [3, 4, 6, 3]
            block_setting = Bottleneck
        elif '18' in model_arch: 
            layer_setting = [3, 4, 6, 3]
            block_setting = BasicBlock
        else:
            raise NotImplementedError("Not implemented ResNet Setting!")
        model_latter = partial_models_adaptive.ResNetAdaptivePartialModel(
            num_classes=n_cls,
            segment_layer_position=segment_layer_position,
            original_input_img_shape=(1, 3, input_size, input_size),
            layer_setting=layer_setting,
            block_setting=block_setting
        )
    elif 'vgg16' in model_arch:
        model_latter = partial_models_adaptive.VGGAdaptivePartialModel(
            num_classes=n_cls, 
            segment_layer_position=segment_layer_position,
            original_input_img_shape=(1, 3, input_size, input_size)
        )
    elif 'google' in model_arch:
        model_latter = partial_models_adaptive.GoogLeNetAdaptivePartialModel(
            num_classes=n_cls,
            segment_layer_position=segment_layer_position,
            original_input_img_shape=(1, 3, input_size, input_size)
        )
    elif 'simple_cnn' in model_arch:
        if 'mnist' in saved_model_path:
            model_latter = partial_models_adaptive.SimpleCNNAdaptivePartialModel(
                original_input_img_shape=(1, 1, 28, 28),
                in_channels=1
            )
        else:
            model_latter = partial_models_adaptive.SimpleCNNAdaptivePartialModel()
    elif 'NeuralNet' in model_arch:
        if 'gtsrb' in saved_model_path:
            model_latter = partial_models_adaptive.NeuralNetAdaptivePartialModel(
                input_size=512,
                hidden_size_list=[512,256],
                num_classes=n_cls,
            )
        else:
            model_latter = partial_models_adaptive.NeuralNetAdaptivePartialModel(
                input_size=512,
                hidden_size_list=[512, 256],
                num_classes=n_cls,
            )
    elif 'encoder' in model_arch:
        # load the encoder
        bad_encoder_model = SimCLR()
        if 'badencoder' in model_arch:
            if ('cifar10' in model_arch):
                if ('clean' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/cifar10/clean_encoder/model_1000.pth', map_location='cpu')
                elif ('gtsrb_backdoored' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/cifar10/gtsrb_backdoored_encoder/model_200.pth', map_location='cpu')
                elif ('stl10_backdoored' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/cifar10/stl10_backdoored_encoder/model_200.pth', map_location='cpu')
                elif ('svhn_backdoored' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/cifar10/svhn_backdoored_encoder/model_200.pth', map_location='cpu')
            elif ('stl10' in model_arch):
                if ('clean' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/stl10/clean_encoder/model_1000.pth', map_location='cpu')
                elif ('gtsrb_backdoored' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/stl10/gtsrb_backdoored_encoder/model_ref_priority_num_0_epoch_200.pth', map_location='cpu')
                elif ('cifar10_backdoored' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/stl10/cifar10_backdoored_encoder/model_ref_airplane_num_0_epoch_200.pth', map_location='cpu')
                elif ('svhn_backdoored' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/stl10/svhn_backdoored_encoder/model_ref_one_num_0_epoch_200.pth', map_location='cpu')            
            bad_encoder_model.load_state_dict(bad_encoder_ckpt['state_dict'])
        elif 'drupe' in model_arch:
            if ('cifar10' in model_arch):
                if ('clean' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/cifar10/clean_encoder/model_1000.pth', map_location='cpu')
                elif ('gtsrb' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/DRUPE-main/saved_models_encoder/shadow_cifar10_downstream_gtsrb/encoder_num_1.pth', map_location='cpu')
                elif ('stl10' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/DRUPE-main/saved_models_encoder/shadow_cifar10_downstream_stl10/encoder_num_1.pth', map_location='cpu')
                elif ('svhn' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/DRUPE-main/saved_models_encoder/shadow_cifar10_downstream_svhn/encoder_num_2.pth', map_location='cpu')
            elif ('stl10' in model_arch):
                if ('clean' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/BadEncoder-main/output/stl10/clean_encoder/model_1000.pth', map_location='cpu')
                elif ('gtsrb' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/DRUPE-main/saved_models_encoder/shadow_stl10_downstream_gtsrb/encoder_num_1.pth', map_location='cpu')
                elif ('cifar10' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/DRUPE-main/saved_models_encoder/shadow_stl10_downstream_cifar10/encoder_num_1.pth', map_location='cpu')
                elif ('svhn' in saved_model_path):
                    bad_encoder_ckpt = torch.load('/DRUPE-main/saved_models_encoder/shadow_stl10_downstream_svhn/encoder_num_1.pth', map_location='cpu')            
            bad_encoder_model.load_state_dict(bad_encoder_ckpt['state_dict'])
        # load the classifier
        classifier_in_bad_encoder = NeuralNet(512, [512, 256], n_cls)
        cls_ckpt = torch.load(saved_model_path, map_location='cpu')
        if ('badencoder' in model_arch):
            classifier_in_bad_encoder.load_state_dict(cls_ckpt['state_dict'])
        elif ('drupe' in model_arch):
            classifier_in_bad_encoder.load_state_dict(cls_ckpt)
        model_latter = bad_encoder_full_model_partial.BadEncoderFullModelAdaptivePartialModel(
            encoder=bad_encoder_model,
            classifier=classifier_in_bad_encoder,
            inspect_layer_position=segment_layer_position,
            original_input_img_shape=(1, 3, input_size, input_size)
        )
    else:
        raise NotImplementedError('Model not supported!')
    
    if 'encoder' not in model_arch:
        ckpt = torch.load(saved_model_path, map_location='cpu')
        if ('Troj' not in saved_model_path) and ('flareon' not in saved_model_path) and ('DRUPE' not in saved_model_path) and ('Data-free_Backdoor' not in saved_model_path) and ('Narcissus' not in saved_model_path) and ('composite' not in saved_model_path):
            try:
                state_dict = ckpt['net_state_dict']
            except KeyError:
                try:
                    state_dict = ckpt['model']
                except KeyError:
                    try:
                        state_dict = ckpt['state_dict']
                    except KeyError:
                        state_dict = ckpt['netC']                    
        else:
            if ('flareon' in saved_model_path) or ('DRUPE' in saved_model_path) or ('Data-free_Backdoor' in saved_model_path) or ('Narcissus' in saved_model_path) or ('composite' in saved_model_path):
                model_latter.load_state_dict(ckpt)
            else:
                model_latter = ckpt
            
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_latter = torch.nn.DataParallel(model_latter)
        model_latter = model_latter.cuda()
        cudnn.benchmark = True
        if ('Troj' not in saved_model_path) and ('flareon' not in saved_model_path) and ('DRUPE' not in saved_model_path) and ('Data-free_Backdoor' not in saved_model_path) and ('Narcissus' not in saved_model_path) and ('composite' not in saved_model_path) and ('encoder' not in model_arch):
            model_latter.load_state_dict(state_dict)

    return model_latter

def compute_feature(model_latter, feature_shape_tensor, other_loss_coefficient, bound_on, class_id, n_cls, other_loss_flag):
    model_latter.eval()
    label = torch.tensor([class_id])

    feature_tensor = torch.rand_like(feature_shape_tensor)
    feature_tensor.requires_grad = True

    criterion_adversarial = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model_latter = model_latter.cuda()
        label = label.cuda()
        feature_tensor = feature_tensor.cuda()
        cudnn.benchmark = True

    optimizer_adversarial_perturb = torch.optim.Adam([feature_tensor], lr=1e-2,
                                                     weight_decay=0.005)  # scale of L2 norm

    for iters in range(1000):
        optimizer_adversarial_perturb.zero_grad()
        _pred = model_latter(feature_tensor)
        loss_other = 0
        if other_loss_flag and other_loss_coefficient != 0:
            for class_i in range(n_cls):
                if class_i != class_id:
                    label_other = torch.tensor([class_i])
                    if torch.cuda.is_available():
                        label_other = label_other.cuda()
                    loss_other += other_loss_coefficient * criterion_adversarial(_pred, label_other)
        loss_other = loss_other / (n_cls-1)
        loss_adversarial_perturb = criterion_adversarial(_pred, label) - loss_other
        loss_adversarial_perturb.backward()

        optimizer_adversarial_perturb.step()

        if bound_on:
            with torch.no_grad():
                feature_tensor.clamp_(0., 999.)

    return feature_tensor

def compute_features(model_arch, model_latter, segment_layer_position, other_loss_coefficient, feature_repetition, n_cls, other_loss_flag):
    features = [[] for i in range(n_cls)]

    model_latter.eval()
    try:
        feature_shape = model_latter.input_shapes[segment_layer_position]
    except IndexError:
        feature_shape = model_latter.input_shapes[1]
    feature_shape_tensor = torch.ones(size = feature_shape)

    if torch.cuda.is_available():
        feature_shape_tensor = feature_shape_tensor.cuda()

    if ('resnet' in model_arch and segment_layer_position >= 1) \
            or ('vgg16' in model_arch and segment_layer_position >= 2) \
            or ('google' in model_arch and segment_layer_position >= 1)\
            or ('cnn' in model_arch and segment_layer_position >= 1):
        bound_on = True
    else:
        bound_on = False

    print('Feature reconstruction in progress!\n')
    with tqdm(total=n_cls) as t:
        for class_id in range(n_cls):
            for feature_repetition_i in range(feature_repetition):
                t.set_postfix(class_id=class_id, No_=f"{feature_repetition_i}/{feature_repetition}")
                feature = compute_feature(model_latter, feature_shape_tensor, other_loss_coefficient, bound_on, class_id, n_cls, other_loss_flag)
                features[class_id].append(feature)
            t.update(1)
    return features

def compute_pred_for_avg_feature(model_latter, features_i, args):
    feature_sum = torch.zeros_like(features_i[0])
    for feature in features_i:
        feature_sum += feature
    avg_feature = feature_sum / args.feature_repetition
    _logits = model_latter(avg_feature)
    _scores = F.softmax(_logits, dim=1)
    _logits = _logits.detach().cpu().numpy()[0]
    _scores = _scores.detach().cpu().numpy()[0]
    if args.metric == 'softmax_score':
        return _scores
    elif args.metric == 'logit':
        return _logits

def compute_metrics_for_array(value_array):
    _a_flat = value_array.flatten()
    _a_flat = _a_flat[_a_flat != 0.]

    _a_flat = np.sort(_a_flat)
    _length = len(_a_flat)
    q1_pos = int(0.25 * _length)
    q3_pos = int(0.75 * _length)
    _q1 = _a_flat[q1_pos]
    _q3 = _a_flat[q3_pos]
    _iqr = _q3 - _q1
    _anomaly_metric = (np.max(_a_flat) - _q3) / _iqr

    return _anomaly_metric

def draw_diagram(diagram_values, n_cls, saved_model_path, model_arch, segment_layer_position):
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(5, 4, figure=fig)
    ax = fig.add_subplot(gs[:4, :])
    mat_show = ax.matshow(diagram_values)

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    ax.set_ylabel('source classes', fontsize=40)
    ax.set_title('target classes', fontsize=40)

    ax2 = fig.add_subplot(gs[4, :])

    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.08, right=0.99, top=0.92, bottom=0.04)

    tendency_every_target_class = np.average(diagram_values, axis=0) * n_cls / (n_cls - 1)
    sns.boxplot(data=tendency_every_target_class, orient='h', ax=ax2,
                fliersize=25
                # flierprops={'marker': 'o',
                #             'markerfacecolor': 'red',
                #             'color': 'black',
                #             },
                )

    _anomaly_metric = compute_metrics_for_array(tendency_every_target_class)

    id_str_start = saved_model_path.rfind('models/') + len('models/')
    id_str_end = saved_model_path.find('/last')
    id_str = saved_model_path[id_str_start:id_str_end]

    if 'equ_pos_on' in saved_model_path:
        id_str = 'equ_pos_on_' + id_str
    elif 'bad_encoder' in model_arch:
        id_str = 'bad_encoder_' + id_str

    id_str += f'({_anomaly_metric:.3f})'

    if segment_layer_position is not None:
        id_str += f'(Ldef_id={segment_layer_position})'

    print(f'id_str:{id_str}')

    plt.savefig(f'./segment_results/SegmentResult--{id_str}.png')
    plt.close()


def get_features(saved_model_path, model_arch, segment_layer_position, n_cls, input_size, args, other_loss_flag=1):
    model_latter = model_segmentation(saved_model_path, model_arch, segment_layer_position, n_cls, input_size)
    model_latter = model_latter.eval()
    features = compute_features(model_arch, model_latter, segment_layer_position, args.other_loss_coefficient, args.feature_repetition, n_cls, other_loss_flag)
    feature_preds = np.zeros(shape=(n_cls, n_cls))
    for class_id in range(n_cls):
        features_i = features[class_id]
        feature_pred = compute_pred_for_avg_feature(model_latter, features_i, args)
        for feature_pred_id in range(len(feature_pred)):
            if feature_pred[feature_pred_id] >= 0:
                feature_preds[class_id][feature_pred_id] = feature_pred[feature_pred_id]
            else:
                feature_preds[class_id][feature_pred_id] = 0

    if args.use_transpose_correction:
        correction_matrix = feature_preds / feature_preds.transpose()
        for x in range(n_cls):
            for y in range(n_cls):
                if x == y:
                    correction_matrix[x][y] = 0.
                if correction_matrix[x][y] > 1.0:
                    correction_matrix[x][y] = correction_matrix[y][x]
        feature_preds *= (1 - correction_matrix)

    # draw_diagram(feature_preds, n_cls, saved_model_path, model_arch, segment_layer_position)

    return features, feature_preds, model_latter
