# Data-Free-Backdoor-Detection
Code of the paper *Barbie: Robust Backdoor Detection Based on Latent Separability*.

## Step 1: Train benign and backdoored models

Use *backdoor_attack_simulation/train_all_models.py* to train benign and backdoored models, including models backdoored with the agnostic/specific backdoor, the patch/blending/filter trigger.

python ./backdoor_attack_simulation/train_all_models.py

To train models backdoored with the composite backdoor, you can use the implementation for the composite backdoor attack at: https://github.com/TemporaryAcc0unt/composite-attack

To train models backdoored with the input-aware dynamic backdoor, you can use the implementation for the input-aware dynamic backdoor attack at: https://github.com/VinAIResearch/input-aware-backdoor-attack-release

To train models backdoored with the narcissus backdoor, you can use the implementation for the narcissus backdoor attack at: https://github.com/ruoxi-jia-group/narcissus-backdoor-attack

To train models backdoored with the data-free backdoor, you can use the implementation for the data-free trojan injection approach attack at: https://github.com/lvpeizhuo/Data-free_Backdoor

## Step 2: Generate the relative competitive score for each model  

python -u relative_competitve_score.py --feature_enhancement 0.0 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name mnist

python -u relative_competitve_score.py --feature_enhancement 0.0 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name cifar10

python -u relative_competitve_score.py --feature_enhancement 0.1 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name imagenet_subset

python -u relative_competitve_score.py --feature_enhancement 0.1 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name gtsrb

## Step 3: Calculate indicators and carry out detection

python indicator_calculation_and_detection.py --feature_enhancement 0.0 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name mnist

python indicator_calculation_and_detection.py --feature_enhancement 0.0 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name cifar10

python indicator_calculation_and_detection.py --feature_enhancement 0.1 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name imagenet_subset

python indicator_calculation_and_detection.py --feature_enhancement 0.1 --feature_enhancement_filter 1.0 --other_loss_coefficient 0.5 --dataset_name gtsrb
