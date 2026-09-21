#!/bin/sh
#SBATCH -J GLMCSEImageNetMaxNorm
#SBATCH -o output/log.out.%j
#SBATCH -e output/log.err.%j
#SBATCH --partition=gpu-a10
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
conda activate LTR
# python main.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 120 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
# python mainIni.py --dataset iNaturelist2018 -a resnext50_32x4d --num_classes 8142 --beta 0.5 --lr 0.1 --epochs 120 -b 128 --momentum 0.9 --weight_decay 1e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
python main.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4
# python mainIni.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 120 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
# python mainIni.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.001 --epochs 10 -b 120 --momentum 0.9 --weight_decay 0.1 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10