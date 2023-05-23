# YAML=$1
# ROOT=./yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --yaml=./yaml/$YAML

python3 train.py --yaml=./yaml/clear10/clear10_feature_res50_moco_public_private.yaml