#!/bin/bash

# H3O数据集训练脚本
# 使用方法: bash train_h3o.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 数据集路径 - 请根据实际情况修改
H3O_DATA_ROOT="/path/to/H3O/dataset"
H3O_ANN_FILE="/path/to/H3O/dataset"

# 配置文件路径
CONFIG_FILE="projects/configs/occformer_h3o/occformer_h3o.py"

# 工作目录
WORK_DIR="work_dirs/occformer_h3o"

# 训练参数
BATCH_SIZE=4
NUM_WORKERS=8
EPOCHS=30
LR=0.0001

# 创建训练命令
python tools/train.py \
    ${CONFIG_FILE} \
    --work-dir ${WORK_DIR} \
    --data-root ${H3O_DATA_ROOT} \
    --ann-file ${H3O_ANN_FILE} \
    --cfg-options \
        data.samples_per_gpu=${BATCH_SIZE} \
        data.workers_per_gpu=${NUM_WORKERS} \
        runner.max_epochs=${EPOCHS} \
        optimizer.lr=${LR} \
    --validate \
    --gpus 1 \
    --seed 0

echo "H3O数据集训练完成！"
