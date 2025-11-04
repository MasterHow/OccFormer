#!/bin/bash

# H3O数据集测试脚本
# 使用方法: bash test_h3o.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 数据集路径 - 请根据实际情况修改
H3O_DATA_ROOT="/path/to/H3O/dataset"
H3O_ANN_FILE="/path/to/H3O/dataset"

# 配置文件路径
CONFIG_FILE="projects/configs/occformer_h3o/occformer_h3o.py"

# 模型检查点路径
CHECKPOINT="work_dirs/occformer_h3o/latest.pth"

# 工作目录
WORK_DIR="work_dirs/occformer_h3o"

# 测试参数
BATCH_SIZE=1
NUM_WORKERS=4

# 创建测试命令
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --work-dir ${WORK_DIR} \
    --data-root ${H3O_DATA_ROOT} \
    --ann-file ${H3O_ANN_FILE} \
    --cfg-options \
        data.samples_per_gpu=${BATCH_SIZE} \
        data.workers_per_gpu=${NUM_WORKERS} \
    --eval mIoU \
    --show-dir ${WORK_DIR}/results \
    --gpus 1

echo "H3O数据集测试完成！"
