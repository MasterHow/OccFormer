#!/usr/bin/env bash

# 检查输入参数的数量
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG GPUS [OPTIONS...]"
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

# 打印用于调试的信息
echo "Config file: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $GPUS"
echo "Port: $PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch "${@:4}" --deterministic --eval bbox

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3
#PORT=${PORT:-29503}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
