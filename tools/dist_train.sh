#!/usr/bin/env bash

# 检查输入参数的数量
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG GPUS [OPTIONS...]"
    exit 1
fi

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

# 打印用于调试的信息
echo "Config file: $CONFIG"
echo "GPUs: $GPUS"
echo "Port: $PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch "${@:3}" --deterministic

##!/usr/bin/env bash
#CONFIG=$1
#GPUS=$2
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29500}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/train.py \
#    $CONFIG \
#    --seed 0 \
#    --launcher pytorch ${@:3}

# #!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-28509}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
