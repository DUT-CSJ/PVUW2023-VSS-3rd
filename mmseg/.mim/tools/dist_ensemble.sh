CONFIG=$1
CHECKPOINT=$2
GPUSNUM=$3
OUT=@4
GPUS=@5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUSNUM \
    --master_port=$PORT \
    $(dirname "$0")/model_ensemble.py \
    $CONFIG \
    $CHECKPOINT \
    $OUT \
    $GPUS \
    --launcher pytorch \
    ${@:4}
