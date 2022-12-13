#!/bin/bash

if [ ! -e /proc/$(pidof nvidia-smi) ]
then
	echo "nvidia-smi does not seem to be running. exiting job"
    exit 1
fi

source /etc/profile
module load anaconda/2022a

HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME=${HF_LOCAL_DIR}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED="true"

python ./py/train_GPT.py










