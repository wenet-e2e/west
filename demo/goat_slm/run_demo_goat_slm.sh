#!/bin/bash

source goat_slm_env/bin/activate || exit 1

export PYTHONPATH=$PYTHONPATH:$PWD/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:goat_slm_env/lib/python3.12/site-packages/onnxruntime/capi


model_dir='pretrained_models/GOAT-SLM'
CUDA_VISIBLE_DEVICES=0 python3 demo_goat_slm.py \
    --slm_model_path $model_dir \
    --max_new_tokens 256 \
    --port 8085 \
    --max_turn_num 6 \
    --is_speech_generate_run
