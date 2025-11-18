# Copyright 2025 Chengdong Liang(liangchengdongd@qq.com)

# This script is used to fine-tune [OSUM](https://huggingface.co/ASLP-lab/OSUM).

[ ! -s west ] && ln -s ../../../west
[ ! -s tools ] && ln -s ../../../tools
export PYTHONPATH=$PYTHONPATH:$PWD
# Change this to all your available gpus, such as "0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

stage=train # data/convert_model/train/decode
data=data
osum_model_path=/path/to/osum/infer.pt
dir=exp/OSUM-finetune
steps=5000  # training steps
# prompt config: https://github.com/ASLP-lab/OSUM/blob/main/OSUM/conf/prompt_config.yaml
osum_prompt_config=/path/to/osum/prompt_config.json

. tools/parse_options.sh

if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
fi

if [ $stage == "convert_model" ] || [ $stage == "all" ]; then
    echo "Convert osum model to west style"
    python tools/convert_scripts/convert_osum.py \
        --osum_model_path $osum_model_path \
        --llm_model_dir Qwen/Qwen2-7B-Instruct/ \
        --wenet_model_dir whiper-medium \
        --prompt_config_path $osum_prompt_config \
        --output_dir $dir/osum-west-test
fi

if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_config_or_dir $dir/osum-west \
        --data_path $data/train.jsonl \
        --output_dir $dir \
        --pack_size 8192 \
        --bf16 True \
        --max_steps $steps \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 100 \
        --learning_rate 3e-4 \
        --weight_decay 0.01 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.5 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --gradient_checkpointing \
        --dataloader_num_workers 2 \
        --dataloader_prefetch_factor 10 \
        --ignore_data_skip True \
        --deepspeed conf/ds_config_zero2.json \
        --accelerator_config conf/accelerator_config.json
fi

if [ $stage == "decode" ] || [ $stage == "all" ]; then
    # mdir=$dir/checkpoint-${steps}
    mdir=$dir/osum-west
    python west/bin/decode.py \
        --data_path $data/aishell2_test/test.jsonl \
        --model_dir $mdir \
        --result_path $mdir/result.jsonl
    python tools/compute_wer.py --char=1 --v=1 \
        $data/test.jsonl $mdir/result.jsonl > $mdir/result.wer
fi
