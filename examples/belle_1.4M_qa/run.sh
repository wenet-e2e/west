# Copyright 2025 Binbin Zhang(binbzha@qq.com)

[ ! -s west ] && ln -s ../../../west
[ ! -s tools ] && ln -s ../../../tools
export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES="1"  # Change this to all your available gpus, such as "0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

model_config_or_dir=pretrain_qwen1.7b_aishell_asr

stage=decode # data/train/decode
data=data

steps=10000  # training steps
pack_size=8192
lr_rate=5e-5
dir=exp/Qwe3-1.7B-Instruct-firered-${pack_size}-${lr_rate}

# Note: Change your model settings in `conf/touch_asu_config.json`


if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
    # TODO:
    mkdir $data
    cp -r /jfs-hdfs/user/Archive/AQA/qa_test/chinese_qa.jsonl $data
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_config_or_dir $model_config_or_dir \
        --data_path $data/data_tn_cn_messages_aishell.list \
        --output_dir $dir \
        --pack_size $pack_size \
        --bf16 True \
        --max_steps $steps \
        --num_data_cycles 100 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 100 \
        --learning_rate $lr_rate \
        --weight_decay 0.01 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.5 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --gradient_checkpointing \
        --dataloader_num_workers 2 \
        --dataloader_prefetch_factor 10 \
        --save_total_limit 10000 \
        --deepspeed conf/ds_config_zero2.json \
        --accelerator_config conf/accelerator_config.json
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
    mdir=$dir/checkpoint-${steps}
    python west/bin/decode.py \
        --data_path $data/chinese_qa.jsonl \
        --model_dir $mdir \
        --result_path $mdir/result.txt
    python tools/get_qa_hyp_ref_text.py $data/chinese_qa_messages.jsonl \
        $mdir/result.txt $mdir/result.json
    python tools/compute-acc-of-contain.py $mdir/result.json
fi
