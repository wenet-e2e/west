# Copyright 2025 Chengdong Liang(liangchengdongd@qq.com)

[ ! -s west ] && ln -s ../../west
[ ! -s tools ] && ln -s ../../tools
export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Change this to all your available gpus, such as "0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

model_config_or_dir=pretrained_llm_asr_model

stage=decode # data/train/decode
data=data

steps=20000  # training steps
pack_size=25000
lr_rate=5e-5
dir=exp/Qwe3-1.7B-Instruct-firered-${pack_size}-${lr_rate}-QA

# Note: Change your model settings in `conf/touch_asu_config.json`


if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
    # TODO:
    mkdir $data
    cp -r /data/to//train_aishell2_shuffle_train_belle_1.4M.list $data/data.list
    cp -r /data/to/chinese_qa.jsonl $data
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_config_or_dir $model_config_or_dir \
        --data_path $data/data.list \
        --output_dir $dir \
        --pack_size $pack_size \
        --bf16 True \
        --max_steps $steps \
        --num_data_cycles 100 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 100 \
        --learning_rate $lr_rate \
        --weight_decay 0.01 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --gradient_checkpointing \
        --dataloader_num_workers 4 \
        --dataloader_prefetch_factor 10 \
        --save_total_limit 10000 \
        --deepspeed conf/ds_config_zero2.json \
        --accelerator_config conf/accelerator_config.json
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
    mdir=$dir/checkpoint-${steps}
    cp conf/generation_config.json $mdir
    python west/bin/decode.py \
        --data_path $data/chinese_qa.jsonl \
        --model_dir $mdir \
        --result_path $mdir/result.jsonl
    python tools/get_qa_hyp_ref_text.py $data/chinese_qa.jsonl \
        $mdir/result.jsonl $mdir/result_hyp_ref.json
    python tools/compute_acc_of_contain.py $mdir/result_hyp_ref.json
fi
