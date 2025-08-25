# Copyright 2025 Binbin Zhang(binbzha@qq.com)

[ ! -s west ] && ln -s ../../../west
[ ! -s tools ] && ln -s ../../../tools
export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES="0"  # Change this to all your available gpus, such as "0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

llm=/bucket/output/jfs-hdfs/user/binbin.zhang/huggingface/hub/Qwen2-1.5B-Instruct
speech_encoder=/jfs-hdfs/user/binbin.zhang/models/wenet/wenetspeech/u2pp_conformer/

stage=train # data/train/decode
data=data
dir=exp/Qwen-1.5B-Instruct-wenetspeech-encoder_pack8192
steps=2000  # training steps

# Note: Change your model settings in `conf/touch_asu_config.json`


if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_config_path conf/touch_asu_config.json \
        --data_path $data/train.jsonl \
        --output_dir $dir \
        --pack_size 8192 \
        --bf16 True \
        --max_steps $steps \
        --num_data_cycles 1000 \
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
        --save_total_limit 10000 \
        --deepspeed conf/ds_config_zero2.json \
        --accelerator_config conf/accelerator_config.json
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
    mdir=$dir/checkpoint-${steps}
    python west/bin/decode.py \
        --data_path $data/test.jsonl \
        --model_dir $PWD/$mdir \
        --result_path $mdir/result.txt
    paste <(awk '{print $1}' $data/test.text) $mdir/result.txt > $mdir/result.hyp
    python tools/compute-wer.py --char=1 --v=1 \
        $data/test.text $mdir/result.hyp > $mdir/result.wer
fi
