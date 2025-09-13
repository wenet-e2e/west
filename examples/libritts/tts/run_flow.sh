# Copyright 2025 Hao Yin(1049755192@qq.com)

[ ! -s west ] && ln -s ../../../west
[ ! -s tools ] && ln -s ../../../tools
export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES="2"  # Change this to all your available gpus, such as "0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

stage=train
data=data
dir=exp/touch_flow-Qwen2.5-0.5B-Audio-libritts
steps=50000  # training steps

if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    echo "Training..."
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_config_or_dir conf/touch_flow_config.json \
        --data_path $data/libritts_shards_all_train.list \
        --output_dir $dir \
        --batch_size 64 \
        --bf16 False \
        --max_steps $steps \
        --num_data_cycles 1000 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 100 \
        --learning_rate 3e-4 \
        --weight_decay 0.02 \
        --warmup_ratio 0.05 \
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
    echo "Decoding..."
    steps=checkpoint-${steps}
    mdir=$dir/${steps}
    adir=$(echo $mdir | sed 's:exp:exp_audio:g')
    mkdir -p $adir
    test_jsonl=$data/libritts/small.test.syn.flow.jsonl

    # flow inference
    python west/bin/tts_flow_inference.py \
        --model_dir $PWD/$mdir \
        --data_path $test_jsonl \
        --save_dir $adir/mel_outputs

    # vocoder inference
    python tools/vocoder.py \
        --mel_path $adir/mel_outputs \
        --output_path $adir/audio_outputs
    audio_dir=$(realpath $adir)/audio_outputs

    # prepare the wav.scp and gt.text file for compute wer.
    python tools/gen_tts_wavscp_text.py \
        $test_jsonl \
        $audio_dir \
        $adir/wav.scp \
        $adir/gt.jsonl

    # Compute WER
    python tools/whisper_asr.py $adir/wav.scp $adir/syn.jsonl
    python tools/compute_wer.py --char=1 --v=1 \
        $adir/gt.jsonl $adir/syn.jsonl > $adir/syn.wer

    # Compute speaker similarity
    python tools/compute_similarity.py $test_jsonl $adir/wav.scp $adir/syn.sim

    # Overall performance
    tail $adir/syn.wer
    tail -n1 $adir/syn.sim
fi