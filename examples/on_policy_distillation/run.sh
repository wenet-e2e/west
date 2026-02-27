#!/bin/bash

project_dir=$(pwd)/../../
[ ! -s west ] && ln -s $project_dir/west
[ ! -s tools ] && ln -s $project_dir/tools
export PYTHONPATH=$PYTHONPATH:$PWD


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

stage=train

. tools/parse_options.sh

deepspeed_config=conf/ds_zero1.json
run_name=on_policy_distillation_qwen_omni_3b
prompt_template=caption
dir=exp/${run_name}

model_name_or_path=${MODEL_NAME_OR_PATH:-models/Qwen2.5-Omni-3B}
teacher_model_name_or_path=${TEACHER_MODEL_NAME_OR_PATH:-models/Qwen3-Omni-30B-A3B-Captioner}
avqa_hf_dataset_path=${AVQA_HF_DATASET_PATH:-data/avqa-processed}
mmau_test_mini_data_dir=${MMAU_TEST_MINI_DATA_DIR:-data/MMAU} # This path is hardcoded in the scripts/download_mmau_test.sh.

if [ $stage == "prepare" ]; then
    echo "Prepare required data and models"
    huggingface-cli download gijs/avqa-processed --local-dir ${avqa_hf_dataset_path} --repo-type dataset

    huggingface-cli download Qwen/Qwen2.5-Omni-3B --local-dir ${model_name_or_path}
    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Captioner --local-dir ${teacher_model_name_or_path}

    bash scripts/download_mmau_test.sh
fi

if [ $stage == "vllm_teacher" ]; then
    # on another machine
    vllm serve ./Qwen3-Omni-30B-A3B-Captioner \
        --served-model-name Qwen3-Omni-30B-A3B-Captioner \
        --port 9999 \
        --enable-log-requests \
        --max-logprobs 128 \
        --enable-log-outputs \
        --max-num-seqs 32 \
        --tensor-parallel-size 4 \
        --trust-remote-code
fi

if [ $stage == "train" ]; then
    teacher_model_name_or_path=http://your-teacher-machine-ip:9999/v1
    torchrun --nproc_per_node=${num_gpus} \
        --nnodes=1 \
        --node-rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=32778 \
        west/bin/train_knowledge_distillation.py \
        --deepspeed ${deepspeed_config} \
        --model_name_or_path ${model_name_or_path} \
        --teacher_model_name_or_path ${teacher_model_name_or_path} \
        --output_dir ${dir} \
        --hf_dataset_path ${avqa_hf_dataset_path} \
        --run_name ${run_name} \
        --template ${prompt_template} \
        --save_steps 100 \
        --num_generations 1 \
        --use_wandb true || exit 1
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LLM_API_KEY=sk-**** # TODO: replace with your own API key
if [ $stage == "decode" ]; then
    iters=(100 200 300)
    temperature=0.0
    batch_size=32
    for iter in ${iters[*]}; do
        model_dir=${dir}/checkpoint-${iter}
        out_dir=${dir}/caption_test_mini_checkpoint_${iter}_temperature_${temperature}
        mkdir -p ${out_dir}

        python3 west/bin/decode_mmau.py \
        --model_path ${model_dir} \
        --data_file ${mmau_test_mini_data_dir}/mmau-test-mini.json \
        --audio_dir ${mmau_test_mini_data_dir} \
        --out_file ${out_dir}/caption_mmau_mini.json \
        --template ${prompt_template} \
        --temperature ${temperature} \
        --batch_size ${batch_size} || exit 1

        python3 cascaded_audio_caption_llm_eval.py \
        --input_file ${out_dir}/caption_mmau_mini.json \
        --output_file ${out_dir}/res_mmau_mini.json \
        --temperature ${temperature} \
        --api_key ${LLM_API_KEY} \
        --max_tokens 256 || exit 1

        python3 ${mmau_test_mini_data_dir}/evaluation.py \
        --input ${out_dir}/res_mmau_mini.json \
        > ${out_dir}/eval_mmau_mini.txt || exit 1
    done
fi
