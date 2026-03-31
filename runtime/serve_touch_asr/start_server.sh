#!/bin/bash
# ==============================================
# Qwen3-Omni ASR vLLM 服务启动脚本
# ==============================================

# 配置（通过环境变量传入，见 start_server.local.sh）
model_path="${MODEL_PATH:?请设置 MODEL_PATH 环境变量，指向模型目录}"
host="0.0.0.0"
port=8001
gpu_ids="0,1,2,3"
tp_size=4
gpu_mem_util=0.9

# 解析参数
save_audio_flag="--save-audio"  # 默认开启，传 --no-save-audio 关闭
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            model_path="$2"
            shift 2
            ;;
        --port|-p)
            port="$2"
            shift 2
            ;;
        --gpu-ids|-g)
            gpu_ids="$2"
            shift 2
            ;;
        --tp-size|-tp)
            tp_size="$2"
            shift 2
            ;;
        --no-save-audio)
            save_audio_flag=""
            shift 1
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model, -m       模型路径 (默认: \$MODEL_PATH)"
            echo "  --port, -p        服务端口 (默认: 8001)"
            echo "  --gpu-ids, -g     GPU ID (默认: 0,1,2,3)"
            echo "  --tp-size, -tp    Tensor 并行数 (默认: 4)"
            echo "  --no-save-audio   关闭保存音频功能 (默认开启)"
            echo ""
            echo "示例:"
            echo "  # 单 GPU 启动"
            echo "  $0 --gpu-ids 0 --tp-size 1"
            echo ""
            echo "  # 多 GPU 启动 (tensor parallel)"
            echo "  $0 --gpu-ids 0,1,2,3 --tp-size 4"
            echo ""
            echo "  # 关闭保存音频"
            echo "  $0 --no-save-audio"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "Qwen3-Omni ASR vLLM 服务"
echo "=================================================="
echo "模型路径: ${model_path}"
echo "服务地址: ${host}:${port}"
echo "GPU IDs:  ${gpu_ids}"
echo "TP Size:  ${tp_size}"
echo "=================================================="

# 进入目录
cd "$(dirname "$0")"

# 定义清理函数（重入保护：只执行一次）
_cleaning=0
cleanup() {
    [ "$_cleaning" -ne 0 ] && return
    _cleaning=1
    trap - SIGINT SIGTERM

    echo ""
    echo "=================================================="
    echo "收到退出信号 (Ctrl+C / 进程终止)"
    echo "正在清理资源 (端口、GPU 进程)..."
    echo "=================================================="

    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        echo "终止 server 进程 (PID=$server_pid) ..."
        kill -TERM "$server_pid" 2>/dev/null
        sleep 2
        kill -0 "$server_pid" 2>/dev/null && kill -9 "$server_pid" 2>/dev/null
    fi

    echo "释放端口 ${port} ..."
    lsof -ti :${port} | xargs -r kill -9 2>/dev/null

    echo "清理 vLLM/Server 残留进程 ..."
    pkill -9 -f "server.py.*--port ${port}" 2>/dev/null
    pkill -9 -f "vllm.*worker.*" 2>/dev/null

    echo "清理 GPU 显存占用进程 ..."
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    for p in $gpu_pids; do
        if ps -p $p -u $USER > /dev/null 2>&1; then
            kill -9 $p 2>/dev/null
        fi
    done

    echo "清理完成！退出。"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 动态覆写 yaml 配置保持与脚本参数一致
echo "正在根据脚本参数覆写 conf/qwen3_omni_moe_asr_only.yaml ..."
python scripts/override_yaml.py \
    --file conf/qwen3_omni_moe_asr_only.yaml \
    --gpu-ids "${gpu_ids}" \
    --tp-size ${tp_size} \
    --gpu-mem-util ${gpu_mem_util}

# 启动服务（后台 + wait 配合 trap，确保 bash 能及时响应中断信号）
echo "正在启动服务进程..."
python server.py \
    --model "${model_path}" \
    --host "${host}" \
    --port "${port}" \
    --gpu-ids "${gpu_ids}" \
    --tensor-parallel-size ${tp_size} \
    --gpu-memory-utilization ${gpu_mem_util} \
    ${save_audio_flag} &

server_pid=$!
wait $server_pid
