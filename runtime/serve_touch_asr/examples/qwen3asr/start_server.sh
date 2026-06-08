#!/bin/bash
# 启动 Qwen3-ASR Realtime ASR 服务。
# startup 参数（port/tp/gpu-mem 等）由 conf/inference_config.yaml 的 startup 段管理，
# 本脚本只负责：环境变量、CUDA_VISIBLE_DEVICES、PYTHONPATH、启动 python。
# CLI 参数可覆盖 yaml 中的 startup 默认值。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

model_path="${MODEL_PATH:-}"
gpu_ids="${GPU_IDS:-0}"
config="${SCRIPT_DIR}/conf/inference_config.yaml"
silero_model_path="${SILERO_MODEL_PATH:-/path/to/silero-vad}"
log_dir="${LOG_DIR:-${SCRIPT_DIR}/logs}"

export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)        model_path="$2"; shift 2 ;;
        --gpu-ids|-g)      gpu_ids="$2"; shift 2 ;;
        --config|-c)       config="$2"; shift 2 ;;
        --silero-model-path) silero_model_path="$2"; shift 2 ;;
        --log-dir)         log_dir="$2"; shift 2 ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "环境变量:"
            echo "  MODEL_PATH         模型目录（必需）"
            echo "  GPU_IDS            GPU 列表 (默认 0)"
            echo "  SILERO_MODEL_PATH  Silero VAD 模型路径"
            echo "  QWEN3_ASR_REPO     Qwen3-ASR 源码路径"
            echo ""
            echo "选项:"
            echo "  --model, -m          模型路径"
            echo "  --gpu-ids, -g        GPU IDs (默认 0)"
            echo "  --config, -c         配置文件 (默认 conf/inference_config.yaml)"
            echo "  --silero-model-path  Silero VAD 路径"
            echo "  --log-dir            日志目录"
            echo ""
            echo "其余启动参数（port/tp/gpu-mem 等）由 yaml startup 段管理。"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "${model_path}" ]]; then
    echo "错误: 请设置 MODEL_PATH 或使用 --model"
    exit 1
fi

if [[ -n "${QWEN3_ASR_REPO:-}" ]]; then
    export PYTHONPATH="${QWEN3_ASR_REPO}:${PYTHONPATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"

_host_name=$(hostname 2>/dev/null)
_host_ips=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')
[ -z "${_host_ips}" ] && _host_ips="<unknown>"

echo "=================================================="
echo "Qwen3-ASR Realtime Service"
echo "=================================================="
echo "Host:      ${_host_name}"
echo "Host IPs:  ${_host_ips}"
echo "模型路径:  ${model_path}"
echo "配置文件:  ${config}"
echo "GPU IDs:   ${gpu_ids}"
echo "=================================================="

cd "${ROOT_DIR}"

_cleaning=0
cleanup() {
    [ "$_cleaning" -ne 0 ] && return
    _cleaning=1
    trap - SIGINT SIGTERM
    echo ""
    echo "收到退出信号，正在清理..."
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null
        sleep 2
        kill -0 "$server_pid" 2>/dev/null && kill -9 "$server_pid" 2>/dev/null
    fi
    pkill -9 -f "server.py.*--config.*${config}" 2>/dev/null
    pkill -9 -f "vllm.*worker.*" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

python "${ROOT_DIR}/scripts/override_yaml.py" \
    --file "${config}" \
    --gpu-ids "${gpu_ids}" \
    --tp-size 1

echo "正在启动 server.py ..."
python server.py \
    --model "${model_path}" \
    --gpu-ids "${gpu_ids}" \
    --config "${config}" \
    --silero-model-path "${silero_model_path}" \
    --log-dir "${log_dir}" &

server_pid=$!
wait "$server_pid"
