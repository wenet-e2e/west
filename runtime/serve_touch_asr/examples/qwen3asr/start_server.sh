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
port="${SERVER_PORT:-}"
config="${SCRIPT_DIR}/conf/inference_config.yaml"
silero_model_path="${SILERO_MODEL_PATH:-/path/to/silero-vad}"
log_dir="${LOG_DIR:-${SCRIPT_DIR}/logs}"

export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# ============================================================
# 资源占用检查函数（简化版）
# ============================================================

check_gpu_usage() {
    local gpu_id="$1"
    local pids
    pids=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "⚠️  GPU $gpu_id 已被占用:"
        for pid in $pids; do
            ps -o user=,pid=,args= -p "$pid" 2>/dev/null || echo "  PID $pid (进程已退出)"
        done
        return 1
    fi
    return 0
}

check_port_usage() {
    local target_port="$1"
    local pid
    pid=$(lsof -t -i :"$target_port" -sTCP:LISTEN 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
        echo "⚠️  端口 $target_port 已被占用:"
        ps -o user=,pid=,args= -p "$pid" 2>/dev/null || echo "  PID $pid (进程已退出)"
        return 1
    fi
    return 0
}

# ============================================================
# 参数解析
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)        model_path="$2"; shift 2 ;;
        --gpu-ids|-g)      gpu_ids="$2"; shift 2 ;;
        --port|-p)         port="$2"; shift 2 ;;
        --config|-c)       config="$2"; shift 2 ;;
        --silero-model-path) silero_model_path="$2"; shift 2 ;;
        --log-dir)         log_dir="$2"; shift 2 ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "环境变量:"
            echo "  MODEL_PATH         模型目录（必需）"
            echo "  GPU_IDS            GPU 列表 (默认 0)"
            echo "  SERVER_PORT        服务端口 (默认 8001)"
            echo "  SILERO_MODEL_PATH  Silero VAD 模型路径"
            echo "  QWEN3_ASR_REPO     Qwen3-ASR 源码路径"
            echo "  LOG_DIR            启动日志与服务日志目录"
            echo "  LOG_TAG/RUN_TAG    追加到启动日志文件名的用途标签"
            echo ""
            echo "选项:"
            echo "  --model, -m          模型路径"
            echo "  --gpu-ids, -g        GPU IDs (默认 0)"
            echo "  --port, -p           服务端口（覆盖 yaml）"
            echo "  --config, -c         配置文件 (默认 conf/inference_config.yaml)"
            echo "  --silero-model-path  Silero VAD 路径"
            echo "  --log-dir            日志目录"
            echo ""
            echo "启动前会自动检查 GPU 和端口是否被占用。"
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

# ============================================================
# 确定实际使用的端口（CLI --port > 环境变量 SERVER_PORT > 默认 8001）
# 注意：服务端口用 SERVER_PORT，避免与 .bashrc 中代理用的 PORT 变量冲突。
# 端口/GPU/TP 均为启动参数，不再写回 yaml。
# ============================================================
port="${port:-8001}"

# ============================================================
# 资源占用检查（GPU 和 端口）
# ============================================================
echo "🔍 检查资源占用..."
_resource_conflict=0

if ! check_gpu_usage "$gpu_ids"; then
    _resource_conflict=1
fi

if ! check_port_usage "$port"; then
    _resource_conflict=1
fi

if [[ $_resource_conflict -eq 1 ]]; then
    echo ""
    echo "❌ 资源冲突，启动中止。请释放上述资源后重试。"
    exit 1
fi
echo "✅ GPU ${gpu_ids} 和端口 ${port} 可用"
echo ""

_host_name=$(hostname 2>/dev/null)
_host_ips=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')
[ -z "${_host_ips}" ] && _host_ips="<unknown>"

_sanitize_log_part() {
    local value="$1"
    value="${value//,/-}"
    value="$(printf "%s" "${value}" | tr -c 'A-Za-z0-9_.-' '_')"
    value="${value##_}"
    value="${value%%_}"
    printf "%s" "${value:-unknown}"
}

_startup_port="${port}"
_timestamp="$(date +%Y%m%d_%H%M%S)"
_host_part="$(_sanitize_log_part "${_host_name}")"
_port_part="$(_sanitize_log_part "port-${_startup_port}")"
_gpu_part="$(_sanitize_log_part "gpu-${gpu_ids}")"
_eager_part="$(_sanitize_log_part "eager-${ENFORCE_EAGER}")"
_conda_part="$(_sanitize_log_part "${CONDA_DEFAULT_ENV:-none}")"
_tag_part=""
if [[ -n "${LOG_TAG:-${RUN_TAG:-}}" ]]; then
    _tag_part="_$(_sanitize_log_part "${LOG_TAG:-${RUN_TAG:-}}")"
fi
startup_log="${log_dir}/qwen3asr_${_timestamp}_${_host_part}"
startup_log+="_${_port_part}_${_gpu_part}_${_eager_part}"
startup_log+="_${_conda_part}${_tag_part}.log"

mkdir -p "${log_dir}"
exec > >(tee -a "${startup_log}") 2>&1

echo "=================================================="
echo "Qwen3-ASR Realtime Service"
echo "=================================================="
echo "Host:      ${_host_name}"
echo "Host IPs:  ${_host_ips}"
echo "模型路径:  ${model_path}"
echo "配置文件:  ${config}"
echo "GPU IDs:   ${gpu_ids}"
echo "Eager:     ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "vLLM V1:   VLLM_USE_V1=${VLLM_USE_V1}"
echo "日志文件:  ${startup_log}"
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

echo "正在启动 server.py ..."
python server.py \
    --model "${model_path}" \
    --gpu-ids "${gpu_ids}" \
    --port "${port}" \
    --config "${config}" \
    --silero-model-path "${silero_model_path}" \
    --log-dir "${log_dir}" &

server_pid=$!
wait "$server_pid"
