#!/bin/bash
# ==============================================
# Qwen3-Omni ASR vLLM 服务启动脚本
# ==============================================

# 激活环境
#source /bucket/output/jfs-hdfs/user/pengshen.zhang/workspace/LLM/.bashrc
#conda activate qwen

# 配置
MODEL_PATH="/bucket/output/jfs-hdfs/user/pengshen.zhang/workspace/transformer_models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"
HOST="0.0.0.0"
PORT=80
GPU_IDS="0,1,2,3"
TP_SIZE=4
GPU_MEM_UTIL=0.9

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --gpu-ids|-g)
            GPU_IDS="$2"
            shift 2
            ;;
        --tp-size|-tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model, -m     模型路径 (默认: Qwen3-Omni-30B-A3B-Instruct)"
            echo "  --port, -p      服务端口 (默认: 8000)"
            echo "  --gpu-ids, -g   GPU ID (默认: 0)"
            echo "  --tp-size, -tp  Tensor 并行数 (默认: 1)"
            echo ""
            echo "示例:"
            echo "  # 单 GPU 启动"
            echo "  $0 --gpu-ids 0"
            echo ""
            echo "  # 多 GPU 启动 (tensor parallel)"
            echo "  $0 --gpu-ids 0,1,2,3 --tp-size 4"
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
echo "模型路径: ${MODEL_PATH}"
echo "服务地址: ${HOST}:${PORT}"
echo "GPU IDs:  ${GPU_IDS}"
echo "TP Size:  ${TP_SIZE}"
echo "=================================================="

# 进入目录
cd "$(dirname "$0")"

# 启动服务
python server.py \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-ids "${GPU_IDS}" \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTIL}
