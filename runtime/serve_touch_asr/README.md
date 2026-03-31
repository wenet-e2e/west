# Qwen3-Omni Realtime ASR Service

基于 [vLLM-Omni](https://github.com/vllm-project/vllm) 推理引擎和 [websockets](https://websockets.readthedocs.io/) 库构建的实时流式语音识别服务，部分兼容 [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) 协议。

提供 Web 前端（麦克风 / 文件上传）和 Python CLI 客户端两种接入方式。

---

## Features

| 类别 | 特性 |
|------|------|
| 并发 | 每个 WebSocket 连接创建独立的 `AudioSession`，包含独立的 `bytearray` 缓冲区、推理游标和 `asyncio.Lock` |
| 双协程 | 连接建立后启动 `inference_processor`（100ms 周期，定时推理）和 `vad_monitor`（50ms 周期，Silero VAD 检测）|
| 输出 | cursor + delta 协议：前端执行 `text = text.slice(0, cursor) + delta` 实现回退覆盖 |
| 回退 | `RollbackConfig` 支持 `ratio` / `chars` / `words` 三种策略（中文走 jieba 分词）|
| 热更新 | `inference_logic.py` 在每次推理前通过 `importlib.reload()` 重载；`conf/inference_config.yaml` 通过文件 mtime 变更检测自动生效 |
| 话轮 | `none`（手动 commit）和 `server_vad`（Silero VAD 自动端点检测 + auto-commit）|
| 健康检查 | `GET /health` 返回 `{"ready": bool, "stage": str}`，前端据此判断是否可以建立 WebSocket 连接 |
| 前端 | session-block + turn-line 分层 DOM，推理配置面板（chunk_ms / prompt / 回退策略 / 话轮检测模式）|

---

## Quick Start

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> `vllm-omni` 需按官方文档单独安装。

### 2. 启动服务

```bash
export MODEL_PATH="/path/to/Qwen3-Omni-30B-A3B-Instruct"
export SILERO_REPO="/path/to/silero-vad"

# 多卡
bash start_server.sh --gpu-ids 0,1,2,3 --tp-size 4

# 单卡
bash start_server.sh --gpu-ids 0 --tp-size 1

# 直接启动
python server.py --model $MODEL_PATH --port 8001 --gpu-ids 0
```

### 3. 前端

```bash
python -m http.server 8080
# 浏览器打开 http://localhost:8080/index.html
```

> **注意：** 现代浏览器要求在 HTTPS 下才能调用麦克风。由于本地测试是 HTTP，你需要配置 Chrome 允许不安全的来源：
> 1. 在 Chrome 地址栏输入：`chrome://flags/#unsafely-treat-insecure-origin-as-secure`
> 2. 找到 **Insecure origins treated as secure** 选项
> 3. 在文本框中填入你的访问地址，例如：`http://10.201.200.200:8080`（如果有多个可用逗号分隔）
> 4. 将右侧下拉框改为 **Enabled**
> 5. 点击底部弹出的 **Relaunch** 重启浏览器，即可正常授权麦克风。

### 4. Python 客户端

```bash
python client.py --server ws://localhost:8001 --input audio.wav

python client.py --input audio.wav --simulate-streaming \
    --rollback-strategy words --rollback-value 2

python client.py --input wav.list --output result.txt --output-format jsonl
```

---

## Architecture

每个 WebSocket 连接（`realtime_handler`）的生命周期：

1. 创建 `AudioSession`（独立缓冲区、独立锁）
2. 检查 `engine_ready`：若未就绪则阻塞等待（正常路径下前端已通过 `GET /health` 确认就绪）
3. 发送 `session.created` 事件（含当前配置快照）
4. 启动两个后台协程：
   - **`inference_processor`**：100ms 轮询。根据 `chunk_ms` 检查缓冲区是否达到阈值，或 `is_committing` 是否被置位。达到条件时调用 `extract_for_inference()` 取出音频，通过 `inference_logic.run_omni_inference()` 送入 vLLM 引擎，流式 yield 出文本并通过 `send_streaming_delta()` 推送 delta 事件。
   - **`vad_monitor`**：50ms 轮询。仅在 `turn_detection == "server_vad"` 时工作。调用 `extract_new_audio_for_vad()` 获取新增音频，feed 给 `StreamingVAD`（Silero VADIterator 封装）。检测到 `speech_started` 时记录 `speech_start_byte`；检测到 `speech_stopped` 时自动置 `is_committing = True`。
5. 主协程：`async for raw_msg in websocket` 接收客户端消息。

### 防幻觉机制

| 机制 | 位置 | 逻辑 |
|------|------|------|
| VAD 门控 | `inference_processor` | `server_vad` 模式下，`vad_speech_active == False` 时跳过定时推理 |
| 空 commit 跳过 | `inference_processor` | VAD 模式 commit 时，若 `speech_start_byte == 0` 且 `accumulated_text == ""`，直接发空 completed |
| 音频偏移 | `extract_for_inference` | VAD 模式下从 `speech_start_byte` 开始取音频，丢弃前导静音 |
| 前导静音跳过 | `inference_processor` | `none` 模式下，`silence_skip.enabled` 时逐帧计算 RMS，有声帧不足则跳过该 chunk |

### 热更新

- `inference_logic.py`：`inference_processor` 每次循环开头执行 `importlib.reload(inference_logic)`
- `conf/inference_config.yaml`：`load_inference_config()` 比较文件 mtime，变更时重新加载。支持热切换 `log_level`

### 引擎初始化（`init_engine`）

分四步：加载 Processor → 实例化 AsyncOmni → 引擎预热（dummy generate，解决"假就绪"问题）→ 设置 `engine_ready = True`。在 `run_server` 中通过 `asyncio.create_task` 后台执行，WebSocket 服务先行启动。

---

## Server Arguments

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model`, `-m` | `qwen3-omni` | 模型路径 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8001` | 监听端口 |
| `--tensor-parallel-size`, `-tp` | `1` | TP GPU 数 |
| `--gpu-memory-utilization` | `0.75` | 显存利用率 |
| `--gpu-ids` | `0` | CUDA_VISIBLE_DEVICES |
| `--save-audio` | `False` | 保存会话音频为 WAV（异步落盘）|
| `--rollback-strategy` | `none` | `none`/`ratio`/`chars`/`words` |
| `--rollback-value` | `0.0` | 回退参数值 |

## Client Arguments

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | `ws://localhost:8001` | WebSocket 地址 |
| `--input` | (必填) | wav 文件或 wav.list |
| `--output` | stdout | 输出文件 |
| `--output-format` | `text` | `text` / `jsonl` |
| `--packet-ms` | `100` | 每包时长 |
| `--sr` | `16000` | 采样率 |
| `--simulate-streaming` | `False` | 模拟实时 1 倍速发送 |
| `--chunk-ms` | 服务端默认 | 覆盖 chunk 阈值 |
| `--prompt` | 服务端默认 | 覆盖 prompt |
| `--use-history` | `true` | 历史文本拼接 |
| `--rollback-strategy` | `none` | 回退策略 |
| `--rollback-value` | `0.0` | 回退参数值 |

---

## WebSocket Protocol

### Client → Server

#### `input_audio_buffer.append`
追加 16kHz PCM16 音频块（base64 编码）。
```json
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
```

#### `input_audio_buffer.commit`
手动结束收音（`none` 模式），触发一次完整推理。
```json
{"type": "input_audio_buffer.commit"}
```

#### `session.update`
更新会话配置，服务端返回 `session.updated` 确认。
```json
{
  "type": "session.update",
  "session": {
    "instructions": "请转录这段音频。",
    "audio": {
      "input": {
        "turn_detection": {
          "type": "server_vad",
          "threshold": 0.5,
          "prefix_padding_ms": 300,
          "silence_duration_ms": 700
        }
      }
    },
    "extra": {
      "chunk_ms": 1000,
      "use_history": true,
      "rollback": {
        "enabled": false,
        "strategy": "none",
        "value": 0.0
      }
    }
  }
}
```
> `turn_detection.type`：`"none"`（手动 commit）或 `"server_vad"`（Silero VAD 自动端点）
> `rollback.strategy`：`"none"` / `"ratio"` / `"chars"` / `"words"`

---

### Server → Client

#### `session.created` / `session.updated`
连接建立（`created`）或 `session.update` 后（`updated`）返回当前配置快照。
```json
{
  "type": "session.created",
  "session": {
    "id": "sess_<uuid>",
    "object": "realtime.session",
    "model": "qwen3-omni",
    "instructions": "请转录这段音频。",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 16000},
        "turn_detection": {
          "type": "none",
          "threshold": 0.5,
          "prefix_padding_ms": 300,
          "silence_duration_ms": 700
        }
      }
    },
    "extra": {
      "chunk_ms": 1000,
      "use_history": true,
      "rollback": {"enabled": false, "strategy": "none", "value": 0.0}
    }
  }
}
```

#### `conversation.item.input_audio_transcription.delta`
增量转录推送。前端执行：`text = text.slice(0, cursor) + delta`
```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "cursor": 5,
  "delta": "今天天气",
  "is_final": false,
  "chunk_id": 3
}
```
> `cursor`：本轮已确认文本的字符长度（截断点）
> `delta`：cursor 之后的新增文本
> `chunk_id`：触发本次推理的 chunk 序号

#### `conversation.item.input_audio_transcription.completed`
本轮转录完成，`transcript` 为最终文本。
```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "transcript": "今天天气真不错。"
}
```

#### `response.done`
响应结束标志，跟在 `completed` 之后。
```json
{"type": "response.done"}
```

#### `input_audio_buffer.speech_started` `[server_vad]`
VAD 检测到语音起始。
```json
{
  "type": "input_audio_buffer.speech_started",
  "audio_start_ms": 320,
  "item_id": "sess_<uuid>_item_1"
}
```

#### `input_audio_buffer.speech_stopped` `[server_vad]`
VAD 检测到语音结束，随即自动触发 commit。
```json
{
  "type": "input_audio_buffer.speech_stopped",
  "audio_end_ms": 1840,
  "item_id": "sess_<uuid>_item_1"
}
```

#### `input_audio_buffer.committed` `[server_vad]`
VAD auto-commit 确认。
```json
{
  "type": "input_audio_buffer.committed",
  "item_id": "sess_<uuid>_item_1"
}
```

### HTTP 端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | `{"ready": bool, "stage": str}`，用于前端判断引擎是否就绪 |
| `/v1/realtime` | WebSocket | 主 WebSocket 端点 |

---

## Project Structure

```
serve_touch_asr/
├── server.py                # WebSocket 服务主体
├── inference_logic.py       # 可热载推理模块（RollbackConfig / StreamingVAD / run_omni_inference）
├── client.py                # Python CLI 客户端
├── index.html               # Web 前端
├── start_server.sh          # 多卡启动脚本
├── conf/
│   ├── inference_config.yaml      # 运行时热配置
│   └── qwen3_omni_moe_asr_only.yaml  # vLLM stage 配置模板
├── scripts/
│   └── override_yaml.py           # YAML 属性覆写工具
├── requirements.txt
└── .gitignore
```
