# Qwen3-ASR Realtime ASR Service

基于 vLLM 的 Qwen3-ASR 实时流式语音识别服务，提供 WebSocket 服务、Web 前端和 Python CLI 客户端。协议部分兼容 OpenAI Realtime API。

核心能力：

- WebSocket 实时音频输入：`/v1/realtime`
- HTTP 健康检查：`/health`
- Web 前端：麦克风实时识别、音频文件上传、配置面板
- Python 客户端：单文件 / wav.list 批量识别
- 服务端 VAD：`none` / `server_vad`
- 历史回退：`ratio` / `chars` / `words` / `tokens`

---

## 目录结构

```text
serve_touch_asr/
├── server.py                       # 服务启动入口，负责 WebSocket/HTTP 路由与连接编排
├── client.py                       # Python CLI 客户端
├── index.html                      # Web 前端页面
├── protocol.py                     # Realtime 协议事件构造
├── requirements.txt
├── engine/                         # 进程级运行时、配置、模型启动
│   ├── runtime.py                  # ServiceRuntime / ServerSettings / EngineRuntime
│   ├── startup.py                  # 模型加载、warmup、ready 状态
│   ├── config.py                   # inference_config.yaml 加载与热更新
│   └── env_snapshot.py             # 启动环境诊断
├── infra/                          # HTTP、日志、SSL 等基础设施
│   ├── http_routes.py              # /health 与首页
│   ├── logging_setup.py
│   └── ssl_context.py
├── session/                        # 单连接状态与后台循环
│   ├── session.py                  # RealtimeSession，音频缓冲与状态
│   ├── state.py                    # VAD / inference 共享状态
│   ├── sender.py                   # WebSocket 事件发送
│   ├── inference_loop.py           # 推理循环
│   └── vad_loop.py                 # server_vad 检测循环
├── model/                          # 推理、VAD、回退与后处理
│   ├── model_inference.py
│   ├── vad.py
│   ├── history_rollback.py
│   └── transcript_postprocess.py
├── examples/
│   └── qwen3asr/
│       ├── conf/inference_config.yaml
│       ├── start_server.sh         # 推荐启动入口
│       └── start_server.local.sh   # 本地环境启动脚本
```

---

## 启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用 example 启动

推荐从 `examples/qwen3asr/` 入口启动，配置、日志和结果都集中在同一个 example 目录下。

```bash
cd runtime/serve_touch_asr

bash examples/qwen3asr/start_server.sh \
  --gpu-ids 0 \
  --tp-size 1
```

本地个人环境可以使用：

```bash
bash examples/qwen3asr/start_server.local.sh
```

启动后检查：

```bash
curl http://localhost:8002/health
```

如果服务使用 TLS，则对应：

```bash
curl -k https://localhost:8002/health
```

### 3. 直接用 `server.py` 启动

```bash
python server.py \
  --model /path/to/Qwen3-ASR \
  --host 0.0.0.0 \
  --port 8002 \
  --gpu-ids 0 \
  --tensor-parallel-size 1
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model`, `-m` | `qwen3-asr` | 模型路径 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8002` | 监听端口 |
| `--tensor-parallel-size`, `-tp` | `1` | tensor parallel 大小 |
| `--gpu-memory-utilization` | `0.6` | vLLM 显存利用率 |
| `--gpu-ids` | `0` | `CUDA_VISIBLE_DEVICES` |
| `--save-audio` | `False` | 是否保存会话音频 |

运行时推理配置在：

```text
examples/qwen3asr/conf/inference_config.yaml
```

该配置支持热更新，服务运行中修改后会自动生效。

---

## example 与 `server.py`

`examples/qwen3asr/` 是 Qwen3-ASR 的推荐运行目录：

| 路径 | 说明 |
|------|------|
| `examples/qwen3asr/conf/inference_config.yaml` | 推理配置，包含 user_prompt、system_prompt、context、language、chunk、VAD、history rollback 等 |
| `examples/qwen3asr/start_server.sh` | 标准启动脚本 |
| `examples/qwen3asr/start_server.local.sh` | 本地个人启动脚本 |

`server.py` 是服务入口，负责：

- 启动 HTTP 路由：`/`、`/health`
- 启动 WebSocket 路由：`/v1/realtime`
- 初始化 vLLM engine、processor、Silero VAD
- 为每个连接创建独立 `RealtimeSession`
- 启动两个后台循环：
  - `inference_loop`：按 `chunk_ms` 或 commit 触发推理
  - `vad_loop`：在 `server_vad` 模式下做语音起止检测和 auto-commit

---

## 服务端

### Web 前端：`index.html`

浏览器访问服务端口即可打开前端：

```text
http://host:port/
https://host:port/
```

前端支持：

- 麦克风实时识别
- 音频文件上传识别
- `AUTO` / `CUSTOM` 服务地址切换
- `chunk_ms`、user_prompt、context、language、VAD、history rollback 配置
- event log 和健康检查状态展示

CUSTOM 填写 `ws://host:port` 或 `wss://host:port`，前端会自动补齐 `/v1/realtime` 和 `/health`。

浏览器麦克风建议使用 HTTPS；HTTP 调试时需要在 Chrome 里允许当前 insecure origin。

### Python 客户端：`client.py`

`client.py` 只是命令行调试入口，不是主要使用方式。`--server` 只传服务地址，不要带 `/v1/realtime`。

```bash
python client.py \
  --server ws://localhost:8002 \
  --input audio.wav
```

如果服务是 TLS，使用 `wss://host:port`。

---

## 测试结果

### 识别效果

Qwen3-ASR 1.7B 的识别效果如下。表中 `N/C/S/D/I` 分别为总字数、正确、替换、删除、插入。

| 场景 | aishell1_test | aishell2_test | ws_meeting | ws_net | test_clean | test_other |
|------|---------------|---------------|------------|--------|------------|------------|
| 非流式 | 1.53% N=104765<br>C=103258 S=1430 D=77 I=96 | 2.72% N=49531 C=48292<br>S=1195 D=44 I=110 | 5.86% N=220358 C=208549<br>S=7440 D=4369 I=1110 | 5.03% N=413868 C=394797<br>S=14093 D=4978 I=1747 | 4.06% N=52576 C=50508<br>S=1588 D=480 I=66 | 5.47% N=52343 C=49624<br>S=2298 D=421 I=144 |
| 伪流式 | 1.53% N=104765<br>C=103258 S=1430 D=77 I=96 | 2.72% N=49531 C=48292<br>S=1195 D=44 I=110 | 5.93% N=220358 C=208412<br>S=7472 D=4474 I=1112 | 5.03% N=413868 C=394798<br>S=14091 D=4979 I=1746 | 3.98% N=52576 C=50551<br>S=1566 D=459 I=65 | 5.70% N=52343 C=49501<br>S=2343 D=499 I=143 |
| +回退<br>1word | 1.53% N=104765<br>C=103258 S=1430 D=77 I=96 | 2.72% N=49531 C=48292<br>S=1195 D=44 I=110 | 5.93% N=220358 C=208411<br>S=7472 D=4475 I=1111 | — | 4.06% N=52576 C=50506<br>S=1589 D=481 I=66 | 5.70% N=52343 C=49504<br>S=2336 D=503 I=143 |
| +回退<br>3token | 1.53% N=104765<br>C=103258 S=1430 D=77 I=96 | 2.72% N=49531 C=48292<br>S=1195 D=44 I=110 | 5.90% N=220358 C=208470<br>S=7467 D=4421 I=1112 | — | 3.95% N=52576 C=50566<br>S=1559 D=451 I=66 | 5.68% N=52343 C=49512<br>S=2338 D=493 I=143 |

### 延迟指标

以下指标引用 `eager=true` 测试结果，格式为 `avg/P50/P90`。

| 指标 | Qwen3-ASR 非流式 | Qwen3-ASR 伪流式 |
|------|------------------|------------------|
| FirstTokenLatency | 367ms/215ms/1243ms | 327ms/185ms/1227ms |
| FinalTokenLatency | 502ms/444ms/710ms | 474ms/438ms/835ms |
| PartialTokenLatency | 505ms/446ms/713ms | 478ms/439ms/896ms |

## 协议概要

客户端发送：

```json
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
```

```json
{"type": "input_audio_buffer.commit"}
```

```json
{
  "type": "session.update",
  "session": {
    "instructions": "你是一个语音转写助手。",
    "audio": {
      "input": {
        "turn_detection": {"type": "server_vad"}
      }
    },
    "extra": {
      "chunk_ms": 1000,
      "user_prompt": "将这段语音转录为纯文本",
      "use_history": true,
      "history_rollback": {
        "enabled": true,
        "strategy": "words",
        "value": 1
      }
    }
  }
}
```

> `instructions`：**系统提示（system message）**，对齐 OpenAI Realtime 语义，进入 `session.asr.system_prompt`。
> `extra.user_prompt`：当前任务指令（user 角色），进入 `session.asr.user_prompt`。
> 二者仅 Qwen3-Omni 生效；Qwen3-ASR 忽略 system/user prompt，使用固定模板。

服务端返回：

| 事件 | 触发时机 |
|------|----------|
| `session.created` | 连接建立，返回配置快照 |
| `session.updated` | `session.update` 后返回最新配置快照 |
| `conversation.item.input_audio_transcription.delta` | 增量转录，cursor 覆盖协议 |
| `conversation.item.input_audio_transcription.completed` | 本轮转录完成 |
| `response.done` | 响应结束标志，跟在 `completed` 之后 |
| `input_audio_buffer.speech_started` | `server_vad` 检测到语音起始 |
| `input_audio_buffer.speech_stopped` | `server_vad` 检测到语音结束 |
| `input_audio_buffer.committed` | `server_vad` auto-commit 确认 |

#### `session.created` / `session.updated`

`extra` 携带运行时配置快照，其中包含**强制语种**和 **ITN 预热状态**：

```json
{
  "type": "session.created",
  "session": {
    "id": "sess_<uuid>",
    "object": "realtime.session",
    "model": "qwen3-asr",
    "instructions": "你是一个语音转写助手。",
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}, "turn_detection": {"type": "server_vad"}}},
    "extra": {
      "chunk_ms": 1000,
      "user_prompt": "将这段语音转录为纯文本",
      "context": "",
      "language": "",
      "itn": {"enabled": true, "available": true, "error": ""},
      "use_history": true,
      "history_rollback": {"enabled": true, "strategy": "words", "value": 1}
    }
  }
}
```

> `instructions`：系统提示，回填 `session.asr.system_prompt`（与 `extra.user_prompt` 区分；不再有 `extra.system_prompt` 字段）。
> `language`：强制语种，空表示自动检测。
> `itn.enabled`：**是否想用**（会话级书面化开关）。
> `itn.available` / `itn.error`：**是否可用**及不可用原因，为**进程级 ITN 预热诊断**（启动 / 配置 reload 时写入），不可用时 `error` 供前端给出安装提示。注意它**不代表识别语种**，本轮真实识别语种见下方 `delta` / `completed` 的 `language` 字段。

> 客户端 `session.update` 只需回传 `extra.itn.enabled`，`available` / `error` 由服务端单向下发。

#### `conversation.item.input_audio_transcription.delta`

增量转录，使用 cursor 覆盖协议。前端处理方式：`text = text.slice(0, cursor) + delta`

```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "cursor": 5,
  "delta": "今天天气",
  "is_final": false,
  "chunk_id": 3,
  "language": "Chinese"
}
```

> `cursor`：本轮已确认文本的字符长度（截断点）。
> `language`：本轮模型实际检测 / 解析出的语种（自动检测时随说话内容变化）。

#### `conversation.item.input_audio_transcription.completed`

本轮转录完成，`transcript` 为最终文本，`language` 为本轮识别语种。

```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "transcript": "今天天气真不错。",
  "language": "Chinese"
}
```

#### `response.done` / `input_audio_buffer.*`

```json
{"type": "response.done"}
{"type": "input_audio_buffer.speech_started", "audio_start_ms": 320, "item_id": "sess_<uuid>_item_1"}
{"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 1840, "item_id": "sess_<uuid>_item_1"}
{"type": "input_audio_buffer.committed", "item_id": "sess_<uuid>_item_1"}
```

---

## 排障

| 现象 | 排查 |
|------|------|
| `Failed to fetch` | 检查 `/health` 是否可达，确认 `ws` 对应 `http`、`wss` 对应 `https` |
| WebSocket 连接失败 | 检查 `/v1/realtime`，确认没有重复写 `/v1/realtime/v1/realtime` |
| 浏览器无法使用麦克风 | 使用 HTTPS，或配置 Chrome insecure origin 白名单 |
| 识别一直为空 | 检查 VAD 是否检测到语音、音频采样率是否为 16kHz PCM |
| 大量 timeout | 降低并发、增大 timeout、查看 GPU 显存和 server 日志 |

---

## 致谢

本项目基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 构建，感谢 Qwen 团队开源模型与相关工作。
