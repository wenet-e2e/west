# On-Policy Distillation Training Recipe

This example provides a complete recipe for training audio language models using **On-Policy Distillation** on audio tasks.

## Overview

On-Policy Distillation training flow:
1. **Student generates completions** (on-policy sampling)
2. **Teacher evaluates student's completions** (computes logits/probabilities)
3. **Student learns to match teacher's distribution** via KL divergence

This approach avoids the distribution shift problem of off-policy distillation.

## Data

| Dataset | Type | Source |
|---------|------|--------|
| AVQA | Training | [AVQA Website](https://mn.cs.tsinghua.edu.cn/avqa/) / [HuggingFace](https://huggingface.co/datasets/gijs/avqa-processed) |
| MMAU | Evaluation | [GitHub](https://github.com/Sakshi113/MMAU) |

## Tasks & Results

### Task 1: Detailed Audio Captioning

**Evaluation Method:** Cascade evaluation where the model first generates detailed captions, which are subsequently used to perform downstream question-answering tasks, thereby gauging the completeness of the produced captions. (See [Omni-Captioner](https://arxiv.org/abs/2510.12720))

**Teacher Backend:** Remote vLLM API (`RemoteKnowledgeDistillationTrainer`)

| Model | MMAU (v05.15.25) |
|-------|------------------|
| Omni-Captioner | 72.8 |
| Step-Audio-R1.1 | 72.3 |
| Qwen2.5-Omni-3B | 65.6 |
| **OPD (Student: Omni-3B, Teacher: Step-Audio-R1.1)** | **69.6** |
| **OPD (Student: Omni-3B, Teacher: Omni-Captioner)** | **69.1** |
| Qwen2.5-Omni-7B | 68.7 |
| **OPD (Student: Omni-7B, Teacher: Omni-Captioner)** | **70.1** |

<details>
<summary><b>Example: Caption Quality Comparison</b></summary>

[mmau_test.wav](https://github.com/stepfun-ai/Step-Audio-R1/blob/main/assets/mmau_test.wav)

#### Before: Qwen2.5-Omni-3B (Baseline)

> The audio features a man speaking in a casual tone, expressing excitement about successfully defeating two opponents in a game with a single hit. The background includes music and sound effects, suggesting an action-packed video game environment.

#### After: Qwen2.5-Omni-3B + On-Policy Distillation

> The audio clip is a brief, high-energy, and immersive gaming session, likely from a first-person shooter or action game, set in a virtual environment that mimics a modern, urban battlefield. The recording begins with a male voice, likely the player, speaking in a casual, conversational tone. The voice is clear, with a neutral to slightly excited inflection, and is delivered in a standard American accent. The player's speech is punctuated by a brief, sharp electronic beep, which is characteristic of a digital interface or game alert.
>
> As the player speaks, the audio transitions into a chaotic, action-packed sequence. The first sound is a sharp, metallic clank, followed by a rapid, high-pitched electronic beep, which is immediately accompanied by a loud, resonant explosion. The explosion is accompanied by a deep, rumbling bass tone, and the sound is layered with a gritty, metallic texture, suggesting a heavy, industrial environment. The player's voice is heard in the background, with a brief, excited exclamation: "I smashed both of them with one hit, very nice."
>
> The next moment is marked by a sudden, intense burst of gunfire and a series of rapid, overlapping electronic beeps and clicks, indicating the use of a weapon or device. The player's voice is heard again, this time with a more urgent, excited tone: "And there's Roy's second life down." The audio is filled with the sounds of gunfire, explosions, and electronic beeps, all layered with a sense of urgency and excitement.
>
> The clip ends abruptly, with a final, sharp electronic beep and a brief, high-pitched electronic tone, followed by a sudden, loud, and distorted electronic sound. The audio is heavily distorted, with a sense of chaos and disorientation, and the clip ends with a brief, high-pitched electronic beep.
>
> Throughout the clip, the player's voice is clear and audible, and the background is filled with the sounds of gunfire, explosions, and electronic beeps, all layered with a sense of urgency and excitement. The overall atmosphere is one of intense action and high stakes, with the player's voice and the game's soundscape creating a vivid, immersive experience. The clip is likely from a first-person shooter or action game, and the player's excitement and commentary suggest a high level of engagement and enjoyment.

</details>

### Task 2: Audio QA

**Teacher Backend:** Local HuggingFace model (`KnowledgeDistillationTrainer`)

| Model | MMAU (v05.15.25) | MMSU |
|-------|------------------|------|
| Qwen2-Audio-7B | 56.9 | 30.38 |
| + GRPO | **67.2** | **54.12** |
| + On-Policy Distillation (Qwen-omni-3b-grpo teacher) | **67.9** | 53.30 |

## Quick Start

### 1. Prepare Data and Models

```bash
bash run.sh --stage prepare
```

This will download:
- AVQA training dataset from HuggingFace
- MMAU test-mini audio files
- Student model (Qwen2.5-Omni-3B)
- Teacher model (Qwen3-Omni-30B-A3B-Captioner)

### 2. Start Teacher vLLM Server (Remote Teacher Mode)

On a separate machine with sufficient GPU memory:

```bash
bash run.sh --stage vllm_teacher
```

This starts a vLLM server with:
- `--max-logprobs 128`: Returns top-k logprobs for distillation
- `--tensor-parallel-size 4`: Distributes model across 4 GPUs

### 3. Training

Update the teacher API URL in `run.sh`:
```bash
teacher_model_name_or_path=http://your-teacher-machine-ip:9999/v1
```

Then run training:
```bash
bash run.sh --stage train
```

### 4. Evaluation (Decode + Cascaded LLM Eval)

Set your LLM API key for cascaded evaluation:
```bash
export LLM_API_KEY=sk-xxxx
```

Run decoding and evaluation:
```bash
bash run.sh --stage decode
```

This runs three steps for each checkpoint:
1. Generate captions using the trained model
2. Use GPT to answer questions based on generated captions
3. Compute accuracy on MMAU benchmark

## Configuration

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | models/Qwen2.5-Omni-3B | Student model path |
| `--teacher_model_name_or_path` | URL or local path | Teacher model (URL for remote, path for local) |
| `--max_completion_length` | 4096 | Maximum generation length |
| `--topk_logits_k` | 64 | Top-k logits for distillation (None = full vocab) |
| `--num_generations` | 1 | Number of rollouts per prompt |
| `--temperature` | 1.0 | Sampling temperature for generation |

### Teacher Modes

**Remote Teacher** (`RemoteKnowledgeDistillationTrainer`):
- Teacher model served via vLLM API
- Suitable for large teacher models
- Set `--teacher_model_name_or_path` to API URL (e.g., `http://host:9999/v1`)

**Local Teacher** (`KnowledgeDistillationTrainer`):
- Teacher model loaded locally with DeepSpeed
- Suitable when both models fit in memory
- Set `--teacher_model_name_or_path` to model path
