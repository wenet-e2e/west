# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
On-Policy Knowledge Distillation Trainer for Qwen2-Audio, Qwen-omni models.

Implements on-policy distillation where:
1. Student model generates completions (on-policy sampling)
2. Teacher model evaluates the student's completions
3. Student learns to match teacher's distribution on its own samples

This avoids distribution shift problems of off-policy distillation.

Supports two modes:
- Local teacher: Teacher model loaded locally
- Remote teacher: Teacher model accessed via API (e.g., vLLM server)
"""

import base64
import io
import logging
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import requests
import soundfile as sf
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (GenerationConfig, PreTrainedModel, ProcessorMixin,
                          Trainer, TrainingArguments)
from trl.models import prepare_deepspeed, unwrap_model_for_generation

# Type alias for reward functions
RewardFunc = Callable[[list, list], list[float]]


class KnowledgeDistillationTrainer(Trainer):
    """
    Trainer for On-Policy Knowledge Distillation.

    The training loop:
    1. Student generates completions from prompts (on-policy)
    2. Teacher computes logits/probabilities on student's completions
    3. Student learns to match teacher's distribution via KL divergence

    Args:
        model: Pre-initialized student model
        teacher_model: Pre-initialized teacher model
        processor: Pre-initialized processor for the student model
        teacher_model_processor: Pre-initialized processor for the teacher model
        reward_funcs: Optional list of reward functions for monitoring (not used in loss)
        args: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        data_collator: Collate function from dataset
    """

    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        processor: ProcessorMixin,
        teacher_model_processor: ProcessorMixin,
        args: TrainingArguments,
        reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        data_collator: Optional[Callable] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )

        set_seed(args.seed, device_specific=True)

        self.num_generations = 1
        # Top-k logits for distillation (None = use full vocabulary)
        self.topk_logits_k = args.topk_logits_k

        # Reward functions for monitoring (optional, not used in loss)
        if reward_funcs is not None:
            self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        else:
            self.reward_funcs = []

        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        # Teacher model setup
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            assert self.is_deepspeed_enabled, "Teacher model requires DeepSpeed to be enabled"
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)

        self.teacher_model_processor = teacher_model_processor

    def _rollout(self, model, inputs: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Generate completions from the actor model.

        Args:
            model: The actor model
            inputs: Dict with input_ids, attention_mask, input_features, feature_attention_mask

        Returns:
            generated_strs: List of generated completion strings
            generated_ids: Tensor of generated token ids [batch * num_generations, seq_len]
            generated_mask: Mask for valid generated tokens (up to and including EOS)
        """
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                generation_config=self.generation_config,
            )

        prompt_length = inputs["input_ids"].size(1)
        generated_ids = prompt_completion_ids[:, prompt_length:]

        # Create mask for valid tokens (up to and including first EOS)
        device = self.accelerator.device
        is_eos = generated_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        generated_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # Decode to strings
        generated_strs = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_strs, generated_ids, generated_mask

    def _get_per_token_logits(self, model, inputs: dict) -> torch.Tensor:
        """Compute per-token logits (for KL computation with full distribution).

        Args:
            model: The model to compute logits from
            inputs: Dict with input_ids, attention_mask, input_features, feature_attention_mask

        Returns:
            logits: Raw logits for each position [batch, seq_len-1, vocab_size]
        """
        # Handle wrapped models (DeepSpeed or Qwen2.5-Omni thinker)
        forward_model = model
        if hasattr(model, "module") and hasattr(model.module, "thinker"):
            forward_model = model.module.thinker
        elif hasattr(model, "thinker"):
            forward_model = model.thinker

        logits = forward_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        ).logits

        # Shift: exclude last logit (no prediction for last token)
        return logits[:, :-1, :]

    def _prepare_student_logprob_inputs(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> dict:
        """Prepare inputs for student model log probability computation.

        Args:
            inputs: Original batch inputs (prompt only)
            generated_ids: Generated token IDs [batch, completion_len]
            generated_mask: Mask for valid generated tokens [batch, completion_len]

        Returns:
            Dict with concatenated prompt + completion inputs
        """
        # Concatenate prompt and generated ids
        prompt_ids = inputs["input_ids"].repeat_interleave(self.num_generations, dim=0)
        prompt_completion_ids = torch.cat([prompt_ids, generated_ids], dim=1)

        # Expand masks for num_generations
        prompt_mask = inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
        attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)
        input_features = inputs["input_features"].repeat(self.num_generations, 1, 1)
        feature_mask = inputs["feature_attention_mask"].repeat_interleave(self.num_generations, dim=0)

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        }

    def _prepare_teacher_logprob_inputs(
        self,
        original_prompts: list,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        audios: list,
        sample_rate: int = 16000,
    ) -> tuple[dict, int]:
        """Prepare inputs for teacher model log probability computation.

        Since teacher model may have different tokenizer, we need to:
        1. Re-process prompts with teacher's processor
        2. Concatenate with student's generated token IDs

        Args:
            original_prompts: List of conversation dicts (user prompts with audio refs)
            generated_ids: Tensor of generated token IDs from student model [batch, seq_len]
            generated_mask: Mask for valid generated tokens [batch, seq_len]
            audios: List of audio arrays
            sample_rate: Audio sample rate

        Returns:
            Tuple of (inputs dict, prompt_length)
        """
        processor = self.teacher_model_processor

        # Get prompt-only texts using teacher's chat template
        prompt_texts = [
            processor.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in original_prompts
        ]

        # Tokenize prompt-only with teacher processor
        prompt_processed = processor(
            text=prompt_texts,
            audio=audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        prompt_length = prompt_processed["input_ids"].shape[1]

        # Get prompt tensors
        prompt_ids = prompt_processed["input_ids"]
        prompt_mask = prompt_processed["attention_mask"]

        # Concatenate: [prompt_ids, generated_ids]
        # Note: generated_ids are from student, but we assume shared vocabulary
        device = self.accelerator.device
        full_input_ids = torch.cat([prompt_ids.to(device), generated_ids], dim=1)
        full_attention_mask = torch.cat([prompt_mask.to(device), generated_mask], dim=1)

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "input_features": prompt_processed["input_features"].to(device),
            "feature_attention_mask": prompt_processed["feature_attention_mask"].to(device),
        }, prompt_length

    def _compute_reverse_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute reverse KL divergence: KL(student || teacher).

        This encourages mode-seeking behavior where student focuses on
        high-probability regions of teacher's distribution.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens [batch, seq_len]
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # KL(student || teacher) = sum_x p_student(x) * (log p_student(x) - log p_teacher(x))
        per_token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

        return per_token_kl

    def _compute_topk_reverse_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        topk: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute reverse KL divergence using student's top-k tokens.

        For reverse KL: KL(student || teacher), we use student's probability as weights,
        so we should select top-k based on student's distribution.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens [batch, seq_len]
            topk: Number of top tokens to consider
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # For reverse KL, use STUDENT's top-k indices (matches the weighting in KL formula)
        _, student_topk_indices = torch.topk(student_logits, k=topk, dim=-1)  # [B, seq, k]

        # Gather student and teacher logits for student's top-k tokens
        student_topk_logits = torch.gather(student_logits, dim=-1, index=student_topk_indices)
        teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=student_topk_indices)

        # Compute probabilities over top-k (re-normalized)
        student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
        student_topk_probs = F.softmax(student_topk_logits, dim=-1)

        # KL(student || teacher) over top-k tokens
        per_token_kl = (student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)

        return per_token_kl

    def _compute_rewards(self, generated_strs: list[str], meta_data: list) -> Optional[torch.Tensor]:
        """Compute rewards for generated completions (for monitoring only).

        Args:
            generated_strs: List of generated completion strings
            meta_data: Tuple of (solutions, prompts, audios, ..., questions, choices)
                       questions and choices are always at the last two positions

        Returns:
            rewards_per_func: Rewards from each reward function [num_samples, num_funcs]
                             or None if no reward functions are configured
        """
        if not self.reward_funcs:
            return None

        solutions = meta_data[0]
        questions = meta_data[-2]  # questions is at second to last position
        choices_list = meta_data[-1]  # choices is at last position

        # Expand solutions for num_generations
        solutions_expanded = [s for s in solutions for _ in range(self.num_generations)]

        # Build prompt_question_list from questions and choices
        # Format: "Question: {question}\nOptions:\n{choice1}\n{choice2}\n..."
        prompt_question_list = []
        for question, choices in zip(questions, choices_list):
            options_str = "\n".join(choices)
            prompt_question = f"Question: {question}\nOptions:\n{options_str}"
            # Expand for num_generations
            for _ in range(self.num_generations):
                prompt_question_list.append(prompt_question)

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(solutions_expanded), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            rewards = reward_func(
                hypos_list=generated_strs,
                ground_truth_list=solutions_expanded,
                prompt_question_list=prompt_question_list,
            )
            rewards_per_func[:, i] = torch.tensor(rewards, dtype=torch.float32, device=device)

        return rewards_per_func

    def _log_metrics(
        self,
        generated_mask: torch.Tensor,
        per_token_kl: torch.Tensor,
        loss: torch.Tensor,
        rewards_per_func: Optional[torch.Tensor] = None,
    ):
        """Log training metrics."""
        completion_lengths = self.accelerator.gather_for_metrics(generated_mask.sum(1)).float()
        self._metrics["completion_length"].append(completion_lengths.mean().item())
        self._metrics["completion_length_min"].append(completion_lengths.min().item())
        self._metrics["completion_length_max"].append(completion_lengths.max().item())

        mean_kl = ((per_token_kl * generated_mask).sum(dim=1) / generated_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())

        # Log rewards if available
        if rewards_per_func is not None:
            total_rewards = rewards_per_func.sum(dim=1)
            gathered_total_rewards = self.accelerator.gather_for_metrics(total_rewards)
            self._metrics["reward"].append(gathered_total_rewards.mean().item())
            self._metrics["reward_min"].append(gathered_total_rewards.min().item())
            self._metrics["reward_max"].append(gathered_total_rewards.max().item())
            # Log individual reward function values
            for i, reward_func in enumerate(self.reward_funcs):
                func_name = reward_func.__name__
                gathered_rewards = self.accelerator.gather_for_metrics(rewards_per_func[:, i])
                self._metrics[f"rewards/{func_name}"].append(gathered_rewards.mean().item())
                self._metrics[f"rewards/{func_name}_min"].append(gathered_rewards.min().item())
                self._metrics[f"rewards/{func_name}_max"].append(gathered_rewards.max().item())

    # ==================== Main Training Loop ====================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute on-policy knowledge distillation loss.

        Training flow:
        1. Student generates completions (on-policy sampling)
        2. Teacher computes logits on student's completions
        3. Student learns to match teacher's distribution via KL divergence
        """
        # Extract meta_data for teacher input preparation
        meta_data = inputs.pop("meta_data")
        original_prompts, audios = meta_data[1], meta_data[2]

        # Step 1: Generate completions from student model (on-policy)
        generated_strs, generated_ids, generated_mask = self._rollout(model, inputs)

        # Compute rewards for monitoring (not used in loss)
        rewards_per_func = self._compute_rewards(generated_strs, meta_data)

        # Step 2: Prepare teacher inputs and compute teacher logits
        teacher_inputs, teacher_prompt_length = self._prepare_teacher_logprob_inputs(
            original_prompts=original_prompts,
            generated_ids=generated_ids,
            generated_mask=generated_mask,
            audios=audios,
        )

        with torch.inference_mode():
            teacher_logits = self._get_per_token_logits(self.teacher_model, teacher_inputs)
        # Slice to get only completion token logits (exclude prompt)
        # Note: -1 because of the shift in _get_per_token_logits
        teacher_completion_logits = teacher_logits[:, teacher_prompt_length - 1:, :]

        # Step 3: Prepare student inputs and compute student logits
        student_logprob_inputs = self._prepare_student_logprob_inputs(inputs, generated_ids, generated_mask)
        student_logits = self._get_per_token_logits(model, student_logprob_inputs)
        # Slice to get only completion token logits
        student_prompt_length = inputs["input_ids"].size(1)
        student_completion_logits = student_logits[:, student_prompt_length - 1:, :]

        # Step 4: Align vocabulary size (student and teacher may have different vocabulary sizes)
        min_vocab_size = min(
            student_completion_logits.shape[2],
            teacher_completion_logits.shape[2],
        )
        student_completion_logits = student_completion_logits[:, :, :min_vocab_size]
        teacher_completion_logits = teacher_completion_logits[:, :, :min_vocab_size]

        assert student_completion_logits.shape == teacher_completion_logits.shape, (
            f"{student_completion_logits.shape} != {teacher_completion_logits.shape}"
        )
        completion_mask = generated_mask.float()

        # Step 5: Compute KL divergence loss
        # Reverse KL is preferred for on-policy distillation:
        # - Uses student's probability as weights (consistent with on-policy sampling)
        # - Mode-seeking: student focuses on what it actually generates
        # - Avoids mode averaging problem of forward KL
        if self.topk_logits_k is not None:
            # Top-k distillation: more efficient, focuses on important tokens
            per_token_kl = self._compute_topk_reverse_kl(
                student_logits=student_completion_logits,
                teacher_logits=teacher_completion_logits,
                mask=completion_mask,
                topk=self.topk_logits_k,
                temperature=1.0,
            )
        else:
            # Full vocabulary distillation
            per_token_kl = self._compute_reverse_kl(
                student_logits=student_completion_logits,
                teacher_logits=teacher_completion_logits,
                mask=completion_mask,
                temperature=1.0,
            )

        # Compute masked mean loss
        loss = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

        # Log metrics (including rewards for monitoring)
        self._log_metrics(completion_mask, per_token_kl, loss, rewards_per_func)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics with averaging."""
        metrics = {key: sum(vals) / len(vals) for key, vals in self._metrics.items()}
        logs = {**logs, **metrics}

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()


class RemoteKnowledgeDistillationTrainer(KnowledgeDistillationTrainer):
    """
    Trainer for On-Policy Knowledge Distillation with Remote Teacher API.

    Instead of loading teacher model locally, this trainer calls a remote API
    (e.g., vLLM server) to get teacher's top-k logprobs.

    The training loop:
    1. Student generates completions from prompts (on-policy)
    2. Teacher API returns top-k logprobs for student's completions
    3. Student learns to match teacher's distribution via KL divergence

    Args:
        model: Pre-initialized student model
        teacher_api_base: URL of the teacher API (e.g., "http://localhost:8000/v1")
        processor: Pre-initialized processor for the student model
        reward_funcs: Optional list of reward functions for monitoring
        args: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        data_collator: Collate function from dataset
    """

    def __init__(
        self,
        model: PreTrainedModel,
        teacher_api_base: str,
        processor: ProcessorMixin,
        args: TrainingArguments,
        reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        data_collator: Optional[Callable] = None,
    ):
        # Initialize parent Trainer directly, not KnowledgeDistillationTrainer
        # since we don't have a local teacher model
        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )

        set_seed(args.seed, device_specific=True)

        self.num_generations = args.num_generations
        self.topk_logits_k = args.topk_logits_k

        if reward_funcs is not None:
            self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        else:
            self.reward_funcs = []

        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        # Remote teacher API setup
        self.teacher_api_base = teacher_api_base
        self.teacher_model_name = self._get_teacher_model_name()
        logging.info(f"Remote teacher API: {teacher_api_base}")
        logging.info(f"Teacher model: {self.teacher_model_name}")

        # No local teacher model
        self.teacher_model = None
        self.teacher_model_processor = None
        self.is_step_audio_think_mode = self.teacher_model_name == "Step-Audio-R1.1"

    def _get_teacher_model_name(self) -> str:
        """Get the teacher model name from the API."""
        try:
            response = requests.get(f"{self.teacher_api_base}/models")
            response.raise_for_status()
            models = response.json()
            return models["data"][0]["id"]
        except Exception as e:
            logging.warning(f"Failed to get teacher model name: {e}")
            return "unknown"

    @staticmethod
    def _encode_audio_to_base64(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """Encode a numpy audio array to base64 WAV format."""
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        audio_data = buffer.read()
        return base64.b64encode(audio_data).decode("utf-8")

    def _build_prompt_from_meta(self, original_prompt: list) -> list:
        """Build prompt in the format expected by the teacher API."""
        text_content = ""
        for msg in original_prompt:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text_content = item.get("text", "")
                            break
                elif isinstance(content, str):
                    text_content = content

        return [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "audio_data"},
                {"type": "text", "text": text_content}
            ]
        }]

    def _call_teacher_api(
        self,
        prompt: list,
        audio_array: np.ndarray,
        answer_text: str,
        top_k: int = 64,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Call Teacher API with audio and get response with prompt_logprobs.

        Args:
            prompt: Prompt messages
            audio_array: Audio data as numpy array
            answer_text: Student's generated completion (prefill for teacher)
            top_k: Number of top-k logprobs to return
            sample_rate: Audio sample rate

        Returns:
            dict with topk_indices, topk_logprobs, actual_ids, actual_tokens, seq_len
        """
        audio_base64 = self._encode_audio_to_base64(audio_array, sample_rate)

        # Check if teacher model is Qwen3-Omni-30B-A3B-Captioner (audio-only input)
        is_qwen_omni3_captioner = self.teacher_model_name == "Qwen3-Omni-30B-A3B-Captioner"

        # Build messages
        messages = []
        for msg in prompt:
            new_msg = {"role": msg["role"]}
            if isinstance(msg.get("content"), list):
                new_content = []
                for item in msg["content"]:
                    if item.get("type") == "audio" or "audio_url" in item:
                        new_content.append({
                            "type": "input_audio",
                            "input_audio": {"data": audio_base64, "format": "wav"}
                        })
                    elif item.get("type") == "text":
                        # Skip text content for Qwen3-Omni-30B-A3B-Captioner model
                        if not is_qwen_omni3_captioner:
                            new_content.append({"type": "text", "text": item.get("text", "")})
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content
            else:
                # For non-list content, skip if captioner model
                if not is_qwen_omni3_captioner:
                    new_msg["content"] = msg.get("content", "")
                else:
                    new_msg["content"] = []
            messages.append(new_msg)

        # Append assistant message with answer_text
        if self.is_step_audio_think_mode:
            # For Step-Audio-R1.1, prompt with <think>\n or without <think>\n are two different distributions.
            messages.append({"role": "assistant", "content": "<think>\n" + answer_text})
        else:
            messages.append({"role": "assistant", "content": answer_text})

        # Build payload
        payload = {
            "model": self.teacher_model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": 1,
            "temperature": 0.7,
            "prompt_logprobs": top_k,
        }

        # Send request
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.teacher_api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        return self._extract_assistant_logprobs(result, answer_text)

    def _extract_assistant_logprobs(self, result: dict, assistant_content: str) -> dict:
        """
        Extract top-k logprobs for the assistant's prefill content.

        The actual token at each position is the FIRST token in the JSON dict.

        Returns:
            dict with topk_indices, topk_logprobs, actual_ids, actual_tokens, seq_len
        """
        prompt_logprobs = result.get("prompt_logprobs", [])

        # Build token sequence from prompt_logprobs
        token_sequence = []
        for pos, logprob_dict in enumerate(prompt_logprobs):
            if logprob_dict is None:
                token_sequence.append({
                    "pos": pos, "token_id": None, "token": None,
                    "logprob_dict": None, "all_tokens": []
                })
                continue

            all_tokens = []
            actual_token_id = None
            actual_token = None

            for idx, (token_id, info) in enumerate(logprob_dict.items()):
                token_info = {
                    "token_id": int(token_id),
                    "token": info["decoded_token"],
                    "logprob": info["logprob"],
                    "rank": info.get("rank", 0)
                }
                all_tokens.append(token_info)
                # First token in dict is the actual token
                if idx == 0:
                    actual_token_id = int(token_id)
                    actual_token = info["decoded_token"]

            token_sequence.append({
                "pos": pos, "token_id": actual_token_id, "token": actual_token,
                "logprob_dict": logprob_dict, "all_tokens": all_tokens
            })

        # Find assistant content by marker
        accumulated = "".join([t["token"] for t in token_sequence if t["token"]])
        # TODO: fix me: the assistant marker is hardcoded, should be configurable
        assistant_markers = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
        if self.is_step_audio_think_mode:
            assistant_markers.append("<|EOT|><|BOT|>assistant\n<think>\n")
        else:
            assistant_markers.append("<|EOT|><|BOT|>assistant")
        assistant_marker_pos = -1

        for marker in assistant_markers:
            pos = accumulated.rfind(marker)
            if pos >= 0:
                assistant_marker_pos = pos + len(marker)
                break

        # Find end marker
        end_marker_pos = accumulated.find("<|im_end|>", assistant_marker_pos) if assistant_marker_pos >= 0 else -1
        if end_marker_pos < 0:
            # WAR: TODO: hardcoded for now
            end_marker_pos = accumulated.find("<|BOT|>assistant\n<think>", assistant_marker_pos)

        prefix_start_idx = None
        prefix_end_idx = None

        if assistant_marker_pos >= 0:
            char_count = 0
            for i, t in enumerate(token_sequence):
                if t["token"] is not None:
                    token_start = char_count
                    token_end = char_count + len(t["token"])
                    if token_end > assistant_marker_pos and prefix_start_idx is None:
                        prefix_start_idx = i
                    if end_marker_pos >= 0:
                        if token_start < end_marker_pos:
                            prefix_end_idx = i
                    else:
                        prefix_end_idx = i
                    char_count = token_end

        if prefix_start_idx is None or prefix_end_idx is None:
            # Fallback: estimate from content length
            estimated_tokens = len(assistant_content) // 3 + 5
            valid_positions = [i for i, t in enumerate(token_sequence) if t["logprob_dict"] is not None]
            if valid_positions:
                end_positions = valid_positions[-estimated_tokens:] if len(valid_positions) > estimated_tokens else valid_positions  # noqa: E501
                prefix_start_idx = end_positions[0]
                prefix_end_idx = end_positions[-1]
            else:
                return {"topk_indices": [], "topk_logprobs": [], "actual_ids": [], "actual_tokens": [], "seq_len": 0}

        # Extract tokens in assistant region
        actual_sequence = []
        for idx in range(prefix_start_idx, prefix_end_idx + 1):
            t = token_sequence[idx]
            if t["logprob_dict"] is None:
                continue
            if t["token"] and ("<|" in t["token"] or t["token"].strip() == ""):
                continue
            actual_sequence.append({
                "pos": idx,
                "token_id": t["token_id"],
                "token": t["token"],
                "logprob_dict": t["logprob_dict"]
            })

        # Extract top-k data
        topk_indices = []
        topk_logprobs = []
        actual_ids = []
        actual_tokens = []

        for item in actual_sequence:
            logprob_dict = item["logprob_dict"]
            sorted_items = sorted(logprob_dict.items(), key=lambda x: x[1]["logprob"], reverse=True)

            pos_indices = [int(tid) for tid, _ in sorted_items]
            pos_logprobs = [info["logprob"] for _, info in sorted_items]

            topk_indices.append(pos_indices)
            topk_logprobs.append(pos_logprobs)
            actual_ids.append(item["token_id"])
            actual_tokens.append(item["token"])

        return {
            "topk_indices": topk_indices,
            "topk_logprobs": topk_logprobs,
            "actual_ids": actual_ids,
            "actual_tokens": actual_tokens,
            "seq_len": len(actual_ids),
        }

    def _compute_reverse_kl_with_teacher_topk(
        self,
        student_logits: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute reverse KL divergence using teacher's top-k tokens.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_topk_indices: Teacher's top-k token indices [batch, seq_len, top_k]
            teacher_topk_logprobs: Teacher's top-k logprobs [batch, seq_len, top_k]
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        student_logits = student_logits / temperature
        # Has to clamp the teacher's top-k indices to the student's vocab size
        student_vocab_size = student_logits.shape[-1]
        teacher_topk_indices = teacher_topk_indices.clamp(max=student_vocab_size - 1)
        # Gather student logits at teacher's top-k indices
        student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)

        # Compute probabilities over top-k (re-normalized)
        student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        student_topk_probs = F.softmax(student_topk_logits, dim=-1)

        # Teacher logprobs need re-normalization over top-k
        teacher_topk_log_probs = F.log_softmax(teacher_topk_logprobs, dim=-1)

        # KL(student || teacher)
        per_token_kl = (student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)

        return per_token_kl

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute on-policy knowledge distillation loss with remote teacher.

        Training flow:
        1. Student generates completions (on-policy sampling)
        2. Call teacher API to get top-k logprobs for student's completions
        3. Student learns to match teacher's distribution via KL divergence
        """
        meta_data = inputs.pop("meta_data")
        original_prompts, audios = meta_data[1], meta_data[2]

        # Step 1: Generate completions from student model
        generated_strs, generated_ids, generated_mask = self._rollout(model, inputs)

        # Compute rewards for monitoring
        rewards_per_func = self._compute_rewards(generated_strs, meta_data)

        device = self.accelerator.device
        batch_size = len(generated_strs)
        student_prompt_length = inputs["input_ids"].size(1)

        # Collect all per_token_kl and masks for batched computation
        all_per_token_kl = []  # List of [1, seq_len] tensors
        all_masks = []         # List of [1, seq_len] tensors

        for batch_idx in range(batch_size):
            original_prompt = original_prompts[batch_idx]
            audio_array = audios[batch_idx]
            answer_text = generated_strs[batch_idx]

            if not answer_text.strip():
                # Skip empty generations
                continue

            try:
                # Step 2: Call teacher API
                prompt = self._build_prompt_from_meta(original_prompt)
                teacher_data = self._call_teacher_api(
                    prompt=prompt,
                    audio_array=audio_array,
                    answer_text=answer_text,
                    top_k=self.topk_logits_k or 64,
                    sample_rate=16000,
                )

                if teacher_data["seq_len"] == 0:
                    continue

                teacher_actual_ids = teacher_data["actual_ids"]
                teacher_seq_len = teacher_data["seq_len"]

                # Step 3: Build student input using teacher's actual_ids
                prompt_ids = inputs["input_ids"][batch_idx:batch_idx+1]
                completion_ids = torch.tensor([teacher_actual_ids], dtype=torch.long, device=device)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

                prompt_mask = inputs["attention_mask"][batch_idx:batch_idx+1]
                completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)
                attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)

                student_inputs = {
                    "input_ids": prompt_completion_ids,
                    "attention_mask": attention_mask,
                    "input_features": inputs["input_features"][batch_idx:batch_idx+1],
                    "feature_attention_mask": inputs["feature_attention_mask"][batch_idx:batch_idx+1],
                }

                # Compute student logits
                student_logits = self._get_per_token_logits(model, student_inputs)
                student_completion_logits = student_logits[:, student_prompt_length - 1:student_prompt_length - 1 + teacher_seq_len, :]  # noqa: E501

                # Prepare teacher top-k data
                raw_topk_indices = teacher_data["topk_indices"]
                raw_topk_logprobs = teacher_data["topk_logprobs"]

                # Pad to consistent length
                max_topk_len = max(len(row) for row in raw_topk_indices) if raw_topk_indices else 0
                padded_indices = []
                padded_logprobs = []
                for indices_row, logprobs_row in zip(raw_topk_indices, raw_topk_logprobs):
                    pad_len = max_topk_len - len(indices_row)
                    padded_indices.append(indices_row + [0] * pad_len)
                    padded_logprobs.append(logprobs_row + [-100.0] * pad_len)

                teacher_topk_indices = torch.tensor(padded_indices, dtype=torch.long, device=device).unsqueeze(0)
                teacher_topk_logprobs = torch.tensor(padded_logprobs, dtype=torch.float32, device=device).unsqueeze(0)

                # Step 4: Compute KL divergence
                per_token_kl = self._compute_reverse_kl_with_teacher_topk(
                    student_logits=student_completion_logits,
                    teacher_topk_indices=teacher_topk_indices,
                    teacher_topk_logprobs=teacher_topk_logprobs,
                    temperature=1.0,
                )

                all_per_token_kl.append(per_token_kl)  # [1, seq_len]
                all_masks.append(completion_mask)      # [1, seq_len]

            except Exception as e:
                logging.warning(f"Error processing sample {batch_idx}: {e}")
                continue

        if not all_per_token_kl:
            # Return zero loss if all samples failed
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pad sequences to same length for batched computation
        max_len = max(t.size(1) for t in all_per_token_kl)
        padded_kl = []
        padded_masks = []
        for kl, mask in zip(all_per_token_kl, all_masks):
            pad_len = max_len - kl.size(1)
            if pad_len > 0:
                kl = F.pad(kl, (0, pad_len), value=0)
                mask = F.pad(mask, (0, pad_len), value=0)
            padded_kl.append(kl)
            padded_masks.append(mask)

        per_token_kl = torch.cat(padded_kl, dim=0)       # [batch, max_len]
        completion_mask = torch.cat(padded_masks, dim=0)  # [batch, max_len]

        # Consistent loss calculation with masking (same as parent class)
        loss = ((per_token_kl * completion_mask).sum(dim=1) /
                completion_mask.sum(dim=1).clamp(min=1)).mean()

        # Reuse _log_metrics from parent class
        self._log_metrics(completion_mask, per_token_kl, loss, rewards_per_func)
        return loss
