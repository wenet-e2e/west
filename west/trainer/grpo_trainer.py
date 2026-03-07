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
GRPO Trainer for Qwen2-Audio, Qwen-omni models.

Simplified implementation for Audio Question Answering with custom reward functions.
Based on R1-V: https://github.com/Deep-Agent/R1-V
"""

from collections import defaultdict
from typing import Callable, Optional, Union

import torch
import transformers
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (GenerationConfig, PreTrainedModel, ProcessorMixin,
                          Trainer, TrainingArguments)
from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax

# Type alias for reward functions
RewardFunc = Callable[[list, list], list[float]]


class GRPOTrainer(Trainer):
    """
    Trainer for Group Relative Policy Optimization (GRPO).

    Reference: DeepSeekMath paper (https://huggingface.co/papers/2402.03300)

    Args:
        model: Pre-initialized Qwen2-Audio model
        ref_model: Pre-initialized reference model (for KL penalty)
        processor: Pre-initialized processor for the model
        reward_funcs: List of reward functions (callables)
        reward_weights: Optional list of weights for each reward function.
            If None, uses equal weights (1.0 for each function).
        args: GRPOConfig training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        data_collator: Collate function from dataset
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        processor: ProcessorMixin,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: TrainingArguments,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        data_collator: Optional[Callable] = None,
        reward_weights: Optional[list[float]] = None,
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

        self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]

        # Set reward weights: default to equal weights if not provided
        if reward_weights is None:
            self.reward_weights = [1.0] * len(self.reward_funcs)
        else:
            assert len(reward_weights) == len(self.reward_funcs), (
                f"reward_weights length ({len(reward_weights)}) "
                f"must match reward_funcs length ({len(self.reward_funcs)})"
            )
            self.reward_weights = reward_weights

        self.num_generations = args.num_generations
        self.beta = args.beta

        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.num_generations,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        self.ref_model = ref_model
        if self.ref_model is not None:
            assert self.is_deepspeed_enabled, "Reference model requires DeepSpeed to be enabled"
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)

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

    def _get_per_token_logps(self, model, inputs: dict) -> torch.Tensor:
        """Compute per-token log probabilities."""
        if hasattr(model, "module") and hasattr(model.module, "thinker"):
            model = model.module.thinker
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        ).logits

        # Shift: exclude last logit and first input_id
        logits = logits[:, :-1, :]
        input_ids = inputs["input_ids"][:, 1:]

        return selective_log_softmax(logits, input_ids)

    def _prepare_logprob_inputs(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> dict:
        """Prepare inputs for log probability computation."""
        # Concatenate prompt and generated ids
        prompt_ids = inputs["input_ids"].repeat_interleave(self.num_generations, dim=0)
        prompt_completion_ids = torch.cat([prompt_ids, generated_ids], dim=1)

        # Expand masks for num_generations
        prompt_mask = inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
        attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)
        input_features = inputs["input_features"].repeat_interleave(self.num_generations, dim=0)
        feature_mask = inputs["feature_attention_mask"].repeat_interleave(self.num_generations, dim=0)

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        }

    def _compute_logprobs(self, model, logprob_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for actor and reference models."""
        per_token_logps = self._get_per_token_logps(model, logprob_inputs)

        with torch.inference_mode():
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, logprob_inputs)

        return per_token_logps, ref_per_token_logps

    def _compute_rewards(self, generated_strs: list[str], meta_data: list) -> torch.Tensor:
        """Compute rewards for generated completions.

        Returns:
            rewards_per_func: Rewards from each reward function [num_samples, num_funcs]
        """
        solutions = meta_data[0]
        solutions_expanded = [s for s in solutions for _ in range(self.num_generations)]

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(solutions_expanded), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            rewards = reward_func(
                hypos_list=generated_strs,
                ground_truth_list=solutions_expanded,
            )
            rewards_per_func[:, i] = torch.tensor(rewards, dtype=torch.float32, device=device)

        return rewards_per_func

    def _compute_advantages(self, total_rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using group-relative normalization (GRPO).

        Returns:
            advantages: Normalized advantages
            std_rewards: Std reward per group (for logging)
        """
        grouped_rewards = total_rewards.view(-1, self.num_generations)
        mean_rewards = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        std_rewards = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        advantages = (total_rewards - mean_rewards) / (std_rewards + 1e-4)
        return advantages, std_rewards

    def _compute_kl(
        self,
        per_token_logps: torch.Tensor,
        ref_per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token KL divergence.

        KL(π || π_ref) ≈ exp(log π_ref - log π) - (log π_ref - log π) - 1
        """
        return torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    def _compute_per_token_loss(
        self,
        per_token_logps: torch.Tensor,
        per_token_kl: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token GRPO loss."""
        # Policy gradient (importance ratio = 1 for on-policy)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        return per_token_loss

    def _log_metrics(
        self,
        generated_mask: torch.Tensor,
        rewards_per_func: torch.Tensor,
        total_rewards: torch.Tensor,
        std_rewards: torch.Tensor,
        per_token_kl: torch.Tensor,
    ):
        """Log training metrics."""
        completion_lengths = self.accelerator.gather_for_metrics(generated_mask.sum(1)).float()
        self._metrics["completion_length"].append(completion_lengths.mean().item())
        self._metrics["completion_length_min"].append(completion_lengths.min().item())
        self._metrics["completion_length_max"].append(completion_lengths.max().item())

        for i, reward_func in enumerate(self.reward_funcs):
            func_name = reward_func.__name__
            mean_reward = self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            self._metrics[f"rewards/{func_name}"].append(mean_reward)

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(total_rewards).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_rewards).mean().item()
        )

        mean_kl = ((per_token_kl * generated_mask).sum(dim=1) / generated_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    # ==================== Main Training Loop ====================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss for a batch of inputs."""
        # Extract meta_data for reward computation
        meta_data = inputs.pop("meta_data")

        # Step 1: Generate completions
        generated_strs, generated_ids, generated_mask = self._rollout(model, inputs)

        # Step 2: Prepare inputs and compute log probabilities
        logprob_inputs = self._prepare_logprob_inputs(inputs, generated_ids, generated_mask)
        per_token_logps, ref_per_token_logps = self._compute_logprobs(model, logprob_inputs)

        # Slice to get only completion token logps
        prompt_length = inputs["input_ids"].size(1)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Step 3: Compute rewards
        rewards_per_func = self._compute_rewards(generated_strs, meta_data)
        # Apply reward weights
        reward_weights_tensor = torch.tensor(
            self.reward_weights, device=rewards_per_func.device, dtype=rewards_per_func.dtype
        )
        total_rewards = (rewards_per_func * reward_weights_tensor).sum(dim=1)

        # Step 4: Compute advantages
        advantages, std_rewards = self._compute_advantages(total_rewards)

        # Step 5: Compute KL divergence
        per_token_kl = self._compute_kl(per_token_logps, ref_per_token_logps)

        # Step 6: Compute loss
        per_token_loss = self._compute_per_token_loss(per_token_logps, per_token_kl, advantages)
        loss = ((per_token_loss * generated_mask).sum(dim=1) / generated_mask.sum(dim=1)).mean()

        # Log metrics
        self._log_metrics(generated_mask, rewards_per_func, total_rewards, std_rewards, per_token_kl)

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
