# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : generation_utils_e2e.py
"""
import copy
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation.beam_constraints import (DisjunctiveConstraint,
                                                      PhrasalConstraint)
from transformers.generation.beam_search import (BeamSearchScorer,
                                                 ConstrainedBeamSearchScorer)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (GenerateEncoderDecoderOutput,
                                           GenerateNonBeamOutput,
                                           GenerateOutput, GenerationConfig,
                                           GenerationMixin, GenerationMode,
                                           LogitsProcessorList,
                                           StoppingCriteriaList, logging)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.utils import ModelOutput

from .sample_util import ras_sampling

logger = logging.get_logger(__name__)


class GenerationWithCE(GenerationMixin):
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.GenerationMixin._greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin._contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin._sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin._beam_search`] if `num_beams>1` and
          `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin._beam_sample`] if `num_beams>1`
          and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin._group_beam_search`], if `num_beams>1`
          and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin._constrained_beam_search`], if
          `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* by calling [`~generation.GenerationMixin._assisted_decoding`], if
            `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        custom_params={},
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object used to stream the generated sequences.
            Generated tokens are passed through `streamer.put(token_ids)` and
            the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """  # noqa: E501
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate
                                    and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate
                                  and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate
                                       and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states else None)

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size,
                                          dtype=torch.long,
                                          device=input_ids.device)

        # ******************************** speech generator **************************
        speech_decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None
        speech_llm_prefix_attention_mask = custom_params.get(
            'speech_llm_prefix_attention_mask', None)
        speech_llm_prefix_embeds = custom_params.get('speech_llm_prefix_embeds',
                                                     None)
        is_speech_generate_run = custom_params.get('is_speech_generate_run',
                                                   True)
        if speech_llm_prefix_embeds is not None and is_speech_generate_run:
            speech_model_kwargs = {}
            for key, value in model_kwargs.items():
                speech_model_kwargs[key] = copy.deepcopy(value)
            speech_model_kwargs['inputs_embeds'] = speech_llm_prefix_embeds
            speech_model_kwargs[
                'attention_mask'] = speech_llm_prefix_attention_mask
            speech_model_kwargs = self._get_initial_cache_position(
                cur_len, input_ids.device, speech_model_kwargs)
        # ****************************************************************************
        model_kwargs = self._get_initial_cache_position(cur_len,
                                                        input_ids.device,
                                                        model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(
            model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(
                generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config,
                                                  **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        # ******************************** speech generator **************************
        text_label = custom_params.get('text_label', None)
        is_teacher_forcing = custom_params.get('is_teacher_forcing', None)
        prompt_unit = custom_params.get('prompt_unit', None)
        multi_turn_for_decoder_indices = custom_params.get(
            'multi_turn_for_decoder_indices', [])
        text_llm_prefix_len = custom_params.get('text_llm_prefix_len', None)
        max_speech_unit_length = int(
            custom_params.get('max_speech_unit_length', 50 * 35))
        enable_thinking = bool(custom_params.get('enable_thinking', False))
        text_n_step = 0
        is_speech_generate_run_again = False

        n_step = 0
        if prompt_unit is not None:
            generated_units = prompt_unit.clone()
        else:
            generated_units = torch.zeros((1, 0),
                                          dtype=torch.int32,
                                          device=input_ids.device)
        this_peer_unit_finished = False if is_speech_generate_run else True
        this_peer_text_finished = False

        speech_generator_model_kwargs = copy.deepcopy(model_kwargs)
        speech_generator_model_kwargs.pop('inputs_embeds')
        is_speech_gen_first_step = True
        unit_embedd = 0
        # ************************************************************

        while self._has_unfinished_sequences(this_peer_finished
                                             and this_peer_unit_finished,
                                             synced_gpus,
                                             device=input_ids.device):
            speech_outputs = None
            if is_speech_generate_run and speech_llm_prefix_embeds is not None and text_n_step == 0:
                # prepare model inputs
                speech_model_inputs = self.prepare_inputs_for_generation(
                    input_ids, **speech_model_kwargs)
                # prepare variable output controls (note: some models won't accept all output controls)
                speech_model_inputs.update(
                    {"output_attentions": output_attentions}
                    if output_attentions else {})
                speech_model_inputs.update(
                    {"output_hidden_states": output_hidden_states}
                    if output_hidden_states else {})
                speech_model_inputs.update({"start_layer": 0})
                speech_outputs = self(**speech_model_inputs, return_dict=True)

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions}
                                if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states}
                                if output_hidden_states else {})
            model_inputs.update({"start_layer": 0})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            if not this_peer_finished:
                # synced_gpus: don't waste resources running the code we don't need;
                # kwargs must be updated before skipping
                # model_kwargs = self._update_model_kwargs_for_generation(
                #     outputs,
                #     model_kwargs,
                #     is_encoder_decoder=self.config.is_encoder_decoder,
                # )
                if synced_gpus and this_peer_finished:
                    continue

                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration  # noqa: E501
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(
                    copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids,
                                                     next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores, )
                    if output_logits:
                        raw_logits += (next_token_logits, )
                    if output_attentions:
                        decoder_attentions += ((outputs.decoder_attentions, )
                                               if self.config.is_encoder_decoder
                                               else (outputs.attentions, ))
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions, )

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states, )
                            if self.config.is_encoder_decoder else
                            (outputs.hidden_states, ))
                        if is_speech_generate_run and text_n_step == 0 and speech_llm_prefix_embeds is not None:
                            # speech_decoder_hidden_states += (
                            #     (speech_outputs.decoder_hidden_states,)
                            #     if self.config.is_encoder_decoder
                            #     else (speech_outputs.hidden_states,)
                            # )
                            speech_decoder_hidden_states_ = []
                            for index, (speech_llm_hidden,
                                        text_llm_hidden) in enumerate(
                                            zip(speech_outputs.hidden_states,
                                                outputs.hidden_states)):
                                speech_decoder_hidden_states_.append(
                                    torch.cat([
                                        speech_llm_hidden,
                                        text_llm_hidden[:,
                                                        text_llm_prefix_len:, :]
                                    ],
                                              dim=-2))
                            speech_decoder_hidden_states += (
                                tuple(speech_decoder_hidden_states_), )
                        else:
                            speech_decoder_hidden_states += (
                                (outputs.decoder_hidden_states, )
                                if self.config.is_encoder_decoder else
                                (outputs.hidden_states, ))

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs,
                                                    num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                if is_teacher_forcing:
                    next_tokens = text_label[0:1]
                    text_label = text_label[1:]
                text_n_step += 1

                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                        1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                if not enable_thinking or self.config.model_type == 'qwen2':  # not enable think
                    if text_n_step > 1:
                        is_speech_generate_run_again = is_speech_generate_run and True
                else:  # enable thinking  # </think>=151668 /n/n=271
                    if (input_ids.shape[-1] >= 3 and input_ids[0][-3].item() == 151668) or \
                            (input_ids.shape[-1] >= 4 and input_ids[0][-4].item() == 10):
                        is_speech_generate_run_again = is_speech_generate_run and True

                # if streamer is not None:
                #     streamer.put(next_tokens.cpu())

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
                if this_peer_finished and not is_speech_generate_run_again:
                    this_peer_unit_finished = True  # 设置的max_text_token太小了，还没有</think>就用完了。
                cur_len += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                # del outputs

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

            if is_speech_generate_run_again:
                if is_speech_gen_first_step:
                    input_hidden_states = []
                    for hidden in speech_decoder_hidden_states:
                        if multi_turn_for_decoder_indices != [] and hidden[
                                self.freeze_layer + 1].size(1) != 1:
                            # print(f"multi_turn_for_decoder_indices: {multi_turn_for_decoder_indices}")
                            for start_index, end_index in multi_turn_for_decoder_indices:
                                input_hidden_states.append(
                                    hidden[self.freeze_layer + 1]
                                    [:, start_index:end_index if end_index != -1 else None, :])
                        else:
                            input_hidden_states.append(
                                hidden[self.freeze_layer + 1])
                    input_hidden_states = torch.cat(input_hidden_states, dim=-2)
                    is_speech_gen_first_step = False

                    input_hidden_states_len = input_hidden_states.size(1)
                    cache_position = torch.arange(input_hidden_states_len,
                                                  device=input_ids.device)
                    attention_mask = torch.ones(1,
                                                input_hidden_states_len,
                                                dtype=torch.long,
                                                device=input_ids.device)
                    # speech_generator_model_kwargs.pop('inputs_embeds')
                    speech_generator_model_kwargs[
                        'cache_position'] = cache_position
                    speech_generator_model_kwargs[
                        'attention_mask'] = attention_mask
                elif not this_peer_text_finished:  # and text_n_step <= 200
                    input_hidden_states = decoder_hidden_states[-1][
                        self.freeze_layer + 1] + unit_embedd
                else:
                    input_hidden_states = unit_embedd

                speech_model_inputs = self.prepare_inputs_for_generation(
                    input_ids=torch.zeros((1, 0),
                                          dtype=torch.int32,
                                          device=input_ids.device),
                    inputs_embeds=input_hidden_states,
                    **speech_generator_model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                speech_model_inputs.update(
                    {"output_attentions": output_attentions}
                    if output_attentions else {})
                speech_model_inputs.update(
                    {"output_hidden_states": output_hidden_states}
                    if output_hidden_states else {})
                speech_model_inputs.update(
                    {"start_layer": self.freeze_layer + 1})

                speech_outputs = self.speech_generator(**speech_model_inputs,
                                                       return_dict=True)

                # 生成unit
                cur_units = torch.zeros((1, 0),
                                        dtype=torch.int32,
                                        device=input_ids.device)
                last_hidden_states = speech_outputs['hidden_states'][-1][:, -1:, :]
                for tok_iter in range(self.config.num_tok_per_group):
                    logits: torch.Tensor = self.head_unit[tok_iter](
                        last_hidden_states).squeeze(1)
                    # if logits_ is None:
                    #     logits_ = logits
                    # else:
                    #     logits_ = torch.cat([logits_, logits], dim=0)
                    # previous_tokens = generated_units[:, -15:-7] if generated_units.shape[1] > 7 else None
                    # if this_peer_finished == False:
                    #     logits[:, 4096] = logits[:, 4097] = -1e10
                    # item_next = torch.argmax(logits, dim=-1, keepdim=True)
                    # item_next = sample(logits, previous_tokens=previous_tokens, repetition_penalty=1.35, temperature=0.7, index=n_step % 1200)[0]  # noqa: E501
                    item_next = ras_sampling(
                        logits.log_softmax(dim=-1).squeeze(dim=0),
                        generated_units).unsqueeze(0)

                    if item_next == 4097 or torch.argmax(
                            logits, dim=-1) == 4097 or torch.argmax(
                                logits, dim=-1
                            ) == 4096 or n_step > max_speech_unit_length:
                        this_peer_unit_finished = True
                        break
                    cur_units = torch.cat([cur_units, item_next], dim=-1)
                    generated_units = torch.cat([generated_units, item_next], dim=-1)
                    n_step += 1
                if not this_peer_unit_finished:
                    unit_embedd = self.emb_unit(cur_units).sum(1, keepdim=True)
                    unit_embedd = unit_embedd

                    speech_generator_model_kwargs = self._update_model_kwargs_for_generation(
                        speech_outputs,
                        speech_generator_model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                    )
                this_peer_text_finished = this_peer_finished

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    speech_units=generated_units,
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                    List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        custom_params={},
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            use_model_defaults (`bool`, *optional*):
                When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
                generation configuration (`model.generation_config`), as opposed to the global defaults
                (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
                `True`.
            custom_generate (`str`, *optional*):
                A string containing the name of a huggingface.co repository. If provided, the custom `generate`
                function defined in that reposity's `custom_generate/generate.py` file will be executed instead of the
                standard `generate` method. Note that the logic is for generation is entirely defined in that
                repository, and the return type may be different from the standard `generate` method.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A
            [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model
                (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model
                (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 0. If requested, load an arbitrary generation recipe from the Hub and
        #    run it instead
        if custom_generate is not None:
            trust_remote_code = kwargs.pop("trust_remote_code", None)
            # Get all `generate` arguments in a single variable. Custom functions
            # are responsible for handling them: they receive the same inputs as
            # `generate`, only with `model` instead of `self`. They can access to
            # methods from `GenerationMixin` through `model`.
            global_keys_to_exclude = {"self", "kwargs"}
            generate_arguments = {
                key: value
                for key, value in locals().items()
                if key not in global_keys_to_exclude
            }
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs)
            return custom_generate_function(model=self, **generate_arguments)

        # 1. Handle `generation_config` and kwargs that might update it, and
        #    validate the `.generate()` call
        tokenizer = kwargs.pop(
            "tokenizer",
            None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop(
            "assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(
            assistant_model,
            tokenizer,
            assistant_tokenizer,
        )

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (
                is_deepspeed_zero3_enabled()
                or is_fsdp_managed_module(self)
            ) and dist.get_world_size() > 1

        logits_processor = (
            logits_processor
            if logits_processor is not None
            else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask",
                                                     None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config,
                                     kwargs_has_attention_mask,
                                     device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (generation_config._pad_token_tensor is not None
                    and batch_size > 1 and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0):
                logger.warning(
                    "A decoder-only architecture is being used, but "
                    "right-padding was detected! For correct generation "
                    "results, please set `padding_side='left'` when "
                    "initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with `inputs_embeds` forwarding must use caching
        # (otherwise we can't detect whether we are generating the first new
        # token or not, and we only want to use the embeddings for the first
        # new token)
        if (not self.config.is_encoder_decoder and model_input_name == "inputs_embeds"):
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs[
                "attention_mask"] = self._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config, model_kwargs)
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(
                    model_kwargs["attention_mask"].shape) > 2:
                raise ValueError(
                    "`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name,
                generation_config)

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            (input_ids, model_kwargs) = (
                self._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size,
                    model_input_name=model_input_name,
                    model_kwargs=model_kwargs,
                    decoder_start_token_id=generation_config.
                    _decoder_start_token_tensor,
                    device=inputs_tensor.device,
                )
            )
        else:
            if model_input_name == "input_ids":
                input_ids = inputs_tensor
            else:
                input_ids = model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get(
            "max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get(
            "min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to
        # avoid computing the whole logit matrix. This can save a lot of memory
        # during the first forward pass. Note that assisted decoding dynamically
        # overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep(
        ) and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length,
                                        has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by
        #   the parameters in `generation_config`.
        # - different models have a different cache name expected by the model
        #   (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache
        #   length
        max_cache_length = generation_config.max_length - 1
        if (inputs_tensor.shape[1] != input_ids_length
                and model_input_name == "inputs_embeds"
                and not self.config.is_encoder_decoder):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(generation_config, model_kwargs,
                                           assistant_model, batch_size,
                                           max_cache_length, device)

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a "
                "device type different"
                f" than your model's device. `input_ids` is on"
                f" {input_ids.device.type}, whereas the model is on"
                f" {self.device.type}. You may experience unexpected behaviors"
                " or slower generation. Please make sure that you have put"
                " `input_ids` to the correct device by calling for example"
                f" input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs)

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}.")
            if batch_size > 1:
                raise ValueError(
                    "assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation in [
                    "static", "hybrid", "sliding_window"
            ]:
                raise ValueError(
                    "assisted generate is not supported with Static cache classes`"
                )
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                target_tokenizer=tokenizer,
                assistant_tokenizer=assistant_tokenizer,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
                )
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE,
                                 GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                custom_params=custom_params,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE,
                                 GenerationMode.BEAM_SEARCH):
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam sample
            result = self._beam_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (not isinstance(generation_config.force_words_ids, list)
                        or len(generation_config.force_words_ids) == 0):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list)
                               for token_ids in word_ids):
                            typeerror()
                        if any(
                                any((not isinstance(token_id, int)
                                     or token_id < 0) for token_id in token_ids)
                                for token_ids in word_ids):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0)
                               for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (generation_config.return_legacy_cache is True
                and hasattr(result, "past_key_values") and getattr(
                    result.past_key_values, "to_legacy_cache") is not None):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is
            passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """  # noqa: E501

    speech_units: torch.LongTensor = None
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
