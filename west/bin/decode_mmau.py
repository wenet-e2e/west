"""Decode script for GRPO-trained models using vLLM.

This script runs inference on the MMAU audio understanding benchmark using
Qwen2-Audio models with vLLM for efficient batch processing.
"""

# Must set environment variables before any imports that might initialize CUDA
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Also set multiprocessing start method
import multiprocessing  # noqa: E402

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import torchaudio  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from west.utils.constants import TEMPLATE_MAP  # noqa: E402


def extract_answer(output_str: str) -> str:
    """Extract content from <answer> tags in the model output."""
    match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
    return match.group(1) if match else output_str


def extract_think(output_str: str) -> str:
    """Extract content from <think> tags in the model output."""
    match = re.search(r"<think>(.*?)</think>", output_str, re.DOTALL)
    return match.group(1) if match else ""


def parse_args():
    parser = argparse.ArgumentParser(description="Test MMAU with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="model dir")
    parser.add_argument("--data_file", type=str, required=True, help="test file")
    parser.add_argument("--audio_dir", type=str, required=True, help="audio dir")
    parser.add_argument("--out_file", type=str, required=True, help="output file for test")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--force", action="store_true", help="force test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor parallel size for vLLM")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--max_audio_duration_in_seconds", type=int, default=30, help="max audio duration in seconds")
    parser.add_argument(
        "--template", type=str, default="default", choices=["default", "think", "new", "caption"], help="prompt type"
    )
    return parser.parse_args()


def _get_audio(wav_path, max_audio_duration_in_seconds=30, target_sample_rate=16000):
    """Load audio file and return (numpy_array, sample_rate) tuple for vLLM."""
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    audio = waveform[0].numpy()
    # cut off to 3000 * 16000 = 4800000 samples for qwen2-audio model
    max_audio_duration_in_samples = max_audio_duration_in_seconds * target_sample_rate
    audio = audio[:max_audio_duration_in_samples]
    return (audio, target_sample_rate)


def _get_prompt(obj_dict, processor, template="default"):
    """Generate prompt for Qwen2-Audio model using processor's chat template.

    Args:
        obj_dict: Dict containing 'question', 'choices', and 'audio_id'
        processor: AutoProcessor for applying chat template
        template: 'default' or 'think' template type

    Returns:
        Formatted prompt string
    """
    # Select template based on type
    prompt_template = TEMPLATE_MAP.get(template, TEMPLATE_MAP["default"])

    # Format the prompt text
    prompt_text = prompt_template.format(
        question=obj_dict['question'],
        choices=obj_dict['choices']
    )

    # Construct message in the format expected by apply_chat_template
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": obj_dict.get('audio_id', 'test.wav')},
            {"type": "text", "text": prompt_text}
        ]
    }]

    # Apply chat template to get the final prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.disable(logging.WARNING)
    logging.info(args)

    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        logging.info(f"The {args.out_file} exists. Do not regenerate it.")
        return

    out_dir = os.path.abspath(os.path.dirname(args.out_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize vLLM
    llm = LLM(
        model=args.model_path,
        max_model_len=8000,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=args.tensor_parallel_size,
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    logging.info(f"Loaded processor from {args.model_path}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Load test data
    with open(args.data_file, "r") as f:
        datas = json.load(f)

    all_outputs = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i:i + batch_size]

        batch_inputs = []
        for bd in batch_data:
            audio_path = os.path.join(args.audio_dir, bd["audio_id"])
            audio_data = _get_audio(audio_path, args.max_audio_duration_in_seconds)
            prompt = _get_prompt(bd, processor, args.template)

            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": [audio_data]}
            })

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            all_outputs.append(generated_text)
            print(generated_text)

        print(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")

    final_output = []
    for input_example, model_output in zip(datas, all_outputs):
        original_output = model_output
        model_answer = extract_answer(original_output).strip()
        model_think = extract_think(original_output).strip()
        result = input_example.copy()
        result["model_output"] = model_answer
        result["model_think"] = model_think
        result["model_response"] = original_output
        final_output.append(result)

    output_path = args.out_file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
