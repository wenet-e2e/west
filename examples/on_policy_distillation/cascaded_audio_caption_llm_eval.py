"""Test MMAU using OpenAI API with audio captions.

This script takes audio captions from a JSON file and uses GPT to predict
the correct answer option based on the caption content.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = """You are an expert audio analyst. Based on a detailed audio caption, you need to answer questions about the audio content by selecting the most appropriate option."""  # NOQA: E501

USER_PROMPT_TEMPLATE = """Below is a detailed caption describing an audio clip:

{caption}

---

Question: {question}

Options:
{options}

Based on the audio caption above, select the single best answer from the options provided.
Reply with ONLY the option text (e.g., "Man" or "A woman"), without any additional explanation or punctuation."""


def parse_args():
    parser = argparse.ArgumentParser(description="Test MMAU with OpenAI API using audio captions")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with audio captions")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for results")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default="https://inference-api.nvidia.com", help="API base URL")
    parser.add_argument("--model", type=str, default="azure/openai/gpt-5.2", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens for generation")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if output exists")
    parser.add_argument("--retry_count", type=int, default=3, help="Number of retries on failure")
    parser.add_argument("--retry_delay", type=float, default=1.0, help="Delay between retries in seconds")
    return parser.parse_args()


def format_options(choices: list) -> str:
    """Format choices as a lettered list."""
    return "\n".join([f"{choice}" for i, choice in enumerate(choices)])


def get_prediction(
    client: OpenAI,
    caption: str,
    question: str,
    choices: list,
    model: str,
    temperature: float,
    max_tokens: int,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """Get model prediction for a single example."""
    options_text = format_options(choices)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        caption=caption,
        question=question,
        options=options_text,
    )

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1:
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logging.error(f"All {retry_count} attempts failed: {e}")
                return ""


def process_single_item(args_tuple):
    """Process a single item. Used for parallel processing."""
    client, item, model, temperature, max_tokens, retry_count, retry_delay = args_tuple

    caption = item.get("model_response", "")
    question = item.get("question", "")
    choices = item.get("choices", [])

    prediction = get_prediction(
        client=client,
        caption=caption,
        question=question,
        choices=choices,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_count=retry_count,
        retry_delay=retry_delay,
    )

    return item, prediction


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(f"Arguments: {args}")

    # Check if output already exists
    if not args.force and os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        logging.info(f"Output file {args.output_file} already exists. Use --force to regenerate.")
        return

    # Create output directory if needed
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Load input data
    logging.info(f"Loading data from {args.input_file}")
    with open(args.input_file, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} examples")

    # Process in parallel
    results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create tasks
        futures = {}
        for idx, item in enumerate(data):
            task_args = (
                client,
                item,
                args.model,
                args.temperature,
                args.max_tokens,
                args.retry_count,
                args.retry_delay,
            )
            future = executor.submit(process_single_item, task_args)
            futures[future] = idx

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            try:
                item, prediction = future.result()
                # Create result with updated model_output
                result = item.copy()
                result["model_output"] = prediction
                print(prediction)
                results[idx] = result
            except Exception as e:
                logging.error(f"Error processing item {idx}: {e}")
                results[idx] = data[idx].copy()
                results[idx]["model_output"] = ""

    # Save results
    logging.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculate accuracy
    correct = 0
    total = 0
    for result in results:
        if result and result.get("model_output") and result.get("answer"):
            total += 1
            pred = result["model_output"].strip()
            answer = result["answer"].strip()
            # Check if prediction matches answer (case-insensitive, handle letter prefix)
            if pred.lower() == answer.lower():
                correct += 1
            # Also check if prediction starts with the answer or vice versa
            elif pred.lower().startswith(answer.lower()) or answer.lower().startswith(pred.lower()):
                correct += 1

    if total > 0:
        accuracy = correct / total * 100
        logging.info(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")

    logging.info("Done!")


if __name__ == "__main__":
    main()
