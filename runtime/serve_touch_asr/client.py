# Copyright (c) 2026 Pengshen Zhang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================
# Qwen3-Omni FastAPI vLLM Client
# Supports streaming and non-streaming audio transcription
# Reference: OpenAI Chat API client pattern
# ==============================================

import argparse
import base64
import io
import json
import os
import string
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple

import librosa
import numpy as np
import requests
import soundfile as sf
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen3OmniMoeProcessor


# ==============================================
# Utility Functions
# ==============================================
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file

    Args:
        file_path: Path to JSONL file

    Returns:
        List of data from JSONL file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def encode_audio_base64(audio_path: str) -> str:
    """Encode audio file to base64

    Args:
        audio_path: Path to audio file

    Returns:
        Base64 encoded audio string
    """
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def parse_input(
    input_str: str
) -> Tuple[List[str], Optional[List[Dict[str, Any]]]]:
    """Parse input string

    Supports single audio file, JSONL file, or comma-separated paths.

    Args:
        input_str: Input string

    Returns:
        audio_paths: List of audio file paths
        item_info: Original info for each audio if input is JSONL,
            None otherwise
    """
    audio_paths = []
    item_info = None

    if os.path.isfile(input_str) and input_str.endswith('.jsonl'):
        # Load data from JSONL file
        data = load_jsonl(input_str)
        audio_paths = []
        item_info = []
        for item in data:
            # Support multiple field names: wav, audio_path, audio
            wav_path = (item.get('wav', '') or item.get('audio_path', '') or
                        item.get('audio', ''))
            if wav_path:
                audio_paths.append(wav_path)
                item_info.append({
                    'key': item.get('key', ''),
                    'ref': item.get('txt', '') or item.get('ref', ''),
                    'original_item': item
                })
    elif os.path.isfile(input_str):
        # Single audio file path
        audio_paths = [input_str]
        item_info = None
    else:
        # Assume comma-separated audio path list
        audio_paths = [path.strip() for path in input_str.split(',')]
        item_info = None

    return audio_paths, item_info


class OutputFileManager:
    """Output file manager (context manager)

    Manages writing to output file or stdout.
    """

    def __init__(self, output_path: Optional[str] = None):
        """Initialize output file manager

    Args:
            output_path: Output file path, None for stdout
        """
        self.output_path = output_path
        self.fout = None
        self.verbose = False
        self.out_file_name = None

    def __enter__(self) -> 'OutputFileManager':
        """Enter context manager

        Returns:
            OutputFileManager instance
        """
        if self.output_path is not None:
            self.fout = open(self.output_path, 'w', encoding='utf-8')
            self.out_file_name = os.path.split(self.output_path)[-1]
            self.verbose = True
        else:
            self.fout = sys.stdout
            self.out_file_name = 'stdout'
            self.verbose = False
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> bool:
        """Exit context manager

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            False, don't suppress exceptions
        """
        if self.output_path is not None and self.fout:
            self.fout.close()
        return False

    def write(self, text: str, add_newline: bool = False) -> None:
        """Write text

        Args:
            text: Text to write
            add_newline: Whether to add newline after text
        """
        self.fout.write(text)
        if add_newline:
            self.fout.write('\n')
        self.fout.flush()


def extract_text_from_response(
    response: str,
    is_thinking: bool = False,
    is_last_chunk: bool = True
) -> str:
    """Extract text from model response

    Args:
        response: Model response
        is_thinking: Whether Thinking model
        is_last_chunk: Whether last chunk (for streaming, intermediate
            chunks remove all punctuation)
    Returns:
        Extracted text
    """
    # Handle Thinking model reasoning tags
    if is_thinking and '</think>' in response:
        response = response.split('</think>')[-1]

    # Strip leading whitespace
    response = response.strip()

    # Remove all punctuation for intermediate chunks
    if not is_last_chunk and response:
        # English punctuation
        english_punctuation = string.punctuation
        # Chinese punctuation
        chinese_punctuation = (
            '，。！？；: 、""''（）【】《》〈〉「」『』〔〕…—～·'
        )
        all_punctuation = english_punctuation + chinese_punctuation

        # Replace punctuation with space (preserve word separation)
        # maketrans requires equal length, replace punct with space
        translation_table = str.maketrans(
            all_punctuation, ' ' * len(all_punctuation))
        response = response.translate(translation_table)

        # Remove trailing spaces
        response = response.rstrip()

    # Clean extra spaces (merge multiple spaces into one)
    response = ' '.join(response.split())

    return response.strip()


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert object to JSON-serializable format

    Handles numpy arrays, scalars, etc., converting to Python types.

    Args:
        obj: Object to convert (numpy array, scalar, dict, list, etc.)

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


class StreamingAudioProcessor:
    """Audio processor

    Converts audio to base64-encoded audio data.
    Input: Audio file
    Output: Base64-encoded audio data (full, chunk, or accumulated)
    """

    def __init__(self, chunk_s: float = 1.0, sr: int = 16000):
        """Initialize audio processor

        Args:
            chunk_s: Seconds per chunk
            sr: Target sample rate
        """
        self.chunk_s = chunk_s
        self.sr = sr
        # Audio cache: {audio_path: (audio, sr)}
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file (with cache, load only once)

        Args:
            audio_path: Path to audio file
        Returns:
            audio: Audio data (numpy array)
            sr: Sample rate
        """
        # Return cached if available
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        # Load and cache
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        self._audio_cache[audio_path] = (audio, sr)
        return audio, sr

    def clear_cache(self, audio_path: Optional[str] = None) -> None:
        """Clear audio cache

        Args:
            audio_path: If specified, clear only this audio's cache;
                otherwise clear all cache
        """
        if audio_path is None:
            self._audio_cache.clear()
        elif audio_path in self._audio_cache:
            del self._audio_cache[audio_path]

    def _audio_to_base64(self, audio: np.ndarray, sr: int) -> str:
        """Convert numpy array audio to base64-encoded WAV

        Args:
            audio: Audio data (numpy array)
            sr: Sample rate
        Returns:
            Base64-encoded audio string
        """
        # Ensure audio is float32, range [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Write audio to in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)

        # Encode to base64
        audio_bytes = buffer.read()
        base64_str = base64.b64encode(audio_bytes).decode('utf-8')

        return base64_str

    def _audio_to_base64_with_duration(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[str, float]:
        """Convert audio to base64 and calculate duration

        Args:
            audio: Audio data (numpy array)
            sr: Sample rate
        Returns:
            base64_audio: Base64-encoded audio data
            duration: Audio duration (seconds)
        """
        base64_audio = self._audio_to_base64(audio, sr)
        duration = len(audio) / sr
        return base64_audio, duration

    def _get_audio_segment(
        self,
        audio: np.ndarray,
        sr: int,
        mode: Literal['current', 'accumulated', 'full'],
        chunk_idx: Optional[int] = None
    ) -> np.ndarray:
        """Get audio segment by mode

        Args:
            audio: Full audio data
            sr: Sample rate
            mode: Output mode
            chunk_idx: Chunk index (required for 'current' or
                'accumulated' mode)
        Returns:
            audio_segment: Audio segment
        """
        if mode == 'full':
            return audio
        elif mode == 'current':
            if chunk_idx is None:
                raise ValueError(
                    "chunk_idx required for 'current' mode")
            chunk_samples = int(self.chunk_s * sr)
            start = (chunk_idx - 1) * chunk_samples
            end = min(start + chunk_samples, len(audio))
            return audio[start:end]
        elif mode == 'accumulated':
            if chunk_idx is None:
                raise ValueError(
                    "chunk_idx required for 'accumulated' mode")
            chunk_samples = int(self.chunk_s * sr)
            end = min(chunk_idx * chunk_samples, len(audio))
            return audio[:end]
        else:
            raise ValueError(f"Unsupported output mode: {mode}")

    def get_chunk_audio_base64(
        self,
        audio_path: str,
        mode: Literal['current', 'accumulated']
    ) -> Generator[Tuple[str, int, float], None, None]:
        """Get base64-encoded chunk data (generator)

        Args:
            audio_path: Path to audio file
            mode: Output mode
                - 'current': Return only current chunk data
                - 'accumulated': Return all data up to current chunk
        Yields:
            base64_audio: Base64-encoded audio data
            sr: Sample rate
            duration: Audio duration (seconds)
        """
        audio, sr = self.load_audio(audio_path)
        chunk_samples = int(self.chunk_s * sr)
        total_samples = len(audio)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        for chunk_idx in range(1, num_chunks + 1):
            audio_segment = self._get_audio_segment(
                audio, sr, mode, chunk_idx=chunk_idx)
            base64_audio, duration = (
                self._audio_to_base64_with_duration(audio_segment, sr))
            yield base64_audio, sr, duration

    def get_full_audio_base64(
        self, audio_path: str
    ) -> Tuple[str, int, float]:
        """Get full audio base64-encoded data

        Args:
            audio_path: Path to audio file
        Returns:
            base64_audio: Base64-encoded audio data
            sr: Sample rate
            duration: Audio duration (seconds)
        """
        audio, sr = self.load_audio(audio_path)
        audio_segment = self._get_audio_segment(audio, sr, 'full')
        base64_audio, duration = (
            self._audio_to_base64_with_duration(audio_segment, sr))
        return base64_audio, sr, duration


# ==============================================
# ASR Client Class
# ==============================================
class ASRClient:
    """FastAPI vLLM ASR Client

    Supports streaming and non-streaming audio transcription
    (Reference: OpenAI Chat API client pattern)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model_path: Optional[str] = None,
        system_prompt: str = "",
        chunk_s: float = 1.0,
        sr: int = 16000,
        is_thinking: Optional[bool] = False
    ):
        """Initialize client

        Args:
            server_url: Server URL
            model_path: Model path (for loading processor)
            system_prompt: System prompt
            chunk_s: Seconds per chunk (for streaming input)
            sr: Target sample rate
            is_thinking: Whether Thinking model
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.processor = None
        self.model_path = model_path
        self.system_prompt = system_prompt

        # Check if Thinking model: only via parameter, default False
        self.is_thinking = is_thinking

        # Create audio processor instance
        self.audio_processor = StreamingAudioProcessor(
            chunk_s=chunk_s, sr=sr)

        # Load processor if model path provided
        if model_path:
            print("[Init]")
            print(f"  Loading Processor: {model_path}")
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                model_path)
            print("  Processor loaded!")

    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        response = self.session.get(f"{self.server_url}/health")
        response.raise_for_status()
        return {"status": "healthy"}

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        response = self.session.get(f"{self.server_url}/v1/models")
        response.raise_for_status()
        return response.json()

    def _build_prompt(
        self, language: str, prompt: Optional[str] = None
    ) -> str:
        """Build prompt (public method)

        Args:
            language: Language code
            prompt: Custom prompt, None for default

        Returns:
            Prompt string
        """
        if prompt is not None:
            return prompt

        if language == "zh":
            if self.is_thinking:
                return "请将这段中文语音转换为纯文本。"
            elif self.system_prompt:
                return self.system_prompt
            else:
                return "请将这段中文语音转换为纯文本。不要分析过程，直接输出转录结果。 /no_think"
        elif language == "en":
            if self.is_thinking:
                return "Transcribe the English audio into text."
            else:
                return ("Transcribe the English audio into text. "
                        "Don't analyze the process, just output the "
                        "transcription result. /no_think")
        else:
            return "Transcribe the audio into text. /no_think"

    def _build_messages_from_base64(
        self,
        base64_audio: str,
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Build message list from base64 audio

        Args:
            base64_audio: Base64-encoded audio data
            prompt: Prompt

        Returns:
            Message list
        """
        # Add current user message (audio + prompt)
        content_list = [
            {"type": "audio",
             "audio": f"data:audio/wav;base64,{base64_audio}"},
            {"type": "text", "text": prompt}
        ]

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        return messages

    def _prepare_request_from_base64(
        self,
        base64_audio: str,
        prompt: str,
        session_id: str,
        prev_hyp: str = "",
        temperature: float = 0.01,
        top_p: float = 0.1,
        top_k: int = 1,
        max_tokens: int = 512,
        stream: bool = False,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Prepare request data from base64 audio

        Args:
            base64_audio: Base64-encoded audio data
            prompt: Prompt
            session_id: Session ID (required, UUID format, for tracking
                multiple chunks of same audio)
            prev_hyp: Previous chunk recognition result (appended to
                prompt)
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            max_tokens: Max tokens
            stream: Whether streaming
            debug: Whether debug mode
        Returns:
            Request data
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized, provide model_path")

        # 1. Build messages
        messages = self._build_messages_from_base64(
            base64_audio, prompt)

        # 2. Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # Add think tag (based on model type)
        if self.is_thinking:
            text = text + "<think>\n\n</think>"

        # Append previous chunk result (ref: asr_streaming.py:183)
        if prev_hyp:
            text = text + prev_hyp

        # 3. Process multimodal info
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True)

        # 4. Build multimodal data
        multi_modal_data = {}
        if audios is not None:
            multi_modal_data['audio'] = make_json_serializable(audios)
        if images is not None:
            multi_modal_data['image'] = make_json_serializable(images)
        if videos is not None:
            multi_modal_data['video'] = make_json_serializable(videos)

        # Debug log
        if debug:
            print("[Request]")
            print(f"  prompt: ##{text}##")
            print(f"  multi_modal keys: {list(multi_modal_data.keys())}")
            print(f"  prev_hyp: {prev_hyp}")

        # 5. Build request
        request_data = {
            "model": self.model_path or "qwen3-omni",
            "prompt": text,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": {"use_audio_in_video": True},
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
            "session_id": session_id  # Required field
        }

        # Ensure all data is JSON-serializable
        request_data = make_json_serializable(request_data)

        return request_data

    def _apply_history_rollback(
        self, history: str, rollback: float = 0.0
    ) -> str:
        """Apply rollback strategy to history text

        Args:
            history: Accumulated text result
            rollback: Rollback amount
                - >= 1: Rollback by character count (remove last N chars)
                - < 1: Rollback by percentage (remove last N%)
                - 0: No rollback

        Returns:
            Rolled back text
        """
        if not history or rollback <= 0:
            return history

        if rollback >= 1:
            # Rollback by character count
            rollback_chars = int(rollback)
            if rollback_chars >= len(history):
                return ""
            return history[:-rollback_chars]
        else:
            # Rollback by percentage
            keep_length = int(len(history) * (1 - rollback))
            return history[:keep_length]

    def transcribe(
        self,
        audio_path: str,
        language: str = "zh",
        prompt: Optional[str] = None,
        temperature: float = 0.01,
        top_p: float = 0.1,
        top_k: int = 1,
        max_tokens: int = 512,
        stream_input: bool = False,
        prev_hyp_rollback: float = 0.0,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Execute audio transcription

        Args:
            audio_path: Path to audio file
            language: Language code (zh/en)
            prompt: Custom prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Max generation tokens
            stream_input: Whether to use streaming input
                - True: accumulated mode (each chunk includes all
                    previous audio)
                - False: full mode (send complete audio at once)
            prev_hyp_rollback: Context rollback amount (only when
                stream_input=True)
                - >= 1: Rollback by char count (remove last N chars)
                - < 1: Rollback by percentage (e.g., 0.1 = 10%)
                - 0: No rollback (default)
            debug: Whether to show debug info (prints intermediate
                results for streaming)

        Returns:
            Transcription result (last chunk result or all results
            list for streaming)
        """
        # Build prompt
        prompt = self._build_prompt(language, prompt)

        if stream_input:
            # Streaming input: accumulated mode, send chunk by chunk
            return self._transcribe_stream_input(
                audio_path, prompt, temperature, top_p, top_k,
                max_tokens, prev_hyp_rollback, debug)
        else:
            # Non-streaming input: full mode, send complete audio
            return self._transcribe_non_stream_input(
                audio_path, prompt, temperature, top_p, top_k,
                max_tokens, debug)

    def _transcribe_non_stream_input(
        self,
        audio_path: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Non-streaming input transcription"""
        # Generate session_id (also used for non-streaming tracking)
        session_id = str(uuid.uuid4())

        # Get full audio base64
        base64_audio, sr, duration = (
            self.audio_processor.get_full_audio_base64(audio_path))

        # Build request
        request_data = self._prepare_request_from_base64(
            base64_audio, prompt, session_id, "", temperature, top_p,
            top_k, max_tokens, False, debug)

        # Send request
        response = self.session.post(
            f"{self.server_url}/v1/chat/completions",
            json=request_data,
            timeout=300)
        response.raise_for_status()
        result = response.json()

        # Extract text
        raw_text = result["choices"][0]["message"]["content"]
        clean_text = extract_text_from_response(
            raw_text, self.is_thinking)

        if debug:
            print("[Response]")
            print(f"  session_id: {session_id}")
            print(f"  raw_text_len: {len(raw_text)}, "
                  f"clean_text_len: {len(clean_text)}")
            if clean_text:
                clean_preview = (clean_text[:150] + "..."
                                 if len(clean_text) > 150
                                 else clean_text)
                print(f"  clean_text: {clean_preview}")

        return {
            "text": clean_text,
            "raw_response": raw_text,
            "stream": False
        }

    def _transcribe_stream_input(
        self,
        audio_path: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        prev_hyp_rollback: float = 0.0,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Streaming input transcription (accumulated mode)

        Supports appending response to next chunk's input.

        """
        # Generate unique session_id for streaming (full UUID)
        # All chunks use same session_id for server tracking
        session_id = str(uuid.uuid4())

        # Get audio chunk generator (accumulated mode)
        audio_generator = self.audio_processor.get_chunk_audio_base64(
            audio_path, mode='accumulated')

        all_results = []
        accumulated_text = ""  # Store accumulated complete result
        chunk_idx = 0

        # Collect all chunk info first to determine if last
        chunks_list = list(audio_generator)
        total_chunks = len(chunks_list)

        if debug:
            print("[Streaming]")
            print(f"  session_id: {session_id}")
            print(f"  Start processing, total chunks: {total_chunks}")
            if prev_hyp_rollback > 0:
                rollback_desc = (
                    f"{int(prev_hyp_rollback)}chars"
                    if prev_hyp_rollback >= 1
                    else f"{prev_hyp_rollback*100:.1f}%")
                print(f"  Rollback strategy: {rollback_desc}")

        # Send request chunk by chunk
        for chunk_idx_in_list, (base64_audio, sr, duration) in enumerate(
                chunks_list):
            chunk_idx += 1
            is_last_chunk = (chunk_idx_in_list == total_chunks - 1)

            # Apply rollback to history (if not last chunk)
            prev_hyp_to_send = self._apply_history_rollback(
                accumulated_text,
                prev_hyp_rollback if not is_last_chunk else 0)

            if debug:
                print("\n" + "=" * 60)
                print(f"[Chunk {chunk_idx}/{total_chunks}]")
                print(f"  session_id: {session_id}")
                print(f"  duration: {duration:.2f}s")
                print(f"  accumulated_text_len: "
                      f"{len(accumulated_text)}chars")
                if (accumulated_text and prev_hyp_rollback > 0 and
                        not is_last_chunk):
                    rollback_chars = (len(accumulated_text) -
                                      len(prev_hyp_to_send))
                    print(f"  rollback: {len(prev_hyp_to_send)}chars"
                          f"(rollback {rollback_chars}chars)")

            # Build request (append rolled-back prev_hyp, same session_id)
            request_data = self._prepare_request_from_base64(
                base64_audio, prompt, session_id, prev_hyp_to_send,
                temperature, top_p, top_k, max_tokens, False, debug)

            # Send request and get result
            response = self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=request_data,
                timeout=300)
            response.raise_for_status()
            result = response.json()

            # Extract text (model returns incremental text)
            raw_text = result["choices"][0]["message"]["content"]
            incremental_text = extract_text_from_response(
                raw_text, self.is_thinking, is_last_chunk=is_last_chunk)

            # Accumulate text (accumulated mode)
            if (len(accumulated_text) > 0 and
                    accumulated_text[-1].isalpha() and
                    len(incremental_text) > 0 and
                    incremental_text[0].isalpha()):
                accumulated_text = accumulated_text + ' '
            accumulated_text = accumulated_text + incremental_text

            # Add chunk info
            chunk_result = {
                'text': incremental_text,  # Current chunk incremental
                'chunk_idx': chunk_idx,
                'chunk_duration': duration,
                'incremental_text': incremental_text,
                'accumulated_text': accumulated_text,
                'prev_context_sent': prev_hyp_to_send,
                'raw_response': raw_text
            }
            all_results.append(chunk_result)

            # Print intermediate results in debug mode
            if debug:
                # Replace newlines with spaces for single-line display
                incremental_display = (
                    incremental_text.replace('\n', ' ')
                    .replace('\r', ' ').strip())
                if incremental_display:
                    incremental_preview = (
                        incremental_display[:80] + "..."
                        if len(incremental_display) > 80
                        else incremental_display)
                    print(f"  cur_hyp: {incremental_preview}")

        if debug:
            print("\n[Streaming]")
            print(f"  Complete, session_id: {session_id}")
            print(f"  Processed: {len(all_results)} chunks")
            print(f"  Final text length: {len(accumulated_text)}chars")

        # Return result
        if len(all_results) == 1:
            # Single chunk, return directly
            return all_results[0]
        else:
            # Multiple chunks, return summary
            return {
                "text": accumulated_text,
                "chunks": all_results,
                "num_chunks": len(all_results),
                "stream_input": True
            }

    def batch_transcribe(
        self,
        audio_paths: List[str],
        language: str = "zh",
        prompt: Optional[str] = None,
        temperature: float = 0.01,
        top_p: float = 0.1,
        top_k: int = 1,
        max_tokens: int = 512,
        stream_input: bool = False,
        prev_hyp_rollback: float = 0.0,
        max_workers: int = 10,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """Batch transcription (using concurrent requests)

        Args:
            audio_paths: List of audio file paths
            language: Language code
            prompt: Custom prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Max generation tokens
            stream_input: Whether to use streaming input
            prev_hyp_rollback: Context rollback amount (only when
                stream_input=True)
            max_workers: Concurrent thread count
            debug: Whether to show debug info

        Returns:
            List of transcription results, each contains audio_path
            and result
        """
        # Build prompt
        prompt = self._build_prompt(language, prompt)

        # Debug mode limit: process only first 10 for debugging
        original_count = len(audio_paths)
        if debug and len(audio_paths) > 10:
            print("[Debug Limit]")
            print(f"  Limit to first 10 (total {original_count})")
            audio_paths = audio_paths[:10]  # Create new list

        if debug:
            print("[Batch Transcription]")
            print(f"  Start processing {len(audio_paths)} audio files")
            print(f"  Input mode: "
                  f"{'streaming' if stream_input else 'non-streaming'}")

        # Define function to process single audio
        def process_single_audio(audio_path: str) -> Dict[str, Any]:
            """Process single audio file"""
            # Check audio file
            if not os.path.exists(audio_path):
                return {
                    "audio_path": audio_path,
                    "text": "",
                    "status": "failed",
                    "error": "file_not_found"
                }

            try:
                # Call transcribe method
                result = self.transcribe(
                    audio_path=audio_path,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    stream_input=stream_input,
                    prev_hyp_rollback=prev_hyp_rollback,
                    debug=False  # No debug for batch processing
                )

                return {
                    "audio_path": audio_path,
                    "text": result.get("text", ""),
                    "status": "success",
                    "raw_response": result
                }
            except Exception as e:
                return {
                    "audio_path": audio_path,
                    "text": "",
                    "status": "failed",
                    "error": str(e)
                }

        # Concurrent processing
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Batch execute requests, always use progress bar
            futures = {
                executor.submit(process_single_audio, path): path
                for path in audio_paths
            }

            # Use dict to save results, maintain original order
            results_dict = {}
            for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc="ASR Batch Transcription"):
                audio_path = futures[future]
                result = future.result()
                # Find original index
                idx = audio_paths.index(audio_path)
                results_dict[idx] = result

            # Sort results by index
            results = [results_dict[idx]
                       for idx in sorted(results_dict.keys())]

        elapsed_time = time.time() - start_time

        if debug:
            success_count = sum(1 for r in results
                                if r['status'] == 'success')
            failed_count = sum(1 for r in results
                               if r['status'] == 'failed')
            avg_time = (round(elapsed_time / len(audio_paths), 2)
                        if audio_paths else 0)
            print("[Batch Transcription]")
            print(f"  Complete, total: {len(audio_paths)}, "
                  f"success: {success_count}, failed: {failed_count}")
            print(f"  Total time: {round(elapsed_time, 2)}s, "
                  f"avg: {avg_time}s/item")

        return results


# ==============================================
# Command Line Interface
# ==============================================


def format_batch_result(
    result: Dict[str, Any],
    audio_path: str,
    item_info: Optional[Dict[str, Any]],
    stream_input: bool,
    output_is_jsonl: bool
) -> str:
    """Format batch processing result (public method)

    Args:
        result: Transcription result
        audio_path: Path to audio file
        item_info: Original info (if from JSONL)
        stream_input: Whether streaming input used
        output_is_jsonl: Whether output JSONL format

    Returns:
        Formatted string
    """
    if output_is_jsonl or item_info:
        # JSONL format output
        if item_info:
            result_item = {
                'key': item_info['key'],
                'ref': item_info['ref'],
                'hyp': result.get('text', ''),
                'stream_input': stream_input
            }
        else:
            result_item = {
                'key': os.path.basename(audio_path),
                'ref': '',
                'hyp': result.get('text', ''),
                'stream_input': stream_input
            }
        if result['status'] == 'failed':
            result_item['error'] = result.get('error', 'unknown_error')
        return json.dumps(result_item, ensure_ascii=False) + '\n'
    else:
        # Plain text output
        text = result.get('text', '')
        return text + '\n' if text else ''


def cmd_transcribe(args):
    """Unified transcription command

    Decides single or batch processing based on batch_size.
    """
    # Convert is_thinking: True if specified, else None
    is_thinking = True if args.is_thinking else None

    client = ASRClient(
        args.server,
        args.model,
        args.system_prompt,
        is_thinking=is_thinking)

    # Unified input processing
    audio_paths, item_info = parse_input(args.input)

    if not audio_paths:
        print("[Error] No valid audio files found")
        return

    # Debug mode limit: process only first 10 for debugging
    original_count = len(audio_paths)
    if args.debug and len(audio_paths) > 10:
        print(f"[Debug Limit] Limit to first 10 "
              f"(total {original_count})")
        audio_paths = audio_paths[:10]  # Create new list
        if item_info:
            item_info = item_info[:10]

    if args.debug:
        print("[Config]")
        print(f"  Server: {args.server}, Model: {args.model}")
        print(f"  Input: {args.input}, "
              f"Audio count: {len(audio_paths)}")
        mode_str = ('streaming' if args.stream_input else 'non-streaming')
        print(f"  Mode: {mode_str}, batch_size: {args.batch_size}")
        if args.stream_input and args.prev_hyp_rollback > 0:
            rollback_desc = (
                f"{int(args.prev_hyp_rollback)}chars"
                if args.prev_hyp_rollback >= 1
                else f"{args.prev_hyp_rollback*100:.1f}%")
            print(f"  Rollback strategy: {rollback_desc}")

    # Health check
    try:
        client.health_check()
        if args.debug:
            print("[Service] Status: healthy\n")
    except Exception as e:
        if args.debug:
            print(f"[Error] Service unavailable: {e}\n")
        return

    # Decide processing mode based on batch_size and audio count
    use_single = (args.batch_size == 1 and len(audio_paths) == 1)

    # Use output file manager
    with OutputFileManager(args.output) as out_mgr:
        if use_single:
            # Single processing: use transcribe method
            if args.debug:
                print("[Single Processing]")
                print(f"  Start processing: {audio_paths[0]}")

            t0 = time.time()
            try:
                result = client.transcribe(
                    audio_paths[0],
                    language=args.lang,
                    prompt=args.prompt,
                    stream_input=args.stream_input,
                    prev_hyp_rollback=args.prev_hyp_rollback,
                    debug=args.debug)

                output_text = result['text']
                elapsed = time.time() - t0

                # Output result
                out_mgr.write(output_text,
                              add_newline=not out_mgr.verbose)

                if args.debug:
                    print(f"\n[Single Processing] Complete, "
                          f"time: {elapsed:.2f}s, "
                          f"text_len: {len(output_text)}chars")
                    if args.stream_input:
                        print(f"  chunks: {result.get('num_chunks', 1)}")
                    if out_mgr.verbose:
                        print(f"  Output: {args.output}")

            except Exception as e:
                print(f"[Error] Transcription failed: {e}")
        else:
            # Batch processing: use batch_transcribe method
            if args.debug:
                print(f"[Batch Processing] Start processing, "
                      f"batch_size: {args.batch_size}")

            results = client.batch_transcribe(
                audio_paths=audio_paths,
                language=args.lang,
                prompt=args.prompt,
                stream_input=args.stream_input,
                prev_hyp_rollback=args.prev_hyp_rollback,
                max_workers=args.batch_size,
                debug=args.debug)

            # Output results
            output_is_jsonl = (args.output and
                               args.output.endswith('.jsonl'))

            for i, result in enumerate(results):
                audio_path = result['audio_path']
                info = (item_info[i]
                        if item_info and i < len(item_info)
                        else None)

                formatted_result = format_batch_result(
                    result, audio_path, info, args.stream_input,
                    output_is_jsonl)

                if formatted_result:
                    out_mgr.write(formatted_result)

            if args.debug and out_mgr.verbose:
                print(f"[Output] Results saved to: {args.output}")


def cmd_health(args):
    """Health check"""
    # Convert is_thinking: True if specified, else None
    is_thinking = True if args.is_thinking else None
    client = ASRClient(args.server, args.model, is_thinking=is_thinking)

    try:
        client.health_check()
        print("[Service] Status: healthy")

        models_info = client.list_models()
        print("[Model List]")
        for model in models_info.get('data', []):
            print(f"  - {model.get('id')}")
    except Exception as e:
        print(f"[Error] Service unavailable: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Qwen3-Omni FastAPI vLLM Client '
                    '(supports streaming and non-streaming)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--server', '-s', type=str, default='http://localhost:8000',
        help='Server URL')
    parser.add_argument(
        '--model', '-m', type=str, required=False,
        help='Model path (for loading processor)')
    parser.add_argument(
        '--system-prompt', type=str,
        default='Transcribe this audio, output only text, no explanation.',
        help='System prompt')
    parser.add_argument(
        '--is-thinking', action='store_true',
        help='Specify as Thinking model. If not specified, auto-detect '
             'from model path')

    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # Unified transcription command (auto-select single or batch)
    transcribe_parser = subparsers.add_parser(
        'transcribe',
        help='Audio transcription (auto-select single or batch by '
             'batch_size)')
    transcribe_parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Input: single audio file, JSONL file, or comma-separated '
             'audio paths')
    transcribe_parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output file path (.jsonl or text, stdout if not specified)')
    transcribe_parser.add_argument(
        '--lang', '-l', type=str, default='zh', choices=['zh', 'en'])
    transcribe_parser.add_argument(
        '--prompt', '-p', type=str,
        default='请转录这段音频的内容，只输出转录文本，不要添加任何解释。')
    transcribe_parser.add_argument(
        '--stream-input', action='store_true',
        help='Use streaming input (accumulated mode), False for full mode')
    transcribe_parser.add_argument(
        '--prev-hyp-rollback', type=float, default=0.0,
        help='Context rollback (only for stream-input): >=1 by chars, '
             '<1 by percentage (e.g., 0.1=10%), 0=no rollback (default)')
    transcribe_parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size: 1=single (batch_size=1 and 1 audio) or '
             'sequential batch, >1=concurrent batch')
    transcribe_parser.add_argument(
        '--debug', '-d', action='store_true',
        help='Show debug information')
    transcribe_parser.set_defaults(func=cmd_transcribe)

    # Health check
    health_parser = subparsers.add_parser('health', help='Health check')
    health_parser.set_defaults(func=cmd_health)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
