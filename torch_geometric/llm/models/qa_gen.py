import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import re
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import faiss
import numpy as np
import pandas as pd
import torch
from jinja2 import Template
from json_repair import repair_json
from openai import OpenAI

logger = logging.getLogger(__name__)

artifact_extraction_prompt = """
Analyze the following transcript and extract semantic artifacts that would be \
    valuable for generating high-quality question-answer pairs.

TRANSCRIPT:
{{text}}

ARTIFACT TYPES TO EXTRACT:
{{artifact_descriptions}}

INSTRUCTIONS:
1. Extract EXACTLY {{max_artifacts}} artifacts for each relevant type - NO MORE
2. Focus ONLY on the most significant and informative elements
3. Provide clear, concise descriptions for each artifact
4. Include context about why each artifact is important
5. Ensure artifacts are specific and actionable for Q&A generation
6. CRITICAL: Do NOT exceed {{max_artifacts}} items per type.
Stop generating after {{max_artifacts}} items.

Output your evaluation as a valid JSON object with the following structure:
```json
{
  "key_concepts": [
    {"text": "concept name", "description": "detailed description", \
        "importance": "why it's important"},
    ...
  ],
  "relationships": [
    {"text": "relationship description", \
        "description": "detailed explanation", "importance": "relevance"},
    ...
  ],
  "themes": [...],
  "entities": [...],
  "processes": [...],
  "insights": [...],
  "technical_terms": [...],
  "contextual_factors": [...]
}

Please ensure your output is ONLY the JSON object with no preamble \
    or additional text.
"""
artifact_extraction_prompt_template = Template(artifact_extraction_prompt)

llm_rank_artifacts_prompt = """You are an expert at evaluating semantic \
    artifacts for question-answer generation.

Given the following extracted artifacts from a transcript, select the \
    TOP {{top_k}} artifacts that would be MOST VALUABLE for generating \
        insightful and complex question-answer pairs.

AVAILABLE ARTIFACTS:
{{artifacts_text}}

SELECTION CRITERIA:
1. Prioritize artifacts that enable multi-hop reasoning questions
2. Choose artifacts with rich relationships and connections
3. Select concepts that can generate "why" and "how" questions
4. Prefer artifacts that reveal cause-effect relationships
5. Include artifacts that support synthesis and analysis questions
6. Consider diversity - avoid selecting only one type

RESPOND WITH ONLY THE NUMBERS of your top {{top_k}} choices in \
    order of importance.
Format: [1, 5, 3, 2, 7]

Your selection (top {{top_k}} numbers only):"""
llm_rank_artifacts_prompt_template = Template(llm_rank_artifacts_prompt)

generate_timeline_prompt = """
You are an expert timeline curator creating a rich event log from \
    transcript segments.

Objectives:
- Surface at most {{max_events}} pivotal moments that show progress, \
    decisions, or issues in the session.
- Each event must cite one of the provided segment IDs (1-based numbering) \
    with exact timestamps taken from that segment.
- Provide enough detail (key actions, quotes, implications) so downstream \
    systems can build cross-part reasoning.

Input Segments (use only these, never invent content):
{% for seg in segments %}
- Segment {{seg.segment_id}} \
    ({{seg.start_time}} - {{seg.end_time}}): {{seg.text}}
{% endfor %}

Required JSON output:
{
  "timeline": [
    {
      "segment_id": <int>,
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "headline": "<= 18 words capturing the moment",
      "key_details": \
        "2 sentences summarising the action/outcome (<= 220 chars)",
      "key_quote": \
        "Exact quote or concise paraphrase from the segment (<= 200 chars)",
      "impact": \
        "Why this matters for securing/operating AI agents (<= 140 chars)",
      "entities": ["Speaker or system names involved" (0-4 items)]
    }
  ]
}

Strict rules:
1. Use only segment IDs provided; keep start/end timestamps within that \
    segment's window.
2. Do not output more than {{max_events}} events; ensure they are ordered \
    chronologically.
3. If you cannot find a quote, use a faithful paraphrase and label it clearly.
4. All strings must be plain text (no markdown) and JSON must be valid.

Return only the JSON object.
Please ensure your output is ONLY the JSON object with no preamble \
    or additional text.
"""
generate_timeline_prompt_template = Template(generate_timeline_prompt)


class Backend(Enum):
    NIM = 'nim'
    VLLM = 'vllm'


class TaskStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    RETRYING = 'retrying'
    VALIDATING = 'validating'


class LLMClient:
    def __init__(
        self,
        generation_model: str,
        evaluation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        backend: str = 'nim',
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,  # Deprecated, kept for compat
        gpu_memory_utilization_generation: Optional[float] = None,
        gpu_memory_utilization_embedding: Optional[float] = None,
        gpu_memory_utilization_evaluation: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        enable_sleep_mode: bool = True,
    ):
        """Initialize LLMClient with generation, evaluation, and \
            embedding models.

        Args:
            generation_model: Model name/path for text generation
            evaluation_model: \
                Model name/path for evaluation (defaults to generation_model)
            embedding_model: \
                Model name/path for embeddings (defaults to generation_model)
            backend: Backend to use (NIM or VLLM)
            api_key: API key (for NIM backend)
            tensor_parallel_size: Number of GPUs for tensor parallelism (VLLM)
            gpu_memory_utilization: GPU memory utilization 0-1 (VLLM) \
                [deprecated, use per-model params]
            gpu_memory_utilization_generation: GPU memory for generation model
            gpu_memory_utilization_embedding: GPU memory for embedding model
            gpu_memory_utilization_evaluation: GPU memory for evaluation model
            max_model_len: Maximum context length (VLLM)
            max_num_batched_tokens: Maximum batched tokens (VLLM)
            enable_sleep_mode: Whether to enable sleep mode for VLLM models
        """
        if isinstance(backend, str):
            self.backend = Backend(backend.lower())
        else:
            self.backend = backend
        self.generation_model_name = generation_model
        self.evaluation_model_name = evaluation_model or generation_model
        self.embedding_model_name = embedding_model or generation_model
        self.api_key = api_key
        self.tensor_parallel_size = tensor_parallel_size

        gpu_mem_gen = (gpu_memory_utilization_generation
                       if gpu_memory_utilization_generation is not None else
                       gpu_memory_utilization)
        gpu_mem_emb = (gpu_memory_utilization_embedding
                       if gpu_memory_utilization_embedding is not None else
                       gpu_memory_utilization)
        gpu_mem_eval = (gpu_memory_utilization_evaluation
                        if gpu_memory_utilization_evaluation is not None else
                        gpu_memory_utilization)

        if self.generation_model_name == self.evaluation_model_name:
            shared_gen_eval_mem = max(gpu_mem_gen, gpu_mem_eval)
            self.gpu_memory_utilization_generation = shared_gen_eval_mem
            self.gpu_memory_utilization_evaluation = shared_gen_eval_mem
            logger.info(
                'Generation and evaluation models are identical (%s), '
                'using shared GPU memory allocation: %.2f',
                self.generation_model_name, shared_gen_eval_mem)
        else:
            self.gpu_memory_utilization_generation = gpu_mem_gen
            self.gpu_memory_utilization_evaluation = gpu_mem_eval

        if self.embedding_model_name == self.generation_model_name:
            self.gpu_memory_utilization_embedding = max(
                gpu_mem_emb, self.gpu_memory_utilization_generation)
            logger.info(
                'Embedding model is same as generation model (%s), '
                'using shared GPU memory allocation: %.2f',
                self.embedding_model_name,
                self.gpu_memory_utilization_embedding)
        elif self.embedding_model_name == self.evaluation_model_name:
            self.gpu_memory_utilization_embedding = max(
                gpu_mem_emb, self.gpu_memory_utilization_evaluation)
            logger.info(
                'Embedding model is same as evaluation model (%s), '
                'using shared GPU memory allocation: %.2f',
                self.embedding_model_name,
                self.gpu_memory_utilization_embedding)
        else:
            self.gpu_memory_utilization_embedding = gpu_mem_emb

        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_sleep_mode = (enable_sleep_mode if backend == 'vllm'
                                  and torch.cuda.is_available() else False)

        self._active_model: Optional[str] = None

        if backend == 'vllm':
            self._init_vllm_clients()
        elif backend == 'nim':
            self._init_nim_clients()
        else:
            raise ValueError(f'Invalid backend: {backend}')

    def _init_vllm_clients(self) -> None:
        """Initialize all three vLLM clients with sleep mode enabled."""
        self.generation_client = self._create_vllm_client(
            self.generation_model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization_generation,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_sleep_mode=self.enable_sleep_mode,
        )

        if self.evaluation_model_name == self.generation_model_name:
            self.evaluation_client = self.generation_client
        else:
            self.evaluation_client = self._create_vllm_client(
                self.evaluation_model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization_evaluation,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_batched_tokens,
                enable_sleep_mode=self.enable_sleep_mode,
            )

        if self.embedding_model_name == self.generation_model_name:
            self.embedding_client = self.generation_client
        elif self.embedding_model_name == self.evaluation_model_name:
            self.embedding_client = self.evaluation_client
        else:
            self.embedding_client = self._create_vllm_client(
                self.embedding_model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization_embedding,
                max_model_len=None,
                max_num_batched_tokens=None,
                enable_sleep_mode=self.enable_sleep_mode,
                task='embed',
            )

        if self.enable_sleep_mode:
            self._sleep_all()

    def update_generation_model(self, model: str) -> None:
        pass

    def _init_nim_clients(self) -> None:
        """Initialize NIM client (shared for all operations)."""
        self.nim_client = self._create_nim_client()

    def _create_vllm_client(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        enable_sleep_mode: bool = True,
        task: str = 'generate',
    ) -> 'LLM':  # noqa: F821
        """Create local vLLM client with OpenAI-compatible API.

        Args:
            model: Model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_model_len: Maximum context length
            max_num_batched_tokens: Maximum number of batched tokens
            enable_sleep_mode: Whether to enable sleep mode
            task: Task type ('generate' or 'embed')

        Returns:
            vLLM client with OpenAI-compatible interface
        """
        try:
            from vllm import LLM
        except ImportError as err:
            raise ImportError('vLLM is not installed. \
                    Please install it with: pip install vllm') from err

        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        os.environ.setdefault('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES',
                              '1')
        os.environ.setdefault('RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE', '1')

        logger.info('Initializing local vLLM with model: %s (task=%s)', model,
                    task)
        llm_kwargs = {
            'model': model,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'enable_sleep_mode': enable_sleep_mode,
            'disable_custom_all_reduce': True,
        }
        if max_model_len is not None:
            llm_kwargs['max_model_len'] = max_model_len
        if max_num_batched_tokens is not None:
            llm_kwargs['max_num_batched_tokens'] = max_num_batched_tokens


        vllm_model = LLM(**llm_kwargs, enforce_eager=True,
                         trust_remote_code=True)
        logger.info('Local vLLM initialized successfully: %s', model)
        return vllm_model

    def _create_nim_client(self) -> OpenAI:
        """Create NIM (NVIDIA API) client."""
        api_key = os.getenv('NVIDIA_API_KEY')

        if not api_key:
            raise ValueError('NVIDIA_API_KEY environment variable is not set')

        return OpenAI(base_url='https://integrate.api.nvidia.com/v1',
                      api_key=api_key)

    def _sleep_all(self) -> None:
        """Put all vLLM models to sleep."""
        if self.backend != Backend.VLLM or not self.enable_sleep_mode:
            return

        seen = set()
        for client in [
                self.generation_client,
                self.evaluation_client,
                self.embedding_client,
        ]:
            if id(client) not in seen:
                seen.add(id(client))
                client.sleep(level=1)
        self._active_model = None

    def _activate_model(self, model_type: str) -> None:
        """Activate a specific model, putting the previous one to sleep.

        Args:
            model_type: One of 'generation', 'evaluation', or 'embedding'
        """
        if self.backend != Backend.VLLM or not self.enable_sleep_mode:
            return

        if self._active_model == model_type:
            return  # Already active

        client_map = {
            'generation': self.generation_client,
            'evaluation': self.evaluation_client,
            'embedding': self.embedding_client,
        }

        new_client = client_map[model_type]

        if self._active_model is not None:
            old_client = client_map[self._active_model]
            if id(old_client) != id(new_client):
                old_client.sleep(level=1)

        new_client.wake_up()
        self._active_model = model_type

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        model_type: str,
        temperature: float = 0,
        max_tokens: int = 100000,
    ) -> str:
        """Internal chat method for generation and evaluation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_type: One of 'generation' or 'evaluation'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated/evaluated text string
        """
        self._activate_model(model_type)

        if model_type == 'generation':
            vllm_client = (self.generation_client
                           if self.backend == Backend.VLLM else None)
            model_name = self.generation_model_name
        else:  # evaluation
            vllm_client = (self.evaluation_client
                           if self.backend == Backend.VLLM else None)
            model_name = self.evaluation_model_name

        if self.backend == Backend.VLLM:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                seed=33,
            )
            outputs = vllm_client.chat(messages, sampling_params)
            return outputs[0].outputs[0].text
        else:
            completion = self.nim_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                stream=True,
                seed=33,
            )
            return ''.join(chunk.choices[0].delta.content
                           for chunk in completion
                           if chunk.choices[0].delta.content)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 100000,
    ) -> str:
        """Generate text using the generation model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        return self._chat(messages, 'generation', temperature, max_tokens)

    def evaluate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 100000,
    ) -> str:
        pass

    def embed(
        self,
        prompts: List[str],
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings using the embedding model.

        Args:
            prompts: List of text strings to embed
            input_type: Text input type

        Returns:
            List of embedding vectors (list of floats)
        """
        self._activate_model('embedding')

        if self.backend == Backend.VLLM:
            outputs = self.embedding_client.embed(prompts)
            return [output.outputs.embedding for output in outputs]
        else:
            embeddings = []
            for prompt in prompts:
                extra_body = {'truncate': 'NONE'}
                if input_type is not None:
                    extra_body['input_type'] = input_type
                response = self.nim_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=prompt,
                    encoding_format='float',
                    extra_body=extra_body,
                )
                embeddings.append(response.data[0].embedding)
            return embeddings

    def cleanup(self):
        """Explicitly cleanup vLLM engines and their resources."""
        if self.backend != Backend.VLLM:
            return

        logger.info("Cleaning up vLLM engines...")

        for attr_name in [
                'generation_client', 'evaluation_client', 'embedding_client'
        ]:
            client = getattr(self, attr_name, None)
            if client is not None:
                try:
                    delattr(self, attr_name)
                    del client
                except Exception as e:
                    logger.warning(f"Error cleaning up {attr_name}: {e}")

        import time
        time.sleep(1)

        logger.info("vLLM engine cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass  # Suppress errors in destructor


class QAGenerationState(TypedDict):

    task_id: str
    file_path: str
    input_dir: Optional[str]
    output_dir: str
    qa_pairs: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    negative_answers: List[List[str]]
    model_selection: str
    retry_count: int
    error_messages: List[str]
    status: str
    metrics: Dict[str, Any]
    feedback: Dict[str, Any]
    client: Optional[LLMClient]  # OpenAI client
    num_pairs: int  # Number of QA pairs to generate
    num_negatives: int  # Number of negative answers per question
    dedup_threshold: float  # Deduplication threshold
    parts: int
    evaluation: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]  # Metrics from LLM evaluation
    quality_threshold: float  # Minimum acceptable quality score
    hard: bool
    use_artifact: bool
    min_complexity: int
    query_type_distribution: str
    reasoning_type_distribution: Optional[str]
    min_hops: int
    max_hops: int
    models: Dict[str, str]
    segments: List[Dict[str, Any]]
    summary: List[Dict[str, Any]]
    top_artifacts: List[str]
    validated_pairs: List[Dict[str, Any]]
    high_quality_qa_pairs: List[Dict[str, Any]]
    not_evaluated_pairs: List[Dict[str, Any]]
    low_quality_qa_pairs: List[Dict[str, Any]]
    cross_part_contexts: List[Dict[str, Any]]
    self_contained_question: (
        bool  # Generate questions without referencing context/transcript
    )
    question_only: bool
    hard_negatives_top_k: int  # Number of hard negatives to mine per question
    hard_negatives_min_sim: (
        float  # Minimum similarity threshold (too dissimilar = not useful)
    )
    hard_negatives_max_sim: float  # Maximum similarity threshold
    hard_negatives_stats: Dict[str, Any]  # Statistics about hard negatives
    enable_hard_negatives: bool
    text_artifacts: List[Any]
    qa_data_to_write: List[Dict[str, Any]]
    model_progression: List[str]
    backend: str  # Backend to use (nim or vllm)


class ArtifactExtractor:
    def __init__(
        self,
        client: Optional['LLMClient'],
    ):
        self.client = client
        self.model = client.generation_model_name
        self.artifact_storage = {}  # In-memory storage for artifacts
        self.artifact_types = {
            'key_concepts': 'Important concepts, terms, or ideas mentioned',
            'relationships': 'Connections and relationships between concepts',
            'themes': 'Overarching themes and topics',
            'entities': 'People, organizations, locations, and specific items',
            'processes': 'Procedures, methods, or sequences described',
            'insights': 'Key insights, conclusions, or findings',
            'technical_terms': 'Domain-specific terminology and definitions',
            'contextual_factors': 'Background information and context',
        }

    def extract_artifacts(
            self, text: str,
            max_artifacts_per_type: int = 3) -> List[Dict[str, Any]]:
        pass

    def _create_extraction_prompt(self, text: str, max_artifacts: int) -> str:
        pass

    def _parse_artifacts_response(self,
                                  response_text: str) -> List[Dict[str, Any]]:
        pass

    def _extract_fallback_artifacts(self, text: str) -> List[Dict[str, Any]]:
        pass

    def _calculate_relevance_scores(self, artifacts: List[Dict[str, Any]],
                                    text: str) -> List[Dict[str, Any]]:
        pass

    def _rank_and_filter_artifacts(
            self, artifacts: List[Dict[str, Any]],
            min_score: float = 0.3) -> List[Dict[str, Any]]:
        pass

    def get_top_artifacts_for_qa(
        self,
        artifacts: List[Dict[str, Any]],
        top_k: int = 5,
        use_llm_ranking: bool = True,
    ) -> List[Dict[str, Any]]:
        pass

    def _rule_based_ranking(self, artifacts: List[Dict[str, Any]],
                            top_k: int) -> List[Dict[str, Any]]:
        pass

    def _llm_rank_artifacts_for_qa(self, artifacts: List[Dict[str, Any]],
                                   top_k: int) -> List[Dict[str, Any]]:
        pass

    def format_artifacts_for_prompt(self, artifacts: List[Dict[str,
                                                               Any]]) -> str:
        pass

    def save_artifacts_to_file(self, filepath: str):
        pass

    def load_artifacts_from_file(self, filepath: str):
        pass


def split_segments_text_structured(segments: List[dict],
                                   parts: int = 3) -> List[str]:
    pass


def format_timestamp(seconds: float) -> str:
    pass


def text_to_sentence_chunks(text: str,
                            sentences_per_chunk: int = 5) -> List[dict]:
    pass


def get_text_artifacts(segments, client):
    pass


def split_segments(segments: List[dict], parts: int = 3) -> List:
    pass


async def load_input(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def create_artifacts(state: QAGenerationState) -> Dict[str, Any]:
    pass


def _normalize_segment_metadata(
        segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pass


def extract_json_from_response(response_text: str) -> List[dict]:
    pass


def extract_timeline_from_response(response_text: str) -> Dict[str, Any]:
    pass


def parse_timestamp(ts: str) -> int:
    pass


def validate_timeline_events(
        events: List[Dict[str, Any]],
        segment_lookup: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pass


def generate_structured_timeline(
    segments: List[Dict[str, Any]],
    client: LLMClient,
    max_events: int = 18,
    max_segments: int = 400,
) -> List[Dict[str, Any]]:
    pass


def format_timeline_highlights(events: List[Dict[str, Any]],
                               limit: int = 8) -> List[str]:
    pass


_SEGMENT_LINE_PATTERN = re.compile(
    r'^Segment\s+(?P<id>\d+)\s*\((?P<start>\d{2}:\d{2}:\d{2})\s*-\s*'
    r'(?P<end>\d{2}:\d{2}:\d{2})\):\s*(?P<text>.*)$')


def _parse_structured_lines(
        raw: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    pass


def build_cross_part_contexts_from_segments(
        splits: List[str], max_sections: int = 3,
        max_chars: int = 1200) -> List[Dict[str, Any]]:
    pass


def build_cross_part_contexts_with_timeline(
    splits: List[str],
    timeline_events: List[Dict[str, Any]],
    highlight_limit: int = 8,
    max_external_events: int = 5,
) -> List[Dict[str, Any]]:
    pass


def transcription_segment_text(segments: List[dict]) -> str:
    pass


infer_video_style_prompt = """
Based on this transcript excerpt, describe the video's style.
Transcript:\n{{segment_text}}

Please ensure your output is ONLY the style keyword, and no other text.

STYLE:"""
infer_video_style_prompt_template = Template(infer_video_style_prompt)


def infer_video_style(segment_text: str, client: LLMClient) -> str:
    pass


def get_summary_block(summary: List[Dict[str, str]]) -> str:
    pass


def get_cross_part_context_block(cross_part_context: str) -> str:
    pass


def get_artifact_block_for_hard_question(artifacts_context):
    pass


def distribute_counts(total_count: int, distribution: Dict[str, float],
                      name: str = '') -> Dict[str, int]:
    pass


generate_hard_question_answer_with_timestamps_prompt = """
You are an expert at extracting complex question and answer \
    pairs from transcripts.

{{artifact_block}}

Guidelines:
1. Generate ONLY Complex Questions (Complexity 4-5):
   - All questions MUST require understanding connections between \
    different parts of the transcript
   - Questions should test deep understanding, not simple facts
{%- if self_contained_question %}
   - Do not mention the existence of a context/transcript in the \
    generated question like "in the transcript", "from the given context", \
        or "in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; if \
    missing, say you cannot generate a question.
{%- endif %}
   - Example: "How does the speaker's initial explanation of X relate to the \
    later implementation of Y?"

2. Question Types to Generate:
   - Multi-hop questions ({{query_counts.get("multi_hop", 0)}} questions): \
    Connect {{min_hops}}-{{max_hops}} separated segments
   - Structural questions ({{query_counts.get("structural", 0)}} questions): \
    Focus on relationships between concepts
   - Contextual questions ({{query_counts.get("contextual", 0)}} questions): \
    Require surrounding context to understand
   - Use the cross-part context snippets to connect evidence that lives \
    outside the current transcript section

{%- if reasoning_counts %}
3. Reasoning Types to Include:
   - Factual questions ({{reasoning_counts.get("factual", 0)}} \
    questions): Ask for complex facts that require synthesizing \
    multiple pieces of information (NOT simple lookups)
   - Relational questions ({{reasoning_counts.get("relational", 0)}} \
    questions): Ask how data points compare or correlate across \
    different segments
   - Inferential questions ({{reasoning_counts.get("inferential", 0)}} \
    questions): Ask about conclusions or implications requiring synthesis
   - Temporal questions ({{reasoning_counts.get("temporal", 0)}} \
    questions): Ask about changes or events over time across segments
   - Procedural questions ({{reasoning_counts.get("procedural", 0)}} \
    questions): Ask about complex multi-step processes or guidelines
   - Visual questions ({{reasoning_counts.get("visual", 0)}} \
    questions): Ask about visual details requiring cross-reference
   - Causal questions ({{reasoning_counts.get("causal", 0)}} \
    questions): Ask about cause-effect chains spanning segments

   Example COMPLEX questions by reasoning type (all must be complexity 4-5):
   - Factual: "What is the total combined budget allocation across \
    all departmental initiatives mentioned, and how does it relate to the \
    overall fiscal year target?"
   - Relational: "How does the performance metric achieved in Q2 compare to \
    both the initial baseline and the revised targets that were set?"
   - Inferential: "Based on the challenges outlined and the proposed \
    solutions, what unstated assumptions underlie the strategic pivot?"
   - Temporal: "How did the implementation timeline evolve from the \
    initial proposal through the mid-year review to the final execution phase?"
   - Procedural: "What is the complete approval workflow including standard \
    requirements, exceptions, and escalation processes?"
   - Visual: "How do the visual elements presented relate to the verbal \
    descriptions provided, and what discrepancies exist between them?"
   - Causal: "What chain of events, starting from the \
    initial decision, led through various complications to the final outcome?"

4. IMPORTANT - Orthogonal Distributions:
   - Each question must have BOTH a query type \
    (multi_hop/structural/contextual) AND a reasoning type
   - For example: A multi_hop factual question \
    or a structural causal question
   - Ensure the final distribution matches both specified percentages
{%- endif %}

Meta-Reasoning Playbook (follow before drafting output):
   1. Scan the current section for key statements and log their timestamps.
   2. Compare against the cross-part context to spot related events elsewhere \
    and capture the matching timestamps.
   3. Design a reasoning chain that depends on evidence from both the local \
    section and external context, outlining each hop.
   4. Confirm each hop's start/end time aligns with the chosen segments, then \
    compose the question and answer around that chain.
   5. Ensure the final answer references all hops and explains why the \
    timestamps substantiate the reasoning.

5. **IMPORTANT - Segment Identification**:
   - The transcript below contains segments formatted as "Segment N \
    (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - For each question-answer pair you generate, identify ALL segment numbers \
    FROM which the question is derived
   - These segments are the source material that should be retrieved when \
    someone asks this question
   - Record these segment numbers in the "segment_ids" field as a list of \
    integers (e.g., [1, 4, 8])
   - For multi-hop questions:
     * The top-level "segment_ids" should be the UNION of all segment IDs \
      across all hops
     * Each hop in "hop_contexts" should specify its own "segment_ids" list
     * Example: If hop 1 uses [1, 3] and hop 2 uses [6, 8], then top-level \
      segment_ids should be [1, 3, 6, 8]
   - The timestamps will be automatically calculated from the segments you \
    identify


4. For Each Question:
   - Must have complexity level {{min_complexity}} or higher
   - Generate the question FROM the identified segments (these segments are \
    the source material)
   - Multi-hop questions must specify hop_count ({{min_hops}}-{{max_hops}})
   - Provide hop_contexts: a list where each hop includes "hop", \
    "segment_ids" (the source segments for this hop), "start_time", \
    "end_time", and a concise "context" summary \
    describing the supporting segment(s)

5. Generate {{num_pairs}} distinct question and answer pairs.

The output should be a JSON array of objects (with the number of objects \
    equal to {{num_pairs}}), where each object contains:
  - "question": the complex question requiring structural understanding of \
    the contexts/transcripts/segments without explicitly referencing the \
    context/transcript/segments in the question
  - "answer": comprehensive answer from the contexts/transcripts/segments \
    without explicitly referencing the context/transcript/segments in the \
    answer
  - "question_complexity": numeric score {{min_complexity}}-5
  - "query_type": one of ["multi_hop", "structural", "contextual"]
{%- if reasoning_counts %}
  - "reasoning_type": one of ["factual", "relational", "inferential", \
    "temporal", "procedural", "visual", "causal"]
{%- endif %}
  - "segment_ids": list of segment numbers (e.g., [1, 4, 8]) that are the \
    source material for this question (these should be retrieved when the \
    question is asked)
  - "start_time": HH:MM:SS timestamp (will be calculated from earliest \
    segment)
  - "end_time": HH:MM:SS timestamp (will be calculated \
    from latest segment)
  - "hop_count": number of hops ({{min_hops}}-{{max_hops}}) \
    for multi_hop questions only
  - "hop_contexts": array of hop detail objects with "hop", \
    "segment_ids" (source segments for this hop), "start_time", \
        "end_time", "context" (required for every question)

Additional Constraints:
- **Output only the JSON. Do not include any \
    introductory or concluding text.**
- **Please ensure that your entire response is \
    strictly valid JSON with no additional text or formatting.**
- **The output should be a JSON array containing \
    exactly {{num_pairs}} objects.**
- **Pay careful attention to the timestamp markers \
    in the transcript to ensure accurate start_time and end_time values.**

Video Style Guidance: {{video_style}}

{{summary_block}}

{{cross_part_context_block}}

Transcript:
---
{{segment_text}}

Output a JSON array where each element is an object with the required keys.
Please ensure your output is ONLY the JSON object \
    with no preamble or additional text.
"""

generate_hard_question_answer_with_timestamps_template = Template(
    generate_hard_question_answer_with_timestamps_prompt)


def _normalize_time_value(value: Any, fallback: str = '') -> str:
    pass


def normalize_hop_contexts(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    pass


def _range_contains(start: int, end: int, candidate: Dict[str, Any]) -> bool:
    pass


def _find_best_matching_range(
        start: int, candidates: List[Dict[str,
                                          Any]]) -> Optional[Dict[str, Any]]:
    pass


def validate_hop_context_time_ranges(
    pair: Dict[str, Any],
    local_ranges: List[Dict[str, Any]],
    cross_ranges: List[Dict[str, Any]],
) -> None:
    pass


def generate_hard_question_answer_with_timestamps(
    segment_text: str,
    video_style: str,
    num_pairs: int,
    client: LLMClient,
    summary: List[Dict[str, str]] = None,
    query_type_distribution: Dict[str, float] = None,
    reasoning_type_distribution: Dict[str, float] = None,
    min_complexity: int = 4,
    min_hops: int = 2,
    max_hops: int = 4,
    top_artifacts: List[Dict[str, Any]] = None,
    artifacts_context: str = None,
    cross_part_context: str = None,
    allowed_ranges_local: Optional[List[Dict[str, Any]]] = None,
    allowed_ranges_cross: Optional[List[Dict[str, Any]]] = None,
    self_contained_question: bool = False,
) -> List[dict]:
    pass


def get_artifact_block(artifacts_context):
    pass


generate_question_answer_with_timestamps_prompt = """
You are an expert at extracting question and answer pairs from transcripts.

{{artifact_block}}

Guidelines:
1. Frame Questions as If You Don't Know the Answer:
   - Write questions that are general and applicable to a wide range of \
    scenarios.
   - Do not directly reference the transcript content or assume the viewer \
    already knows the transcript details.
{%- if self_contained_question %}
   - Do not mention the existence of a context/transcript in the generated \
    question like "in the transcript", "from the given context", or \
        "in Segment 148". Produce a natural, standalone question.
   - Only use facts present in the provided context/transcript; \
    if missing, say you cannot generate a question.
{%- endif %}
   - Example: "What are some emerging trends in the field of \
    artificial intelligence?"

2. Ensure Questions are Contextually Relevant:
   - Questions should be about broad topics that could be answered by a wide \
    range of transcripts.

3. Diversity in Question Types:
   - Include a mix of factual questions (e.g., "What are the primary benefits \
    of cloud computing?"), why/how questions (e.g., "Why has AI adoption \
    grown in recent years?"), and procedural questions (e.g., "How did the\
    IT manager explain the process for requesting new software?").

{%- if reasoning_counts %}
4. Reasoning Types to Include:
   - Factual questions ({{reasoning_counts.get("factual", 0)}} \
    questions): Ask for specific, verifiable facts unique to this document
   - Relational questions ({{reasoning_counts.get("relational", 0)}} \
    questions): Ask how data points compare or correlate
   - Inferential questions ({{reasoning_counts.get("inferential", 0)}} \
    questions): Ask about conclusions or implications
   - Temporal questions ({{reasoning_counts.get("temporal", 0)}} \
    questions): Ask about changes or events over time
   - Procedural questions ({{reasoning_counts.get("procedural", 0)}} \
    questions): Ask about steps or guidelines
   - Visual questions ({{reasoning_counts.get("visual", 0)}} \
    questions): Ask about unique visual details
   - Causal questions ({{reasoning_counts.get("causal", 0)}} \
    questions): Ask about causes, reasons, or contributing factors

   Example questions by reasoning type:
   - Factual: "How many paid time off days are employees \
    eligible for per year?"
   - Relational: "How did sales revenue compare to expenses in Q3 2023?"
   - Inferential: "What does the report suggest about future AI investment \
    trends?"
   - Temporal: "What was the change in machinery value between 2018 and 2019?"
   - Procedural: "What are the steps for submitting a travel reimbursement?"
   - Visual: "Which report includes the Department of Agriculture logo?"
   - Causal: "Why did patient recovery times improve after the new protocol?"
{%- endif %}

5. Generate {{num_pairs}} distinct question and answer pairs:
   - Each pair must be unique and have varying complexity levels between 1 \
    (simple) and 5 (complex).
   - For each pair, identify the start and end time (in HH:MM:SS format) \
    that best supports the answer.

6. **IMPORTANT - Segment Identification**:
   - The transcript below contains segments formatted as "Segment N \
    (HH:MM:SS - HH:MM:SS): text" where N starts from 1
   - For each question-answer pair you generate, identify ALL segment numbers \
    FROM which the question is derived
   - These segments contain the source material that inspired the \
    question and provide the answer
   - Record these segment numbers in the "segment_ids" field as a list of \
    integers (e.g., [1, 3, 6])
   - Think of it as: "If someone asks this question in a retrieval system, \
    these are the segments that should be retrieved"
   - The timestamps (start_time and end_time) will be automatically \
    calculatedfrom the segments you identify
   - If a question requires information from multiple segments, include all \
    relevant segment numbers

The output should be a JSON array of objects (with the number of \
    objects equal to {{num_pairs}}), where each object contains:
  - "question": the generated question without explicitly referencing the \
    context/transcript/segments in the question.
  - "answer": the answer without explicitly referencing the \
    context/transcript/segments in the answer.
  - "question_complexity": a numeric score from 1 (simple) to 5 (complex).
  - "segment_ids": list of segment numbers (e.g., [1, 3, 6]) that are the \
    source material for this question-answer pair.
  - "start_time": the HH:MM:SS timestamp (will be calculated from earliest \
    segment).
  - "end_time": the HH:MM:SS timestamp (will be calculated from latest \
    segment).

Additional Constraints:
- **Output only the JSON. Do not include any introductory or concluding text.**
- **Please ensure that your entire response is strictly valid JSON with no \
    additional text or formatting.**
- **The output should be a JSON array containing exactly {{num_pairs}} \
    objects.**
- **Pay careful attention to the timestamp markers in the transcript to \
    ensure accurate start_time and end_time values.**

Video Style Guidance: {{video_style}}

{{summary_block}}

Transcript with Timestamps:
---
{{segment_text}}

Output a JSON array where each element is an object with the following keys:
  - "question"
  - "answer"
  - "question_complexity"
  - "segment_ids"
  - "start_time"
  - "end_time"

Please ensure your output is ONLY the JSON object with no preamble \
    or additional text.
"""
generate_question_answer_with_timestamps_template = Template(
    generate_question_answer_with_timestamps_prompt)


def generate_question_answer_with_timestamps(
    segment_text: str,
    video_style: str,
    num_pairs: int,
    client: LLMClient,
    summary: List[Dict[str, str]] = None,
    reasoning_type_distribution: Dict[str, float] = None,
    top_artifacts: List[Dict[str, Any]] = None,
    artifacts_context: str = None,
    self_contained_question: bool = False,
) -> List[dict]:
    pass


def extract_time_ranges_from_structured_text(
        structured_text: str) -> List[Dict[str, Any]]:
    pass


def process_segments_info(qa_pair: Dict[str, Any],
                          segments: List[Dict[str, Any]]) -> None:
    pass


async def generate_qa_pairs(state: QAGenerationState) -> Dict[str, Any]:
    pass


def get_embeddings(
    texts: List[str],
    client: LLMClient,
    input_type: Optional[str] = 'passage',
    batch_size: int = 20,
):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_truncated = [
            text[:1600] if len(text) > 1600 else text for text in batch
        ]
        resp = client.embed(
            prompts=batch_truncated,
            input_type=input_type,
        )
        embs.extend(resp)
    return embs


def save_index_and_metadata(embs, chunks, prefix, out_dir):
    pass


def validate_answer_spans_hybrid(
    qa_pairs: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    base_name: str,
    output_dir: str,
    client: LLMClient,
    sim_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    pass


def deduplicate_qa(
    qa_pairs: List[Dict[str, Any]],
    client: LLMClient,
    threshold: float = 0.9,
) -> List[Dict[str, Any]]:
    pass


def convert_posixpath_to_string(data):
    pass


async def validate_qa_pairs(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def mine_hard_negative_segments(
        state: QAGenerationState) -> Dict[str, Any]:
    pass


generate_negative_answers_prompt = """
Given the following question from a transcript, \
    generate {{num_negatives}} plausible but incorrect answers.

QUESTION: {{question}}
CORRECT ANSWER: {{correct_answer}}

Guidelines:
1. Create answers that are factually incorrect but sound reasonable
2. Make answers similar in structure and length to the correct answer
3. Ensure negative answers are clearly wrong when \
    compared to the correct answer
4. Vary the types of incorrectness (wrong facts, wrong reasoning, \
    wrong conclusions)

Output as a JSON array of strings containing exactly \
    {{num_negatives}} negative answers.
Example: ["negative answer 1", "negative answer 2", "negative answer 3"]

Do not include any other text, only the JSON array.
"""
generate_negative_answers_prompt_template = Template(
    generate_negative_answers_prompt)


def generate_negative_answers(
    question: str,
    correct_answer: str,
    segment_text: str,
    num_negatives: int,
    client: LLMClient,
) -> List[str]:
    pass


async def generate_negative_answers_node(
        state: QAGenerationState) -> Dict[str, Any]:
    pass


llm_evaluate_qa_pair_prompt = """
You are an expert evaluator of question-answer pairs.

Please evaluate the following question and answer \
based on the provided context:

CONTEXT:
```
{{context}}
```

{{cross_part_context_block}}

QUESTION: {{question}}

ANSWER: {{answer}}

Evaluate this QA pair on the following criteria (score 1-10):
1. Relevance: Is the question relevant to the main topics in the context?
2. Accuracy: Is the answer factually correct according to the context?
3. Context Support: Is the answer well-supported by the context?
4. Clarity: Is the question clear and unambiguous?

For each criterion, provide:
- Score (1-10)
- Brief justification

Then provide an overall score (1-10) and final assessment.

Output your evaluation as a valid JSON object with the following structure:
```json
{
  "relevance": { "score": X, "justification": "..." },
  "accuracy": { "score": X, "justification": "..." },
  "context_support": { "score": X, "justification": "..." },
  "clarity": { "score": X, "justification": "..." },
  "overall": { "score": X, "assessment": "..." },
  "improvements": "Suggestions for improving this QA pair"
}

Please ensure your output is ONLY the JSON \
object with no preamble or additional text.
"""
llm_evaluate_qa_pair_prompt_template = Template(llm_evaluate_qa_pair_prompt)


async def llm_evaluate_qa_pair(
    question: str,
    answer: str,
    context: str,
    cross_part_context_block: str,
    client: LLMClient,
) -> Dict[str, Any]:
    pass


async def evaluate_qa_pairs(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def filter_by_quality(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def monitor_progress(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def store_results(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def feedback_controller(state: QAGenerationState) -> Dict[str, Any]:
    pass


async def end(state: QAGenerationState) -> Dict[str, Any]:
    pass


def compute_metrics(
    records: List[dict],
    query_type_distribution: Optional[str] = None,
    reasoning_type_distribution: Optional[str] = None,
) -> Dict[str, Any]:
    pass


def route_after_validation(state: QAGenerationState) -> str:
    pass


def should_retry(state: QAGenerationState) -> str:
    pass
