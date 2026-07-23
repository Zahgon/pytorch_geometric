import os
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp

CLIENT_INITD = False

CLIENT = None
GLOBAL_NIM_KEY = ""
SYSTEM_PROMPT = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Separate each with a new line. Do not output anything else. Try to focus on key triples that form a connected graph."  # noqa
MAX_OUTER_RETRIES = 5  # Maximum number of times the entire multiprocessing job is retried. # noqa
RETRY_DELAY = 5  # Fixed sleep time (in seconds) between outer retries.
MAX_NIM_RETRIES = 200  # Maximum number of attempts to call the NIM API inside one worker.  # noqa
BASE_DELAY = 0.5  # Initial wait time before retrying a failed network call.


class TXT2KG():
    def __init__(
        self,
        NVIDIA_NIM_MODEL: Optional[
            str] = "nvidia/llama-3.1-nemotron-70b-instruct",
        NVIDIA_API_KEY: Optional[str] = "",
        ENDPOINT_URL: Optional[str] = "https://integrate.api.nvidia.com/v1",
        local_LM: bool = False,
        chunk_size: int = 512,
    ) -> None:
        self.local_LM = local_LM
        if self.local_LM:
            self.initd_LM = False
        else:
            self.NVIDIA_API_KEY = NVIDIA_API_KEY
            self.NIM_MODEL = NVIDIA_NIM_MODEL
            self.ENDPOINT_URL = ENDPOINT_URL

        self.chunk_size = chunk_size

        self.doc_id_counter = 0
        self.relevant_triples = {}
        self.total_chars_parsed = 0
        self.time_to_parse = 0.0

    def save_kg(self, path: str) -> None:
        pass

    def _chunk_to_triples_str_local(self, txt: str) -> str:
        pass

    def add_doc_2_KG(
        self,
        txt: str,
        QA_pair: Optional[Tuple[str, str]] = None,
    ) -> None:
        pass

    def _extract_relevant_triples(
        self,
        txt: str,
        max_retries: int = MAX_OUTER_RETRIES,
        retry_delay: float = RETRY_DELAY,
    ) -> List[Tuple[str, str, str]]:
        pass


known_reasoners = [
    "llama-3.1-nemotron-ultra-253b-v1",
    "kimi-k2-instruct",
    "nemotron-super-49b-v1_5",
    "gpt-oss",
]


def _chunk_to_triples_str_cloud(
        txt: str, GLOBAL_NIM_KEY='',
        NIM_MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        ENDPOINT_URL="https://integrate.api.nvidia.com/v1",
        post_text=SYSTEM_PROMPT) -> str:
    pass


def _parse_n_check_triples(triples_str: str) -> List[Tuple[str, str, str]]:
    processed = []
    split_by_newline = triples_str.split("\n")
    if len(split_by_newline) > 1:
        split_triples = split_by_newline
        llm_obeyed = True
    else:
        split_triples = triples_str[1:-1].split(") (")
        llm_obeyed = False
    for triple_str in split_triples:
        try:
            if llm_obeyed:
                triple_str = triple_str.replace("(", "").replace(")",
                                                                 "").replace(
                                                                     "'", "")
            split_trip = triple_str.split(',')
            split_trip = [(i[1:] if i[0] == " " else i) for i in split_trip]
            split_trip = [(i[:-1].lower() if i[-1] == " " else i)
                          for i in split_trip]
            potential_trip = tuple(split_trip)
        except:  # noqa
            continue
        if 'tuple' in str(type(potential_trip)) and len(
                potential_trip
        ) == 3 and "note:" not in potential_trip[0].lower():
            if potential_trip[0] != '' and potential_trip[
                    1] != '' and potential_trip[2] != '':
                processed.append(potential_trip)
    return processed


def _llm_then_python_parse(chunks, py_fn, llm_fn, **kwargs):
    relevant_triples = []
    for chunk in chunks:
        relevant_triples += py_fn(llm_fn(chunk, **kwargs))
    return relevant_triples


def _multiproc_helper(
    rank,
    chunks_for_rank,
    py_fn,
    llm_fn,
    NIM_KEY,
    NIM_MODEL,
    ENDPOINT_URL,
    max_retries=MAX_NIM_RETRIES,
    base_delay=BASE_DELAY,
):

    for attempt in range(max_retries):
        try:
            return _llm_then_python_parse(
                chunks_for_rank,
                py_fn,
                llm_fn,
                GLOBAL_NIM_KEY=NIM_KEY,
                NIM_MODEL=NIM_MODEL,
                ENDPOINT_URL=ENDPOINT_URL,
            )

        except Exception:
            if attempt == max_retries - 1:
                raise

            from random import uniform
            sleep_time = base_delay * (2**min(attempt, 6))
            sleep_time += uniform(0, 0.1)
            time.sleep(sleep_time)


def _get_num_procs():
    pass


def _chunk_text(text: str, chunk_size: int = 512) -> list[str]:
    """Function to chunk text into sentence-based segments.
    Co-authored with Claude AI.
    """
    if not text:
        return []

    sentence_endings = '.!?'

    chunks = []

    while text:
        if len(text) <= chunk_size:
            chunks.append(text.strip())
            break

        chunk = text[:chunk_size]

        best_split = chunk_size
        for ending in sentence_endings:
            last_ending = chunk.rfind(ending)
            if last_ending != -1:
                best_split = min(
                    best_split, last_ending + 1 +
                    (1 if last_ending + 1 < len(chunk)
                     and chunk[last_ending + 1].isspace() else 0))

        if best_split < len(text) and text[best_split].isalpha():
            space_split = text[:best_split].rfind(' ')
            if space_split != -1:
                best_split = space_split

        chunks.append(text[:best_split].strip())

        text = text[best_split:].lstrip()

    return chunks


Triple = Union[List[str], Tuple[str, ...]]


def _merge_triples_deterministically(
        triples: List[List[Triple]]) -> List[Tuple[str, ...]]:
    """Flatten a list of lists of triples and return a deterministic,
    reproducible sorted list of tuples.

    Args:
        triples (List[List[Triple]]): A list of lists of triples, where each
            triple is a list or tuple of strings or other comparable values.
            Typically, each inner list comes from a worker.

    Returns:
        List[Tuple[str, ...]]: A flattened list of triples as tuples, sorted
            deterministically. Sorting is Unicode-safe and reproducible across
            Python versions using `str.casefold()`. Tuples are immutable to
            ensure hashability and stability in dicts/sets.
    """
    flat_triples = [tuple(t) for sublist in triples for t in sublist]

    flat_triples.sort(key=lambda triple: tuple(
        s.casefold() if isinstance(s, str) else s for s in triple))

    return flat_triples
