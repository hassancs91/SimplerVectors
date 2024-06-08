from pydantic import BaseModel, Field
from typing import List
import re

import openai
import numpy as np

# from sklearn.metrics.pairwise import cosine_similarity


class ChunkInfo(BaseModel):
    text: str
    num_characters: int = Field(description="Number of characters in the chunk")
    num_words: int = Field(description="Number of words in the chunk")


class TextChunks(BaseModel):
    num_chunks: int = Field(description="Total number of chunks")
    chunks: List[ChunkInfo]


def chunk_by_max_chunk_size(
    text: str, max_chunk_size: int, preserve_sentence_structure: bool = False
) -> TextChunks:
    """
    Split the given text into chunks based on the maximum chunk size trying to reserve sentence endings if preserve_sentence_structure is enabled

    Parameters:
    - text (str): The input text to be split into chunks.
    - max_chunk_size (int): The maximum size of each chunk.
    - preserve_sentence_structure: Whether to consider preserving the sentence structure when splitting the text.

    Returns:
    - TextChunks: An object containing the total number of chunks and a list of ChunkInfo objects.
    - num_chunks (int): The total number of chunks.
    - chunks (List[ChunkInfo]): A list of ChunkInfo objects, each representing a chunk of the text.
        - chunk (str): The chunk of text.
        - num_characters (int): The number of characters in the chunk.
        - num_words (int): The number of words in the chunk.
    """

    if preserve_sentence_structure:
        sentences = re.split(r"(?<=[.!?]) +", text)
    else:
        sentences = [
            text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)
        ]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if preserve_sentence_structure:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # If the current chunk is empty and the sentence is longer than the max size,
                    # accept this sentence as a single chunk even if it exceeds the max size.
                    chunks.append(sentence)
                    sentence = ""
        else:
            chunks.append(sentence)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    chunk_infos = [
        ChunkInfo(text=chunk, num_characters=len(chunk), num_words=len(chunk.split()))
        for chunk in chunks
    ]

    return TextChunks(num_chunks=len(chunk_infos), chunks=chunk_infos)