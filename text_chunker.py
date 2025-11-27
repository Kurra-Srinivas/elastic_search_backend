"""Text chunking utility for splitting documents into smaller pieces"""
from typing import List
import logging

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 50  # characters overlap between chunks


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a word boundary
        if end < text_length:
            # Look for a good break point (space, newline, or punctuation)
            break_chars = ['\n\n', '\n', '. ', '! ', '? ', ' ', '']
            best_break = end
            
            for break_char in break_chars:
                if break_char:
                    break_pos = text.rfind(break_char, start, end)
                    if break_pos != -1:
                        best_break = break_pos + len(break_char)
                        break
            
            end = best_break
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= text_length:
            break
    
    return chunks


def chunk_text_by_sentences(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE, overlap_sentences: int = 1) -> List[str]:
    """
    Split text into chunks by sentences (better for semantic meaning)
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum size of each chunk (in characters)
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chunk_size:
        return [text] if text else []
    
    # Split by sentence endings
    import re
    sentences = re.split(r'([.!?]\s+)', text)
    
    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in combined_sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed max size, save current chunk
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk).strip())
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return chunks if chunks else [text]

