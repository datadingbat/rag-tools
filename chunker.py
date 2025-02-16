# chunker.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import re
import unicodedata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    RECURSIVE = 'recursive'
    ELASTIC_NATIVE = 'elastic_native'

class ChunkingMode(Enum):
    FAST = 'fast'
    THOROUGH = 'thorough'

@dataclass
class ElasticChunkingConfig:
    """Configuration for Elastic's native inference chunking"""
    strategy: str = "sentence"  # 'sentence' or 'word'
    max_chunk_size: int = 250   # 20-300 for sentence, 10-300 for word
    sentence_overlap: int = 1    # 0 or 1, only for sentence strategy
    overlap: Optional[int] = None  # For word strategy, up to half of max_chunk_size

    def to_dict(self) -> Dict:
        """Convert config to Elasticsearch format"""
        config = {
            "strategy": self.strategy,
            "max_chunk_size": self.max_chunk_size
        }
        if self.strategy == "sentence" and self.sentence_overlap is not None:
            config["sentence_overlap"] = self.sentence_overlap
        elif self.strategy == "word" and self.overlap is not None:
            config["overlap"] = self.overlap
        return config
    
@dataclass
class RecursiveChunkingConfig:
    """Configuration for recursive chunking"""
    mode: ChunkingMode = ChunkingMode.THOROUGH
    model_limit: int = 1536
    long_word_limit: int = 45
    min_chunk_size: int = 100
    retry_attempts: int = 3
    custom_separators: List[str] = None
    keep_html_tags: bool = True

@dataclass
class ChunkingConfig:
    """Main chunking configuration"""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    recursive_config: Optional[RecursiveChunkingConfig] = None
    elastic_config: Optional[ElasticChunkingConfig] = None

class EnhancedRecursiveChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._compile_patterns()
        if self.config.mode == ChunkingMode.THOROUGH:
            self._chunk_cache = {}
    
    def _compile_patterns(self):
        """Compile patterns based on mode"""
        if self.config.mode == ChunkingMode.FAST:
            # Simple patterns for fast mode
            self.format_pattern = re.compile(r'\n\n')
            self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        else:
            # Complex patterns for thorough mode
            html_pattern = r'(<p>.*?</p>)|(\n\n)'
            self.format_pattern = re.compile(html_pattern, re.DOTALL) if self.config.keep_html_tags else re.compile(r'\n\n')
            self.sentence_pattern = re.compile(
                r'(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!Jr)(?<!Sr)(?<!Prof)'
                r'(?<!\w\.\w)(?<=[.!?])\s+(?=[A-Z])'
            )
    
    def clean_text(self, text: str) -> str:
        """Clean text based on mode"""
        if not text:
            return ""
            
        if self.config.mode == ChunkingMode.FAST:
            # Simple cleaning for fast mode
            return ' '.join(text.split())
        else:
            # Thorough cleaning
            if self.config.keep_html_tags:
                # Preserve HTML tags
                tag_map = {}
                for i, tag in enumerate(re.finditer(r'<[^>]+>', text)):
                    placeholder = f"__TAG_{i}__"
                    tag_map[placeholder] = tag.group()
                    text = text.replace(tag.group(), placeholder)
            
            # Normalize and clean
            text = unicodedata.normalize('NFKC', text)
            text = ''.join(char if unicodedata.category(char)[0] != 'C' or char in '\n\t' 
                          else ' ' for char in text)
            text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
            
            if self.config.keep_html_tags:
                for placeholder, tag in tag_map.items():
                    text = text.replace(placeholder, tag)
            
            return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """Main chunking method that delegates to appropriate mode"""
        if self.config.mode == ChunkingMode.FAST:
            return self._chunk_text_fast(text)
        else:
            return self._chunk_text_thorough(text)
    
    def _chunk_text_fast(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """Fast mode chunking implementation"""
        if len(text) <= self.config.model_limit:
            return [{"text": text, "chunknumber": 0}]
        
        chunks = []
        chunk_number = 0
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in self.format_pattern.split(text) if p.strip()]
        
        current_chunk = ""
        for para in paragraphs:
            if len(para) <= self.config.model_limit:
                if current_chunk and len(current_chunk) + len(para) + 2 <= self.config.model_limit:
                    current_chunk = f"{current_chunk}\n\n{para}"
                else:
                    if current_chunk:
                        chunks.append({"text": current_chunk, "chunknumber": chunk_number})
                        chunk_number += 1
                    current_chunk = para
            else:
                # Process large paragraph
                if current_chunk:
                    chunks.append({"text": current_chunk, "chunknumber": chunk_number})
                    chunk_number += 1
                    current_chunk = ""
                
                # Simple sentence splitting
                sentences = self.sentence_pattern.split(para)
                current_sentence = ""
                
                for sentence in sentences:
                    if len(sentence) <= self.config.model_limit:
                        if current_sentence and len(current_sentence) + len(sentence) + 1 <= self.config.model_limit:
                            current_sentence = f"{current_sentence} {sentence}"
                        else:
                            if current_sentence:
                                chunks.append({"text": current_sentence, "chunknumber": chunk_number})
                                chunk_number += 1
                            current_sentence = sentence
                    else:
                        # Handle very long sentence with simple splitting
                        if current_sentence:
                            chunks.append({"text": current_sentence, "chunknumber": chunk_number})
                            chunk_number += 1
                        
                        # Simple word boundary splitting
                        while sentence:
                            split_point = self.config.model_limit
                            if len(sentence) > split_point:
                                space_pos = sentence.rfind(' ', 0, split_point)
                                if space_pos > 0:
                                    split_point = space_pos
                            
                            chunks.append({"text": sentence[:split_point].strip(), "chunknumber": chunk_number})
                            chunk_number += 1
                            sentence = sentence[split_point:].strip()
                
                if current_sentence:
                    chunks.append({"text": current_sentence, "chunknumber": chunk_number})
                    chunk_number += 1
        
        if current_chunk:
            chunks.append({"text": current_chunk, "chunknumber": chunk_number})
        
        return chunks
    
    def _chunk_text_thorough(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Thorough mode chunking implementation:
        - Splits text by paragraphs (double newlines).
        - Combines paragraphs until adding another would exceed the model limit.
        - If a single paragraph is too long, splits it into sentences (using punctuation).
        - Combines sentences until adding one would exceed the limit.
        - If a sentence is too long, finds a nearby space to split it, or splits at the limit.
        Returns a list of chunks with their respective chunk numbers.
        """
        model_limit = self.config.model_limit

        # Base case: if the entire text fits within the limit, return it as one chunk.
        if len(text) <= model_limit:
            return [{"text": text, "chunknumber": 0}]
        
        chunks = []
        chunk_number = 0
        current_chunk = ""

        # Split text into paragraphs based on double newlines.
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            # Attempt to combine paragraphs.
            tentative_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            if len(tentative_chunk) <= model_limit:
                current_chunk = tentative_chunk
            else:
                # If the current chunk is non-empty, save it.
                if current_chunk:
                    chunks.append({"text": current_chunk, "chunknumber": chunk_number})
                    chunk_number += 1
                    current_chunk = ""
                
                # Now, if the paragraph itself is too long, split it into sentences.
                if len(para) > model_limit:
                    # Split by punctuation (period, question mark, exclamation mark)
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_sent_chunk = ""
                    for sentence in sentences:
                        tentative_sent = f"{current_sent_chunk} {sentence}".strip() if current_sent_chunk else sentence
                        if len(tentative_sent) <= model_limit:
                            current_sent_chunk = tentative_sent
                        else:
                            # If current_sent_chunk is not empty, finalize it.
                            if current_sent_chunk:
                                chunks.append({"text": current_sent_chunk, "chunknumber": chunk_number})
                                chunk_number += 1
                                current_sent_chunk = sentence
                            else:
                                # Sentence itself is too long. Look for a space within the limit.
                                break_point = model_limit
                                space_pos = sentence.rfind(" ", 0, break_point)
                                if space_pos == -1:
                                    # No space found; break at the limit.
                                    space_pos = break_point
                                # Append the chunk up to the break point.
                                chunks.append({"text": sentence[:space_pos].strip(), "chunknumber": chunk_number})
                                chunk_number += 1
                                # Set the remainder as the new current_sent_chunk.
                                current_sent_chunk = sentence[space_pos:].strip()
                    # Append any remaining sentences.
                    if current_sent_chunk:
                        chunks.append({"text": current_sent_chunk, "chunknumber": chunk_number})
                        chunk_number += 1
                else:
                    # The paragraph fits on its own but couldn't combine with the previous chunk.
                    current_chunk = para

        # Append any remaining text as the last chunk.
        if current_chunk:
            chunks.append({"text": current_chunk, "chunknumber": chunk_number})
        
        return chunks


def get_chunking_strategy() -> ChunkingConfig:
    """Get user preferences for chunking strategy."""
    print("\nChunking Strategy Selection:")
    print("1. Recursive Chunking - Custom implementation with fine-grained control")
    print("2. Elastic Native Chunking - Built-in Elasticsearch inference chunking")
    
    while True:
        choice = input("\nSelect chunking strategy (1 or 2) [default=1]: ").strip()
        if not choice or choice == "1":
            return _configure_recursive_chunking()
        elif choice == "2":
            return _configure_elastic_chunking()
        else:
            print("Invalid choice. Please select 1 or 2.")


def _configure_recursive_chunking() -> ChunkingConfig:
    """Configure recursive chunking settings interactively."""
    print("\nRecursive Chunking Mode:")
    print("1. Fast Mode - Faster processing, basic text handling")
    print("2. Thorough Mode - Better accuracy, handles complex formatting")
    
    while True:
        choice = input("\nSelect mode (1 or 2) [default=2]: ").strip()
        if choice == "1":
            mode = ChunkingMode.FAST
            break
        elif choice == "2" or not choice:
            mode = ChunkingMode.THOROUGH
            break
        else:
            print("Invalid choice. Please select 1 or 2.")
    
    recursive_config = RecursiveChunkingConfig(mode=mode)
    return ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        recursive_config=recursive_config
    )


def _configure_elastic_chunking() -> ChunkingConfig:
    """Configure Elastic native chunking settings interactively."""
    from .chunker import ElasticChunkingConfig  # Ensure you have access to this class.
    
    print("\nElastic Native Chunking Configuration:")
    print("1. Use default settings")
    print("2. Custom configuration")
    
    while True:
        choice = input("\nSelect option (1 or 2) [default=1]: ").strip()
        if not choice or choice == "1":
            return ChunkingConfig(
                strategy=ChunkingStrategy.ELASTIC_NATIVE,
                elastic_config=ElasticChunkingConfig()
            )
        elif choice == "2":
            return _get_custom_elastic_config()
        else:
            print("Invalid choice. Please select 1 or 2.")


def _get_custom_elastic_config() -> ChunkingConfig:
    """Get custom configuration for Elastic native chunking interactively."""
    print("\nChunking Strategy:")
    print("1. Sentence-based chunking")
    print("2. Word-based chunking")
    
    while True:
        choice = input("\nSelect strategy (1 or 2) [default=1]: ").strip()
        if not choice or choice == "1":
            strategy = "sentence"
            break
        elif choice == "2":
            strategy = "word"
            break
        else:
            print("Invalid choice. Please select 1 or 2.")
    
    # Get max chunk size.
    while True:
        size = input(f"\nMax chunk size (20-300 for sentence, 10-300 for word) [default=250]: ").strip()
        if not size:
            max_chunk_size = 250
            break
        try:
            max_chunk_size = int(size)
            min_size = 20 if strategy == "sentence" else 10
            if min_size <= max_chunk_size <= 300:
                break
            else:
                print(f"Size must be between {min_size} and 300")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get overlap settings based on strategy.
    if strategy == "sentence":
        while True:
            overlap = input("\nSentence overlap (0 or 1) [default=1]: ").strip()
            if not overlap:
                sentence_overlap = 1
                break
            if overlap in ("0", "1"):
                sentence_overlap = int(overlap)
                break
            else:
                print("Please enter 0 or 1.")
        
        elastic_config = ElasticChunkingConfig(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            sentence_overlap=sentence_overlap
        )
    else:  # word strategy.
        while True:
            overlap = input(f"\nWord overlap (up to {max_chunk_size//2}) [default=100]: ").strip()
            if not overlap:
                word_overlap = 100
                break
            try:
                word_overlap = int(overlap)
                if 0 <= word_overlap <= max_chunk_size//2:
                    break
                else:
                    print(f"Overlap must be between 0 and {max_chunk_size//2}")
            except ValueError:
                print("Please enter a valid number.")
        
        elastic_config = ElasticChunkingConfig(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            overlap=word_overlap
        )
    
    return ChunkingConfig(
        strategy=ChunkingStrategy.ELASTIC_NATIVE,
        elastic_config=elastic_config
    )
