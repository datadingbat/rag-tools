# index_elser.py
import os
import logging
from elasticsearch import Elasticsearch
from typing import Dict, List, Optional

# Use a relative import if this file is part of a package:
from .elastic_config import ElasticsearchConfig
from .chunker import (
    EnhancedRecursiveChunker, 
    ChunkingConfig, 
    ChunkingMode,
    ChunkingStrategy,
    get_chunking_strategy  # You might not use this directly here.
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentLoader:
    """Document loader with support for ELSER and multiple chunking strategies."""
    
    def __init__(self, chunking_config: ChunkingConfig):
        """
        Initialize document loader with specified chunking configuration.
        
        Args:
            chunking_config: Configuration specifying chunking strategy and settings
        """
        self.chunking_config = chunking_config
        
        # Initialize the chunker if using recursive strategy.
        # (Note: if the strategy is ELASTIC_NATIVE, we don't need to create a chunker.)
        if self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
            # Note: We pass in the recursive_config portion.
            self.chunker = EnhancedRecursiveChunker(self.chunking_config.recursive_config)
        
        # Initialize Elasticsearch client
        config = ElasticsearchConfig()
        self.es = config.get_client()
        
        # Validate connection
        if not self.es.ping():
            raise ConnectionError("Could not connect to Elasticsearch")
        logger.info("Successfully connected to Elasticsearch")

    def create_enhanced_index(self, index_name: str) -> None:
        """Create index with appropriate mappings for ELSER.
        
        Args:
            index_name: Name of the index to create
        """
        if self.es.indices.exists(index=index_name):
            logger.info(f"Index {index_name} exists. Deleting...")
            self.es.indices.delete(index=index_name)

        # Choose mapping based on chunking strategy.
        if self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
            # Mapping for recursive chunking with explicit chunk storage.
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "passages": {
                            "type": "nested",
                            "properties": {
                                "text": {"type": "text"},
                                "chunk_number": {"type": "integer"},
                                "content_type": {"type": "keyword"},
                                "sparse_vector": {"type": "sparse_vector"}
                            }
                        }
                    }
                }
            }
        else:
            # Mapping for native ELSER chunking.
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "semantic_text",  # ELSER field type.
                            "inference_id": ".elser-2-elasticsearch"
                        },
                        "chunk_number": {"type": "integer"},
                        "content_type": {"type": "keyword"}
                    }
                }
            }
        
        self.es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created enhanced index '{index_name}'")

    def detect_content_type(self, text: str) -> str:
        """Detect the type of content in the text.
        
        Args:
            text: Content to analyze
            
        Returns:
            String indicating the content type
        """
        text_lower = text.lower()
        # (Detection logic unchanged)
        if (text.strip().startswith(
            ('def ', 'class ', '{', 'function', '//', 'int ', 'public ', 'private ', 'String', 'boolean')
        )) or ('return' in text and ('{' in text or ';' in text)) or ('import ' in text) or any(
            marker in text for marker in ['void ', 'null', 'true', 'false']
        ) or (text.count(';') > 2):
            return 'code'
        if any(marker in text_lower for marker in ['post ', 'get ', 'put ', 'delete ', '_api', 'endpoint', 'request', 'response']) \
           or ('curl' in text_lower) or (text.strip().startswith('{')):
            return 'api'
        if any(marker in text_lower for marker in [
            'variable', 'function', 'class', 'method', 'operator', 
            'control flow', 'loop', 'conditional', 'parameter',
            'let\'s understand', 'concept of', 'in programming'
        ]):
            return 'concept'
        if any(marker in text_lower for marker in [
            'let\'s', 'now that we have', 'suppose that', 'as an example',
            'for example', 'notice that', 'note that', 'you can see',
            'try this', 'copy and paste', 'in this example'
        ]):
            return 'tutorial'
        if any(marker in text_lower for marker in [
            'elasticsearch', 'kibana', 'elastic stack', 'index', 'query',
            'mapping', 'analyzer', 'pipeline', 'ingest', 'cluster'
        ]):
            return 'technical'
        return 'default'

    def _create_inference_pipeline(self, index_name: str, chunking_settings: Optional[Dict] = None) -> str:
        """Create ELSER inference pipeline.
        
        Args:
            index_name: Name of the index
            chunking_settings: Optional chunking configuration for native mode
            
        Returns:
            ID of the created pipeline
        """
        pipeline_body = {
            "description": "Pipeline for ELSER inference with chunking",
            "processors": [
                {
                    "inference": {
                        "model_id": ".elser_model_2",
                        "target_field": "_ingest._value.sparse_vector",
                        "field_map": {"_ingest._value.text": "text_field"}
                    }
                }
            ]
        }
        
        if chunking_settings:
            pipeline_body["processors"][0]["inference"]["chunking_settings"] = chunking_settings
        
        pipeline_id = f"{index_name}_pipeline"
        
        self._delete_pipeline(pipeline_id)
        
        self.es.ingest.put_pipeline(id=pipeline_id, body=pipeline_body)
        logger.info(f"Created inference pipeline: {pipeline_id}")
        return pipeline_id

    def process_and_index_file(self, file_path: str, index_name: str, batch_size: int = 10) -> None:
        """Process and index a file using the configured chunking strategy.
        
        Args:
            file_path: Path to the file to process
            index_name: Name of the index
            batch_size: Number of chunks to process in each batch
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
                # Use recursive chunker
                chunks = self.chunker.chunk_text(content)
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    for chunk in batch:
                        doc = {
                            "content": chunk["text"],
                            "chunk_number": chunk["chunknumber"],
                            "content_type": self.detect_content_type(chunk["text"])
                        }
                        max_retries = 3
                        retry_count = 0
                        while retry_count < max_retries:
                            try:
                                self.es.index(index=index_name, body=doc)
                                break
                            except Exception as e:
                                retry_count += 1
                                if retry_count == max_retries:
                                    logger.error(f"Failed to index chunk {chunk['chunknumber']} after {max_retries} attempts")
                                    raise
                                logger.warning(f"Retry {retry_count} for chunk {chunk['chunknumber']}")
                    
                    logger.info(f"Processed chunks {i+1} to {min(i+batch_size, len(chunks))}")
                
                logger.info(f"Successfully indexed {len(chunks)} chunks to '{index_name}'")
                
            else:
                # For Elastic native chunking
                pipeline_id = self._create_inference_pipeline(
                    index_name, 
                    self.chunking_config.elastic_config.to_dict() if self.chunking_config.elastic_config else None
                )
                doc = {
                    "content": content,
                    "content_type": self.detect_content_type(content)
                }
                self.es.index(index=index_name, pipeline=pipeline_id, body=doc)
                logger.info(f"Successfully indexed document to '{index_name}' with native chunking")
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def _delete_pipeline(self, pipeline_id: str) -> None:
        """Delete an existing pipeline if it exists."""
        if self.es.ingest.get_pipeline(id=pipeline_id, ignore=[404]):
            self.es.ingest.delete_pipeline(id=pipeline_id)
            logger.info(f"Deleted existing pipeline: {pipeline_id}")

    def cleanup(self, index_name: str) -> None:
        """Cleanup resources associated with an index."""
        pipeline_id = f"{index_name}_pipeline"
        self._delete_pipeline(pipeline_id)

def main():
    """Main entry point for document processing."""
    try:
        # The UI (or calling workflow) should supply the chunking configuration.
        # Here we call get_chunking_strategy() for testing purposes.
        chunking_config = get_chunking_strategy()
        
        loader = EnhancedDocumentLoader(chunking_config)
        
        loader.create_enhanced_index("ml_documentation_enhanced")
        loader.process_and_index_file("ml.txt", "ml_documentation_enhanced")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    finally:
        if 'loader' in locals():
            loader.cleanup("ml_documentation_enhanced")

if __name__ == "__main__":
    main()
