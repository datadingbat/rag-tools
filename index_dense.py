# index_dense.py
import os
import re
import logging
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from .elastic_config import ElasticsearchConfig
from .chunker import (
    EnhancedRecursiveChunker, 
    ChunkingConfig, 
    ChunkingMode,
    ChunkingStrategy,
    get_chunking_strategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Document loader for processing and indexing with different chunking strategies."""
    
    def __init__(self, chunking_config: ChunkingConfig):
        """Initialize document loader with specified chunking configuration."""
        self.chunking_config = chunking_config
        self._model = None
        self._es = None
        self._connection_validated = False
        
        # Initialize chunker if using recursive strategy
        if chunking_config.strategy == ChunkingStrategy.RECURSIVE:
            self.chunker = EnhancedRecursiveChunker(chunking_config.recursive_config)
            
        # Define specialized instructions for different content types
        self.instructions = {
            'code': "Represent the Elasticsearch/Painless code block for technical implementation and learning:",
            'concept': "Represent the programming concept explanation for learning and understanding:",
            'tutorial': "Represent the programming tutorial text for step-by-step learning and implementation:",
            'technical': "Represent the Elasticsearch technical documentation for comprehension and implementation:",
            'api': "Represent the Elasticsearch API documentation for technical implementation:",
            'default': "Represent the technical documentation for comprehension:"
        }
        
        logger.info("DocumentLoader initialized")

    @property
    def es(self):
        """Lazy initialize Elasticsearch client."""
        if self._es is None:
            config = ElasticsearchConfig()
            self._es = config.get_client()
        return self._es

    def validate_connection(self):
        """Validate Elasticsearch connection when needed."""
        if not self._connection_validated:
            try:
                config = ElasticsearchConfig()
                config.validate_connection(self.es)
                info = self.es.info()
                logger.info(f"Elasticsearch version: {info['version']['number']}")
                self._connection_validated = True
            except Exception as e:
                logger.error(f"Connection validation failed: {str(e)}")
                raise

    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            logger.info("Loading Instructor-XL model...")
            self._model = SentenceTransformer('hkunlp/instructor-xl')
            logger.info("Instructor-XL model loaded successfully")
        return self._model

    def create_index(self, index_name: str):
        """Create index with appropriate mappings."""
        # Validate connection before creating index
        self.validate_connection()
        
        if self.es.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists")
            return
        

        # Choose mapping based on chunking strategy.
        if self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "passages": {
                            "type": "nested",
                            "properties": {
                                "text": {"type": "text"},
                                "chunknumber": {"type": "integer"},
                                "content_type": {"type": "keyword"},
                                "vector": {
                                    "type": "dense_vector",
                                    "dims": 768,
                                    "index": True,
                                    "similarity": "cosine"
                                }
                            }
                        }
                    }
                }
            }
        else:
            # Mapping for native chunking.
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "passages": {
                            "type": "nested",
                            "properties": {
                                "text": {"type": "text"},
                                "vector": {
                                    "type": "dense_vector",
                                    "dims": 384,  # E5 model dimensions.
                                    "index": True,
                                    "similarity": "cosine"
                                }
                            }
                        }
                    }
                }
            }
        
        self.es.indices.create(index=index_name, body=mapping)
        logger.info(f"Index {index_name} created successfully")

    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of content to determine appropriate instruction.
        
        Args:
            text: Content to analyze.
            
        Returns:
            String indicating the content type.
        """
        text_lower = text.lower()
        
        # Check if it's a code block.
        if (text.strip().startswith(('def ', 'class ', '{', 'function', '//', 'int ', 'public ', 'private ', 'String', 'boolean'))) or \
           ('return' in text and ('{' in text or ';' in text)) or \
           ('import ' in text) or \
           any(marker in text for marker in ['void ', 'null', 'true', 'false']) or \
           (text.count(';') > 2):
            return 'code'
        
        # Check if it's API documentation.
        if any(marker in text_lower for marker in ['post ', 'get ', 'put ', 'delete ', '_api', 'endpoint', 'request', 'response']) or \
           ('curl' in text_lower) or \
           (text.strip().startswith('{')):
            return 'api'
        
        # Check if it's a programming concept explanation.
        if any(marker in text_lower for marker in [
            'variable', 'function', 'class', 'method', 'operator', 
            'control flow', 'loop', 'conditional', 'parameter',
            'let\'s understand', 'concept of', 'in programming'
        ]):
            return 'concept'
        
        # Check if it's tutorial-style content.
        if any(marker in text_lower for marker in [
            'let\'s', 'now that we have', 'suppose that', 'as an example',
            'for example', 'notice that', 'note that', 'you can see',
            'try this', 'copy and paste', 'in this example'
        ]):
            return 'tutorial'
            
        # Check if it's technical Elasticsearch documentation.
        if any(marker in text_lower for marker in [
            'elasticsearch', 'kibana', 'elastic stack', 'index', 'query',
            'mapping', 'analyzer', 'pipeline', 'ingest', 'cluster'
        ]):
            return 'technical'
            
        return 'default'

    def _create_inference_pipeline(self, index_name: str, chunking_settings: dict) -> str:
        """
        Create inference pipeline with chunking settings.
        
        Args:
            index_name: Name of the index.
            chunking_settings: Dictionary of chunking configuration.
            
        Returns:
            ID of the created pipeline.
        """
        pipeline_body = {
            "description": "Text embedding pipeline with chunking",
            "processors": [
                {
                    "inference": {
                        "model_id": ".multilingual-e5-small",
                        "target_field": "passages",
                        "field_map": {
                            "content": "text_field"
                        },
                        "chunking_settings": chunking_settings
                    }
                }
            ]
        }
        
        pipeline_id = f"{index_name}_pipeline"
        
        # Delete existing pipeline if it exists.
        self._delete_pipeline(pipeline_id)
        
        # Create new pipeline.
        self.es.ingest.put_pipeline(
            id=pipeline_id,
            body=pipeline_body
        )
        logger.info(f"Created inference pipeline: {pipeline_id}")
        return pipeline_id

    def process_and_index_file(self, file_path: str, index_name: str):
        """Process and index a file using the configured chunking strategy."""
        try:
            # Validate connection before processing
            self.validate_connection()
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
                # Use recursive chunker
                passages = self.chunker.chunk_text(content)
                
                # Process each passage
                for passage in passages:
                    content_type = self.detect_content_type(passage["text"])
                    instruction = self.instructions.get(content_type, self.instructions['default'])
                    
                    # Generate embedding using the lazily-loaded model
                    embedding = self.model.encode([[instruction, passage["text"]]])[0]
                    
                    passage["content_type"] = content_type
                    passage["vector"] = embedding.tolist()
                
                final_doc = {
                    "content": content,
                    "passages": passages
                }
                self.es.index(index=index_name, body=final_doc)
            else:
                # Create inference pipeline for native chunking
                pipeline_id = self._create_inference_pipeline(
                    index_name, 
                    self.chunking_config.elastic_config.to_dict()
                )
                self.es.index(
                    index=index_name,
                    pipeline=pipeline_id,
                    body={"content": content}
                )
            
            logger.info(f"Successfully indexed document to {index_name}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
        
    def _delete_pipeline(self, pipeline_id: str):
        """
        Delete an existing pipeline if it exists.
        
        Args:
            pipeline_id: ID of the pipeline to delete.
        """
        if self.es.ingest.get_pipeline(id=pipeline_id, ignore=[404]):
            self.es.ingest.delete_pipeline(id=pipeline_id)
            logger.info(f"Deleted existing pipeline: {pipeline_id}")

    def cleanup(self, index_name: str):
        """
        Cleanup resources associated with an index.
        
        Args:
            index_name: Name of the index to clean up.
        """
        pipeline_id = f"{index_name}_pipeline"
        self._delete_pipeline(pipeline_id)

def main():
    """Main entry point for document processing."""
    try:
        # Get chunking configuration interactively.
        chunking_config = get_chunking_strategy()
        
        # Initialize loader with the chosen configuration.
        loader = DocumentLoader(chunking_config)
        
        # Create index.
        loader.create_index("ml_documentation")
        
        # Process and index the file.
        loader.process_and_index_file("ml.txt", "ml_documentation")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    finally:
        if 'loader' in locals():
            loader.cleanup("ml_documentation")

if __name__ == "__main__":
    main()
