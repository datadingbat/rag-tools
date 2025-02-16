# indexing_manager.py
from typing import Dict, Any, Optional, List, Union
import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass

from .index_dense import DocumentLoader
from .index_elser import EnhancedDocumentLoader
from .chunker import (
    ChunkingConfig, 
    ChunkingStrategy,
    ChunkingMode,
    RecursiveChunkingConfig,
    ElasticChunkingConfig
)
from .pdf2txt import PDFTextExtractor

logger = logging.getLogger(__name__)

@dataclass
class IndexConfig:
    """Configuration for index creation."""
    name: str
    mapping_type: str
    mapping_source: str
    exists: bool = False
    settings: Optional[Dict[str, Any]] = None

class IndexingManager:
    """Manages indexing operations including PDF, text, and re-indexing."""
    
    def __init__(self, es_client):
        """
        Initialize indexing manager.
        
        Args:
            es_client: Elasticsearch client instance
        """
        self.es = es_client
        self.pdf_extractor = PDFTextExtractor()
        
        # Initialize base chunking configs
        self.chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            recursive_config=RecursiveChunkingConfig(
                mode=ChunkingMode.THOROUGH
            )
        )
        
        # Initialize loaders
        self.dense_loader = DocumentLoader(self.chunking_config)
        self.elser_loader = EnhancedDocumentLoader(self.chunking_config)
        
        logger.info("IndexingManager initialized")

    def process_pdf(self, pdf_path: str, index_config: IndexConfig, embedding_type: str = "both") -> bool:
        """
        Process and index PDF document.
        
        Args:
            pdf_path: Path to PDF file
            index_config: Index configuration
            embedding_type: Type of embeddings to generate ("dense", "elser", or "both")
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text from PDF
            page_texts = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            # Save to temporary file
            temp_file = Path(pdf_path).with_suffix('.txt')
            self.pdf_extractor.save_extracted_text(
                page_texts=page_texts,
                output_path=str(temp_file),
                split_pages=False
            )
            
            # Process text file
            success = self.process_text(str(temp_file), index_config, embedding_type)
            
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
                
            return success
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return False

    def process_text(self, text_path: str, index_config: IndexConfig, embedding_type: str = "both") -> bool:
        """
        Process and index text file.
        
        Args:
            text_path: Path to text file
            index_config: Index configuration
            embedding_type: Type of embeddings to generate
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Processing text file: {text_path}")
            
            # Prepare indices based on embedding type
            if embedding_type in ["dense", "both"]:
                dense_index = f"{index_config.name}_dense"
                self._prepare_index(dense_index, index_config, "dense")
                
            if embedding_type in ["elser", "both"]:
                elser_index = f"{index_config.name}_elser"
                self._prepare_index(elser_index, index_config, "elser")
            
            # Process file for each requested embedding type
            if embedding_type in ["dense", "both"]:
                self.dense_loader.process_and_index_file(text_path, dense_index)
                logger.info(f"Dense vectors indexed to {dense_index}")
                
            if embedding_type in ["elser", "both"]:
                self.elser_loader.process_and_index_file(text_path, elser_index)
                logger.info(f"ELSER vectors indexed to {elser_index}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            return False

    def reindex(self, source_index: str, dest_config: IndexConfig, reindex_type: str = "both") -> bool:
        """
        Re-index from existing index.
        
        Args:
            source_index: Source index name
            dest_config: Destination index configuration
            reindex_type: Type of re-indexing to perform
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Re-indexing from {source_index}")
            
            # Verify source index
            if not self.es.indices.exists(index=source_index):
                raise ValueError(f"Source index {source_index} does not exist")
            
            # Get source mapping if needed
            if dest_config.mapping_type == "source_copy":
                mapping = self.es.indices.get_mapping(index=source_index)
                dest_config.settings = mapping[source_index]["mappings"]
            
            # Prepare destination indices
            if reindex_type in ["dense", "both"]:
                dense_dest = f"{dest_config.name}_dense"
                self._prepare_index(dense_dest, dest_config, "dense")
                
            if reindex_type in ["elser", "both"]:
                elser_dest = f"{dest_config.name}_elser"
                self._prepare_index(elser_dest, dest_config, "elser")
            
            # Perform re-indexing
            if reindex_type in ["dense", "both"]:
                self._reindex_with_type(source_index, dense_dest, "dense")
                
            if reindex_type in ["elser", "both"]:
                self._reindex_with_type(source_index, elser_dest, "elser")
                
            return True
            
        except Exception as e:
            logger.error(f"Error during re-indexing: {str(e)}")
            return False

    def _prepare_index(self, index_name: str, config: IndexConfig, index_type: str):
        """Prepare index with appropriate mapping."""
        try:
            # Delete existing index if needed
            if self.es.indices.exists(index=index_name):
                if not config.exists:
                    logger.info(f"Deleting existing index: {index_name}")
                    self.es.indices.delete(index=index_name)
                else:
                    logger.info(f"Keeping existing index: {index_name}")
                    return
            
            # Get mapping based on configuration
            mapping = self._get_mapping(config, index_type)
            
            # Create index
            self.es.indices.create(
                index=index_name,
                body=mapping
            )
            logger.info(f"Created index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error preparing index: {str(e)}")
            raise

    def _get_mapping(self, config: IndexConfig, index_type: str) -> Dict[str, Any]:
        """Get appropriate mapping based on configuration."""
        try:
            if config.mapping_type == "source_copy" and config.settings:
                return {"mappings": config.settings}
                
            elif config.mapping_type == "file":
                with open(config.mapping_source, 'r') as f:
                    return json.load(f)
                    
            elif config.mapping_type == "index_copy":
                mapping = self.es.indices.get_mapping(index=config.mapping_source)
                return {"mappings": mapping[config.mapping_source]["mappings"]}
                
            elif config.mapping_type == "template":
                return self._get_template_mapping(config.mapping_source, index_type)
                
            else:
                return self._get_default_mapping(index_type)
                
        except Exception as e:
            logger.error(f"Error getting mapping: {str(e)}")
            raise

    def _get_template_mapping(self, template_id: str, index_type: str) -> Dict[str, Any]:
        """Get mapping from template."""
        # Template mappings for different index types
        templates = {
            "template_1": {
                "dense": {
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
                },
                "elser": {
                    "mappings": {
                        "properties": {
                            "content": {
                                "type": "semantic_text",
                                "inference_id": ".elser-2-elasticsearch"
                            },
                            "chunk_number": {"type": "integer"},
                            "content_type": {"type": "keyword"}
                        }
                    }
                }
            },
            # Add more templates as needed
        }
        
        if template_id not in templates:
            raise ValueError(f"Unknown template: {template_id}")
            
        return templates[template_id][index_type]

    def _get_default_mapping(self, index_type: str) -> Dict[str, Any]:
        """Get default mapping for index type."""
        if index_type == "dense":
            return self.dense_loader.create_index
        else:
            return self.elser_loader.create_enhanced_index

    def _reindex_with_type(self, source_index: str, dest_index: str, index_type: str):
        """Perform re-indexing with specific embedding type."""
        try:
            # Set up reindex body
            reindex_body = {
                "source": {
                    "index": source_index
                },
                "dest": {
                    "index": dest_index
                }
            }
            
            # Add pipeline if needed
            if index_type == "elser":
                pipeline_id = self.elser_loader._create_inference_pipeline(
                    dest_index,
                    self.chunking_config.elastic_config.to_dict() if self.chunking_config.elastic_config else None
                )
                reindex_body["dest"]["pipeline"] = pipeline_id
            
            # Execute reindex
            self.es.reindex(body=reindex_body, wait_for_completion=True)
            logger.info(f"Completed re-indexing to {dest_index}")
            
        except Exception as e:
            logger.error(f"Error during re-indexing: {str(e)}")
            raise

    def cleanup(self, index_names: List[str]):
        """
        Cleanup resources for multiple indices.
        
        Args:
            index_names: List of index names to clean up
        """
        for index in index_names:
            try:
                if "_dense" in index:
                    self.dense_loader.cleanup(index)
                elif "_elser" in index:
                    self.elser_loader.cleanup(index)
            except Exception as e:
                logger.error(f"Error cleaning up {index}: {str(e)}")