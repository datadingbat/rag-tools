# index_workflow.py
from typing import Dict, Any, Optional, Tuple
import logging
import os
from pathlib import Path
from .indexing_manager import IndexingManager, IndexConfig

logger = logging.getLogger(__name__)

class IndexWorkflowManager:
    """Manages the workflow between UI and indexing operations."""
    
    def __init__(self, es_client):
        """Initialize workflow manager with Elasticsearch client."""
        self.indexing_manager = IndexingManager(es_client)
        self.es = es_client

    def handle_pdf_workflow(self, pdf_path: str, dest_config: IndexConfig) -> Tuple[bool, str]:
        """
        Handle complete PDF indexing workflow.
        
        Args:
            pdf_path: Path to PDF file
            dest_config: Destination index configuration
            
        Returns:
            Tuple of (success status, message)
        """
        try:
            # Validate PDF
            if not self._validate_pdf(pdf_path):
                return False, "Invalid PDF file"

            # Process PDF
            success = self.indexing_manager.process_pdf(pdf_path, dest_config)
            
            if success:
                return True, "PDF processed and indexed successfully"
            else:
                return False, "Failed to process PDF"
                
        except Exception as e:
            logger.error(f"Error in PDF workflow: {str(e)}")
            return False, f"Error: {str(e)}"

    def handle_text_workflow(self, text_path: str, dest_config: IndexConfig) -> Tuple[bool, str]:
        """
        Handle complete text file indexing workflow.
        
        Args:
            text_path: Path to text file
            dest_config: Destination index configuration
            
        Returns:
            Tuple of (success status, message)
        """
        try:
            # Validate text file
            if not self._validate_text_file(text_path):
                return False, "Invalid text file"

            # Process text
            success = self.indexing_manager.process_text(text_path, dest_config)
            
            if success:
                return True, "Text file processed and indexed successfully"
            else:
                return False, "Failed to process text file"
                
        except Exception as e:
            logger.error(f"Error in text workflow: {str(e)}")
            return False, f"Error: {str(e)}"

    def handle_reindex_workflow(
        self, 
        source_index: str, 
        dest_config: IndexConfig,
        mapping_selection: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Handle complete re-indexing workflow.
        
        Args:
            source_index: Source index name
            dest_config: Destination index configuration
            mapping_selection: Mapping selection details
            
        Returns:
            Tuple of (success status, message)
        """
        try:
            # Validate source index
            if not self._validate_source_index(source_index):
                return False, f"Source index {source_index} not found or invalid"

            # Apply mapping selection
            dest_config = self._apply_mapping_selection(dest_config, mapping_selection)
            
            # Perform re-indexing
            success = self.indexing_manager.reindex(source_index, dest_config)
            
            if success:
                return True, "Re-indexing completed successfully"
            else:
                return False, "Failed to complete re-indexing"
                
        except Exception as e:
            logger.error(f"Error in re-index workflow: {str(e)}")
            return False, f"Error: {str(e)}"

    def validate_mapping_file(self, mapping_file: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Validate mapping file content.
        
        Args:
            mapping_file: Path to mapping file
            
        Returns:
            Tuple of (is_valid, mapping_dict, message)
        """
        try:
            if not os.path.exists(mapping_file):
                return False, None, "Mapping file not found"

            with open(mapping_file, 'r') as f:
                import json
                mapping = json.load(f)
                
            # Validate mapping structure
            if not isinstance(mapping, dict):
                return False, None, "Invalid mapping format"
                
            if "mappings" not in mapping:
                return False, None, "Missing 'mappings' in mapping file"
                
            # Add more specific validation as needed
            
            return True, mapping, "Mapping file is valid"
            
        except json.JSONDecodeError:
            return False, None, "Invalid JSON format in mapping file"
        except Exception as e:
            return False, None, f"Error validating mapping file: {str(e)}"

    def suggest_mapping_templates(self, source_index: str) -> Dict[str, Any]:
        """
        Suggest appropriate mapping templates based on source index.
        
        Args:
            source_index: Source index name
            
        Returns:
            Dictionary of suggested templates
        """
        try:
            # Get source index mapping
            source_mapping = self.es.indices.get_mapping(index=source_index)
            
            # Analyze mapping characteristics
            has_vectors = self._check_for_vectors(source_mapping[source_index]["mappings"])
            has_elser = self._check_for_elser(source_mapping[source_index]["mappings"])
            
            # Build suggestions
            suggestions = {
                "recommended": [],
                "alternatives": [],
                "reason": ""
            }
            
            if has_vectors:
                suggestions["recommended"].append("template_1")
                suggestions["reason"] += "Source index contains vector fields. "
            
            if has_elser:
                suggestions["recommended"].append("template_2")
                suggestions["reason"] += "Source index uses ELSER. "
            
            # Add alternative templates
            suggestions["alternatives"] = ["template_3", "template_4"]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting templates: {str(e)}")
            return {"error": str(e)}

    def _validate_pdf(self, pdf_path: str) -> bool:
        """Validate PDF file."""
        try:
            if not os.path.exists(pdf_path):
                return False
                
            if not pdf_path.lower().endswith('.pdf'):
                return False
                
            # Add more PDF validation if needed
            
            return True
            
        except Exception:
            return False

    def _validate_text_file(self, text_path: str) -> bool:
        """Validate text file."""
        try:
            if not os.path.exists(text_path):
                return False
                
            # Try to read file
            with open(text_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Read first KB
                
            return True
            
        except Exception:
            return False

    def _validate_source_index(self, index_name: str) -> bool:
        """Validate source index."""
        try:
            if not self.es.indices.exists(index=index_name):
                return False
                
            # Add more index validation if needed
            
            return True
            
        except Exception:
            return False

    def _apply_mapping_selection(
        self, 
        dest_config: IndexConfig, 
        mapping_selection: Dict[str, Any]
    ) -> IndexConfig:
        """Apply mapping selection to destination config."""
        try:
            dest_config.mapping_type = mapping_selection.get("type", "default")
            dest_config.mapping_source = mapping_selection.get("source", "")
            
            if mapping_selection.get("settings"):
                dest_config.settings = mapping_selection["settings"]
                
            return dest_config
            
        except Exception as e:
            logger.error(f"Error applying mapping selection: {str(e)}")
            raise

    def _check_for_vectors(self, mapping: Dict[str, Any]) -> bool:
        """Check if mapping contains vector fields."""
        def check_nested(props):
            for field, config in props.items():
                if isinstance(config, dict):
                    if config.get("type") in ["dense_vector", "sparse_vector"]:
                        return True
                    if "properties" in config:
                        if check_nested(config["properties"]):
                            return True
            return False
            
        return check_nested(mapping.get("properties", {}))

    def _check_for_elser(self, mapping: Dict[str, Any]) -> bool:
        """Check if mapping uses ELSER."""
        def check_nested(props):
            for field, config in props.items():
                if isinstance(config, dict):
                    if config.get("type") == "semantic_text":
                        return True
                    if "properties" in config:
                        if check_nested(config["properties"]):
                            return True
            return False
            
        return check_nested(mapping.get("properties", {}))