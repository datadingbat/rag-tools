# config.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import logging
from elasticsearch import Elasticsearch
from .elastic_config import ElasticsearchConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchParameters:
    """Configuration parameters for search functionality."""
    size: int = 8
    min_score: float = 0.07
    rank_constant: int = 5
    rank_window_size: int = 50
    dense_top_k: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format."""
        return {
            "size": self.size,
            "min_score": self.min_score,
            "rank_constant": self.rank_constant,
            "rank_window_size": self.rank_window_size,
            "dense_top_k": self.dense_top_k
        }

    def update(self, **kwargs) -> None:
        """Update parameters with validation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Add validation rules
                if key == "size" and not (1 <= value <= 20):
                    raise ValueError("Size must be between 1 and 20")
                elif key == "min_score" and not (0.0 <= value <= 1.0):
                    raise ValueError("Min score must be between 0.0 and 1.0")
                elif key == "rank_constant" and not (1 <= value <= 100):
                    raise ValueError("Rank constant must be between 1 and 100")
                elif key == "rank_window_size" and not (10 <= value <= 200):
                    raise ValueError("Window size must be between 10 and 200")
                elif key == "dense_top_k" and not (1 <= value <= 20):
                    raise ValueError("Dense top K must be between 1 and 20")
                    
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

def get_elasticsearch_client() -> Elasticsearch:
    """Initialize and validate Elasticsearch client."""
    try:
        config = ElasticsearchConfig()
        client = config.get_client()
        config.validate_connection(client)
        
        # Get cluster info
        info = client.info()
        logger.info(f"Connected to Elasticsearch {info['version']['number']}")
        logger.info(f"Cluster: {info['cluster_name']}")
        
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
        raise