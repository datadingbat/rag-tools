# utils.py
from typing import Dict, List, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SearchResultFormatter:
    """Handles formatting of search results and related data."""
    
    @staticmethod
    def format_search_results(results: List[Dict[str, Any]], wrap_width: int = 80) -> str:
        """Format search results for display."""
        if not results:
            return "⚠️ No search results found."
            
        formatted = []
        for i, result in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"Content Type: {result.get('content_type', 'unknown')}")
            print(f"Chunk Number: {result.get('chunk_number', 'N/A')}")
            print(f"Relevance Score: {result.get('score', 0):.3f}")
            print("-" * 40)
            print(f"Content:\n{result.get('text', '')}")
            print("-" * 80)

    @staticmethod
    def format_query_explanation(strategy: str, query: str, search_params: Any) -> str:
        """Format query explanation based on search strategy."""
        if strategy in ["2", "4"]:  # ELSER/Hybrid
            explanation = [
                "\n=== Hybrid Search Process ===",
                "-" * 80,
                "\nStep 1: ELSER Semantic Query",
                "This query retrieves semantically relevant documents:",
                json.dumps({
                    "size": search_params.rank_window_size,
                    "_source": ["content", "content_type", "chunk_number"],
                    "query": {
                        "semantic": {
                            "field": "content",
                            "query": query
                        }
                    }
                }, indent=2)
            ]
            # Add rest of explanation...
            return "\n".join(explanation)
            
        elif strategy == "3":  # Dense
            return "\n=== Dense Vector Search Process ===\n" + \
                   "1. Generate query embedding using Instructor-XL\n" + \
                   "2. Execute dense vector search with cosine similarity"

    @staticmethod
    def display_indices_list(indices: List[Dict[str, Any]]) -> str:
        """
        Format and display list of indices.
        
        Args:
            indices: List of index information dictionaries
            
        Returns:
            Formatted string for display
        """
        if not indices:
            return "No indices found."
            
        output = [
            f"\nFound {len(indices)} indices:",
            f"{'Index Name':<50} {'Health':<8} {'Docs':<10} {'Size':<10}",
            "-" * 80
        ]
        
        for idx in indices:
            name = idx.get('index', 'N/A')
            health = idx.get('health', 'N/A')
            docs = idx.get('docs.count', 'N/A')
            size = idx.get('store.size', 'N/A')
            
            # Color-code health status
            health_color = {
                'green': '\033[92m',
                'yellow': '\033[93m',
                'red': '\033[91m'
            }.get(health.lower(), '')
            reset_color = '\033[0m'
            
            output.append(
                f"{name:<50} {health_color}{health:<8}{reset_color} {docs:<10} {size:<10}"
            )
            
        return "\n".join(output)

    @staticmethod
    def format_mapping_info(mapping: Dict[str, Any], settings: Dict[str, Any]) -> str:
        """
        Format index mapping and settings information.
        
        Args:
            mapping: Index mapping dictionary
            settings: Index settings dictionary
            
        Returns:
            Formatted string for display
        """
        output = []
        
        # Format creation date
        for index_name, index_settings in settings.items():
            creation_date = index_settings.get('settings', {}).get('index', {}).get('creation_date')
            if creation_date:
                created = datetime.fromtimestamp(int(creation_date) / 1000)
                output.append(f"\nCreated: {created}")
        
        # Format mapping
        output.append("\nField Mappings:")
        for index_name, index_mapping in mapping.items():
            properties = index_mapping.get('mappings', {}).get('properties', {})
            
            for field_name, field_info in properties.items():
                output.append(f"\n{field_name}:")
                for prop, value in field_info.items():
                    output.append(f"  {prop}: {value}")
        
        return "\n".join(output)

    @staticmethod
    def format_pipeline_list(pipelines: Dict[str, Any]) -> str:
        """
        Format list of pipelines for display.
        
        Args:
            pipelines: Dictionary of pipeline information
            
        Returns:
            Formatted string for display
        """
        if not pipelines:
            return "No ingest pipelines found."
            
        output = [f"\nFound {len(pipelines)} pipelines:"]
        
        for name, pipeline in pipelines.items():
            output.extend([
                f"\n=== Pipeline: {name} ===",
                f"Description: {pipeline.get('description', 'No description')}",
                f"Processors: {len(pipeline.get('processors', []))}"
            ])
            
            for i, processor in enumerate(pipeline.get('processors', []), 1):
                processor_type = next(iter(processor.keys()))
                output.extend([
                    f"\nProcessor {i}:",
                    f"Type: {processor_type}"
                ])
                
                config = processor[processor_type]
                if isinstance(config, dict):
                    for key, value in config.items():
                        output.append(f"  {key}: {value}")
                        
            output.append("")  # Add spacing between pipelines
            
        return "\n".join(output)

    @staticmethod
    def format_health_status(health: str) -> str:
        """
        Format health status with color coding.
        
        Args:
            health: Health status string
            
        Returns:
            Color-coded health status string
        """
        colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m'
        }
        reset = '\033[0m'
        
        return f"{colors.get(health.lower(), '')}{health}{reset}"

class IndexUtils:
    """Utilities for index management and analysis."""
    
    @staticmethod
    def parse_size(size_str: str) -> int:
        """Parse size string to bytes."""
        try:
            if not size_str or size_str == '-':
                return 0
                
            multipliers = {
                'b': 1,
                'kb': 1024,
                'mb': 1024 ** 2,
                'gb': 1024 ** 3,
                'tb': 1024 ** 4
            }
            
            size_str = size_str.lower()
            for unit, multiplier in multipliers.items():
                if size_str.endswith(unit):
                    try:
                        return int(float(size_str[:-len(unit)]) * multiplier)
                    except ValueError:
                        return 0
            return 0
            
        except Exception:
            return 0

    @staticmethod
    def format_indices_table(indices: List[Dict[str, Any]], sort_by: str = 'name') -> str:
        """Format indices information as a table."""
        if not indices:
            return "No indices found."
            
        headers = {
            'name': 'Index Name',
            'docs': 'Doc Count',
            'size': 'Size',
            'health': 'Health'
        }
        
        # Sort indices
        if sort_by == 'size':
            sorted_indices = sorted(
                indices,
                key=lambda x: IndexUtils.parse_size(x.get('store.size', '0b')),
                reverse=True
            )
        elif sort_by == 'docs':
            sorted_indices = sorted(
                indices,
                key=lambda x: int(x.get('docs.count', 0)),
                reverse=True
            )
        else:
            sorted_indices = sorted(indices, key=lambda x: x.get('index', ''))
            
        # Format table
        table = [
            f"{'Index Name':<40} {'Health':<8} {'Status':<8} {'Docs':<10} {'Size':<10}",
            "-" * 76
        ]
        
        for idx in sorted_indices:
            name = idx.get('index', 'N/A')
            health = idx.get('health', 'N/A')
            status = idx.get('status', 'N/A')
            docs = idx.get('docs.count', 'N/A')
            size = idx.get('store.size', 'N/A')
            
            table.append(f"{name:<40} {health:<8} {status:<8} {docs:<10} {size:<10}")
            
        return "\n".join(table)

    @staticmethod
    def format_pipeline_info(pipeline: Dict[str, Any]) -> str:
        """Format pipeline information for display."""
        info = [
            f"Description: {pipeline.get('description', 'No description')}",
            f"Processors: {len(pipeline.get('processors', []))}\n"
        ]
        
        for i, processor in enumerate(pipeline.get('processors', []), 1):
            processor_type = next(iter(processor.keys()))
            info.append(f"Processor {i}:")
            info.append(f"Type: {processor_type}")
            
            config = processor[processor_type]
            if isinstance(config, dict):
                for key, value in config.items():
                    info.append(f"  {key}: {value}")
            info.append("")
            
        return "\n".join(info)

class ValidationUtils:
    """Utilities for input validation and verification."""
    
    @staticmethod
    def validate_search_params(params: Dict[str, Any]) -> List[str]:
        """Validate search parameters."""
        errors = []
        
        if 'size' in params and not (1 <= params['size'] <= 20):
            errors.append("Size must be between 1 and 20")
            
        if 'min_score' in params and not (0.0 <= params['min_score'] <= 1.0):
            errors.append("Min score must be between 0.0 and 1.0")
            
        if 'rank_constant' in params and not (1 <= params['rank_constant'] <= 100):
            errors.append("Rank constant must be between 1 and 100")
            
        if 'rank_window_size' in params and not (10 <= params['rank_window_size'] <= 200):
            errors.append("Window size must be between 10 and 200")
            
        return errors

    @staticmethod
    def validate_index_name(name: str) -> bool:
        """Validate index name format."""
        import re
        # Check for lowercase letters, numbers, hyphens, and underscores
        pattern = r'^[a-z0-9][a-z0-9_-]*$'
        return bool(re.match(pattern, name))

class TimeUtils:
    """Utilities for time-related operations."""
    
    @staticmethod
    def format_timestamp(timestamp: int) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromtimestamp(timestamp / 1000)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return "Invalid timestamp"

    @staticmethod
    def get_relative_time(timestamp: int) -> str:
        """Get relative time description."""
        try:
            dt = datetime.fromtimestamp(timestamp / 1000)
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 365:
                return f"{diff.days // 365} years ago"
            elif diff.days > 30:
                return f"{diff.days // 30} months ago"
            elif diff.days > 0:
                return f"{diff.days} days ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600} hours ago"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60} minutes ago"
            else:
                return "just now"
        except Exception:
            return "unknown time ago"