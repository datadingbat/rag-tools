# search.py
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import json
from .config import SearchParameters

logger = logging.getLogger(__name__)

class DenseSearcher:
    """Handles dense vector search functionality."""
    
    def __init__(self, es_client: Elasticsearch):
        """Initialize dense searcher with Elasticsearch client."""
        self.es = es_client
        self._model = None
        self.query_instruction = "Represent the question for retrieving relevant technical documentation:"
        #logger.info("DenseSearcher initialized")

    @property
    def model(self):
        """Lazy load the model only when needed."""
        if self._model is None:
            logger.info("Loading Instructor-XL model...")
            self._model = SentenceTransformer('hkunlp/instructor-xl')
            logger.info("Instructor-XL model loaded successfully")
        return self._model

    def search(self, query: str, index_name: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query string
            index_name: Name of the index to search
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        try:
            logger.info(f"Starting dense search for query: {query}")
            query_embedding = self.model.encode([[self.query_instruction, query]])[0].tolist()
            
            search_query = {
                "size": 100,
                "_source": False,
                "query": {
                    "nested": {
                        "path": "passages",
                        "inner_hits": {
                            "size": top_k,
                            "name": "passages",
                            "_source": ["passages.text", "passages.content_type", "passages.chunknumber"],
                            "sort": [{"_score": {"order": "desc"}}]
                        },
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'passages.vector') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    }
                },
                "sort": [{"_score": {"order": "desc"}}]
            }
            
            logger.info(f"Executing search against index: {index_name}")
            response = self.es.search(index=index_name, body=search_query)
            
            results = []
            seen_texts = set()
            
            for hit in response['hits']['hits']:
                inner_hits = hit.get('inner_hits', {}).get('passages', {}).get('hits', {}).get('hits', [])
                for inner_hit in inner_hits:
                    source = inner_hit.get('_source', {})
                    text = source.get('text', '').strip()
                    
                    if not text or text in seen_texts:
                        continue
                        
                    seen_texts.add(text)
                    score = inner_hit.get('_score', 0) - 1
                    
                    results.append({
                        'text': text,
                        'content_type': source.get('content_type', ''),
                        'chunk_number': source.get('chunknumber', None),
                        'score': score
                    })
                    
                    if len(results) >= top_k:
                        break
                if len(results) >= top_k:
                    break
            
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]
            
            logger.info(f"Retrieved {len(results)} results after deduplication")
            if results:
                scores = [r['score'] for r in results]
                logger.info(f"Score range: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Dense search error: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            raise

class EnhancedSearcher:
    """Handles hybrid search combining ELSER and content-type boosting."""
    
    def __init__(self, es_client: Elasticsearch):
        """Initialize enhanced searcher with Elasticsearch client."""
        self.es = es_client
        #logger.info("EnhancedSearcher initialized")

    def hybrid_search(
        self, 
        query: str, 
        index_name: str,
        params: SearchParameters,
        content_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using semantic search and reciprocal rank fusion.
        
        Args:
            query: Search query string
            index_name: Name of the index to search
            params: Search parameters configuration
            content_type_filter: Optional filter for specific content types
            
        Returns:
            List of search results
        """
        try:
            logger.info(f"Starting hybrid search with query: {query}")
            logger.info(f"Parameters: window_size={params.rank_window_size}, "
                       f"size={params.size}, min_score={params.min_score}")

            # Build semantic search query
            body_semantic = {
                "size": params.rank_window_size,
                "_source": ["content", "content_type", "chunk_number"],
                "query": {
                    "semantic": {
                        "field": "content",
                        "query": query
                    }
                }
            }

            # Build content-type boosted query
            body_typed = {
                "size": params.rank_window_size,
                "_source": ["content", "content_type", "chunk_number"],
                "query": {
                    "bool": {
                        "must": {
                            "semantic": {
                                "field": "content",
                                "query": query
                            }
                        },
                        "should": [
                            {
                                "constant_score": {
                                    "filter": {"term": {"content_type": "code"}},
                                    "boost": 1.2
                                }
                            },
                            {
                                "constant_score": {
                                    "filter": {"term": {"content_type": "technical"}},
                                    "boost": 1.1
                                }
                            }
                        ]
                    }
                }
            }

            # Apply content type filter if provided
            if content_type_filter:
                filter_clause = {"term": {"content_type": content_type_filter}}
                for body in [body_semantic, body_typed]:
                    if "bool" not in body["query"]:
                        body["query"] = {"bool": {"must": [body["query"]]}}
                    body["query"]["bool"]["filter"] = [filter_clause]

            # Execute both queries
            semantic_results = self.es.search(index=index_name, body=body_semantic)
            typed_results = self.es.search(index=index_name, body=body_typed)

            semantic_hits = self._process_hits(semantic_results["hits"]["hits"])
            typed_hits = self._process_hits(typed_results["hits"]["hits"])

            logger.info(f"Retrieved {len(semantic_hits)} semantic and {len(typed_hits)} typed results")

            # Combine using RRF
            results_dict = {"semantic": semantic_hits, "typed": typed_hits}
            combined_results = self._reciprocal_rank_fusion(results_dict, k=params.rank_constant)

            logger.info(f"Combined raw results count: {len(combined_results)}")
            
            # Log sample results
            for doc in combined_results[:5]:
                logger.info(f"Score: {doc.get('score', 0):.3f}, "
                          f"Content (first 150 chars): {doc.get('text', '')[:150]}...")
            
            # Filter and sort results
            filtered_results = [
                r for r in combined_results
                if isinstance(r, dict)
                and "score" in r 
                and isinstance(r["score"], (int, float))
                and r["score"] >= params.min_score
            ][:params.size]

            logger.info(f"{len(filtered_results)} results remain after filtering "
                       f"(min_score >= {params.min_score})")
                       
            return filtered_results

        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            raise

    def _process_hits(self, hits: List[Dict]) -> List[Dict[str, Any]]:
        """Process and standardize Elasticsearch hits."""
        processed = []
        for hit in hits:
            content = hit["_source"].get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
                
            processed.append({
                "id": hit["_id"],
                "text": str(content),
                "content_type": hit["_source"].get("content_type", ""),
                "chunk_number": hit["_source"].get("chunk_number", None),
                "score": hit.get("_score", 0)
            })
        return processed

    def _reciprocal_rank_fusion(
        self, 
        results_dict: Dict[str, List[Dict[str, Any]]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            results_dict: Dictionary mapping method names to result lists
            k: RRF constant
            
        Returns:
            Combined and reranked results list
        """
        doc_scores = {}
        
        # Log pre-RRF rankings
        for method_name, ranked_list in results_dict.items():
            logger.info(f"Pre-RRF ranking for {method_name} (Top 5):")
            for idx, result in enumerate(ranked_list[:5]):
                logger.info(f"Rank {idx+1}, Score: {result.get('score', 0):.3f}")
                
            # Calculate RRF scores
            for idx, result in enumerate(ranked_list):
                doc_id = result.get("id")
                if not doc_id:
                    continue
                rank = idx + 1
                rr_score = 1.0 / (k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rr_score

        # Sort and format results
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        merged_results = []
        
        for doc_id, score in sorted_results:
            original_info = None
            for method_results in results_dict.values():
                for r in method_results:
                    if r.get("id") == doc_id:
                        original_info = r
                        break
                if original_info:
                    break
                    
            if original_info:
                merged_results.append({
                    "id": doc_id,
                    "text": original_info.get("text", ""),
                    "content_type": original_info.get("content_type", "unknown"),
                    "chunk_number": original_info.get("chunk_number", None),
                    "score": score
                })
                
        logger.info(f"RRF returned {len(merged_results)} results")
        return merged_results

    def format_results(self, results: List[Dict[str, Any]], wrap_width: int = 80) -> str:
            """
            Format search results for display.
            
            Args:
                results: List of search result dictionaries
                wrap_width: Maximum line width for text wrapping
                
            Returns:
                Formatted string of search results
            """
            if not results:
                return "⚠️ No search results found."
                
            formatted = []
            for i, result in enumerate(results, 1):
                text = result.get("text", "").replace("\n", " ").strip()
                # Split long text into wrapped lines
                wrapped_text = "\n".join(
                    [text[i:i+wrap_width] for i in range(0, len(text), wrap_width)]
                )
                
                formatted.append(
                    f"\n{i}. Score: {result.get('score', 0):.3f}\n"
                    f"Content Type: {result.get('content_type', 'unknown')}\n"
                    f"Chunk Number: {result.get('chunk_number', 'N/A')}\n"
                    f"{'-' * 40}\n"
                    f"{wrapped_text}"
                )
                
            summary = f"\nFound {len(results)} matching results\n"
            return summary + "\n".join(formatted)

    def display_query_explanation(self, query: str, params: SearchParameters) -> str:
        """
        Generate a human-readable explanation of the search query.
        
        Args:
            query: The search query
            params: Search parameters configuration
            
        Returns:
            String explaining the query structure
        """
        explanation = [
            "\n=== Hybrid Search Process ===",
            "-" * 80,
            "\nStep 1: ELSER Semantic Query",
            "This query retrieves semantically relevant documents:",
            json.dumps({
                "size": params.rank_window_size,
                "_source": ["content", "content_type", "chunk_number"],
                "query": {
                    "semantic": {
                        "field": "content",
                        "query": query
                    }
                }
            }, indent=2),
            
            "\nStep 2: ELSER Content-Type Boosted Query",
            "This query adds content-type-specific boosts:",
            json.dumps({
                "size": params.rank_window_size,
                "_source": ["content", "content_type", "chunk_number"],
                "query": {
                    "bool": {
                        "must": {
                            "semantic": {
                                "field": "content",
                                "query": query
                            }
                        },
                        "should": [
                            {
                                "constant_score": {
                                    "filter": {"term": {"content_type": "code"}},
                                    "boost": 1.2
                                }
                            },
                            {
                                "constant_score": {
                                    "filter": {"term": {"content_type": "technical"}},
                                    "boost": 1.1
                                }
                            }
                        ]
                    }
                }
            }, indent=2),
            
            "\nStep 3: Reciprocal Rank Fusion (RRF)",
            "Results from both queries are combined using RRF with these parameters:",
            f"- Window Size: {params.rank_window_size} (top K results from each query)",
            f"- RRF Constant: {params.rank_constant} (k in RRF formula)",
            f"- Minimum Score Threshold: {params.min_score}",
            "\nRRF Formula: score(d) = Σ 1/(k + r_i)",
            "where:",
            "- k is the RRF constant",
            "- r_i is document rank in each result list",
            f"- Final results are filtered to top {params.size} documents"
        ]
        
        return "\n".join(explanation)

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.es.indices.stats(index=index_name)
            mapping = self.es.indices.get_mapping(index=index_name)
            
            return {
                "doc_count": stats["indices"][index_name]["total"]["docs"]["count"],
                "size_bytes": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "field_count": len(mapping[index_name]["mappings"].get("properties", {})),
                "creation_date": mapping[index_name]["settings"]["index"].get("creation_date"),
                "last_modified": stats["indices"][index_name]["total"]["indexing"]["index_time_in_millis"]
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}

    def validate_index_content(self, index_name: str) -> Dict[str, Any]:
        """
        Validate index content and structure.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation = {
                "exists": self.es.indices.exists(index=index_name),
                "issues": []
            }
            
            if not validation["exists"]:
                validation["issues"].append(f"Index {index_name} does not exist")
                return validation
            
            # Check mapping
            mapping = self.es.indices.get_mapping(index=index_name)
            properties = mapping[index_name]["mappings"].get("properties", {})
            
            # Validate required fields
            required_fields = ["content", "content_type", "chunk_number"]
            missing_fields = [field for field in required_fields 
                            if field not in properties]
            
            if missing_fields:
                validation["issues"].append(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )
            
            # Validate field types
            if "content" in properties:
                content_type = properties["content"].get("type")
                if content_type != "semantic_text":
                    validation["issues"].append(
                        f"'content' field should be type 'semantic_text', found '{content_type}'"
                    )
            
            # Check document count
            stats = self.es.indices.stats(index=index_name)
            doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
            if doc_count == 0:
                validation["issues"].append("Index is empty")
            
            validation["valid"] = len(validation["issues"]) == 0
            return validation
            
        except Exception as e:
            logger.error(f"Error validating index: {str(e)}")
            return {
                "exists": False,
                "valid": False,
                "issues": [str(e)]
            }