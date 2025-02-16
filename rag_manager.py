# rag_manager.py
from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime
from anthropic import Anthropic
from .search import DenseSearcher, EnhancedSearcher
from .config import SearchParameters

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages RAG functionality, including prompt management and LLM integration."""
    
    def __init__(self, es_client):
        """
        Initialize RAG manager with necessary components.
        
        Args:
            es_client: Elasticsearch client instance
        """
        self.dense_searcher = DenseSearcher(es_client)
        self.enhanced_searcher = EnhancedSearcher(es_client)
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Initialize system prompts
        self.system_prompts = {
            "pure": (
                "You are a helpful AI assistant with expertise in Elasticsearch and document processing.\n"
                "Provide clear, accurate information based on your knowledge.\n\n"
                "When answering:\n"
                "- Be concise but thorough\n"
                "- Use clear examples when helpful\n"
                "- Acknowledge any limitations or uncertainties\n"
                "- Focus on practical, actionable information"
            ),
            "rag": (
                "You are a helpful AI assistant with expertise in Elasticsearch and document processing.\n"
                "You have been provided with relevant documentation passages that are:\n"
                "1. Ranked by relevance\n"
                "2. Grouped by content type (code, technical, concept, etc.)\n"
                "3. Selected using retrieval-augmented generation (RAG)\n\n"
                "When answering:\n"
                "- Draw primarily from the provided passages, citing them by passage number\n"
                "- Use code examples when available\n"
                "- Distinguish between conceptual and technical details\n"
                "- Indicate if supplementing with external knowledge\n"
                "- Be specific about which passages support your statements\n"
                "- If passages conflict, explain the discrepancy"
            )
        }
        #logger.info("RAGManager initialized successfully")

    def get_pure_llm_response(self, query: str) -> str:
        """
        Get response from LLM without retrieval context.
        
        Args:
            query: User query string
            
        Returns:
            Formatted LLM response
        """
        try:
            logger.info(f"Getting pure LLM response for query: {query}")
            message = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system=self.system_prompts["pure"],
                messages=[{"role": "user", "content": query}]
            )
            response = self._format_llm_response(message)
            logger.info("Successfully received pure LLM response")
            return response
            
        except Exception as e:
            logger.error(f"Error getting pure LLM response: {str(e)}")
            raise

    def get_rag_response(self, 
                        query: str, 
                        search_results: List[Dict[str, Any]], 
                        conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get RAG-enhanced response using search results.
        
        Args:
            query: User query string
            search_results: List of retrieved passages
            conversation_history: Optional list of previous conversation turns
            
        Returns:
            Formatted RAG response
        """
        try:
            logger.info(f"Getting RAG response for query: {query}")
            logger.info(f"Using {len(search_results)} search results")
            
            context_data = self._organize_context(search_results)
            
            # Build prompt with context
            prompt = self._build_rag_prompt(query, context_data, conversation_history)
            
            message = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system=self.system_prompts["rag"],
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = self._format_llm_response(message)
            logger.info("Successfully received RAG response")
            return response
            
        except Exception as e:
            logger.error(f"Error getting RAG response: {str(e)}")
            raise

    def get_dense_rag_response(self, 
                             query: str, 
                             index_name: str,
                             params: SearchParameters,
                             conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get RAG response using dense vector retrieval.
        
        Args:
            query: User query string
            index_name: Name of the index to search
            params: Search parameters
            conversation_history: Optional list of previous conversation turns
            
        Returns:
            Formatted RAG response using dense retrieval
        """
        try:
            logger.info(f"Getting dense RAG response for query: {query}")
            
            # Perform dense search
            dense_results = self.dense_searcher.search(
                query=query,
                index_name=index_name,
                top_k=params.dense_top_k
            )
            
            if not dense_results:
                logger.warning("No relevant passages found via dense retrieval")
                return "⚠️ No relevant passages found via dense retrieval."
            
            # Get RAG response using dense results
            context_data = self._organize_context(dense_results)
            prompt = self._build_rag_prompt(query, context_data, conversation_history)
            
            message = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system=self.system_prompts["rag"],
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = self._format_llm_response(message)
            logger.info("Successfully received dense RAG response")
            return response
            
        except Exception as e:
            logger.error(f"Error getting dense RAG response: {str(e)}")
            raise

    def _organize_context(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Organize search results into structured context.
        
        Args:
            search_results: List of search results to organize
            
        Returns:
            Dictionary containing organized context data
        """
        # Group results by content type
        content_groups = {}
        for result in search_results:
            ctype = result['content_type']
            content_groups.setdefault(ctype, []).append(result)

        # Calculate statistics
        type_counts = {ctype: len(results) for ctype, results in content_groups.items()}
        all_scores = [r['score'] for r in search_results]
        score_range = {
            'min': min(all_scores) if all_scores else 0,
            'max': max(all_scores) if all_scores else 0,
            'avg': sum(all_scores) / len(all_scores) if all_scores else 0
        }

        # Format context passages
        context_parts = []
        for ctype, results in content_groups.items():
            context_parts.append(f"\n{ctype.upper()} PASSAGES ({len(results)} found):")
            for i, result in enumerate(results, 1):
                relevance = "HIGH" if result['score'] > score_range['avg'] else "MEDIUM"
                context_parts.append(
                    f"[{ctype.upper()} Passage {i}] "
                    f"(Relevance: {relevance}, Score: {result['score']:.3f}):\n"
                    f"{result['text']}"
                )

        return {
            'formatted_context': "\n\n".join(context_parts),
            'type_counts': type_counts,
            'score_range': score_range
        }

    def _build_rag_prompt(self, 
                         query: str, 
                         context_data: Dict[str, Any],
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build RAG prompt with context and optional conversation history.
        
        Args:
            query: User query string
            context_data: Organized context data
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "I have a question about Elasticsearch documentation.",
            "Here are relevant passages organized by relevance:",
            "",
            context_data['formatted_context'],
            "",
            "Context Statistics:",
            "- Passages by type: " + ", ".join(f"{k}: {v}" for k, v in context_data['type_counts'].items()),
            f"- Score Range: min={context_data['score_range']['min']:.2f}, "
            f"max={context_data['score_range']['max']:.2f}, "
            f"avg={context_data['score_range']['avg']:.2f}"
        ]

        # Add conversation history if provided
        if conversation_history:
            prompt_parts.extend([
                "",
                "Previous conversation context:",
                *[f"{turn['role']}: {turn['content']}" for turn in conversation_history[-3:]]
            ])

        prompt_parts.extend([
            "",
            f"Question: {query}",
            "",
            "Provide a comprehensive answer using the above context."
        ])

        return "\n".join(prompt_parts)

    def _format_llm_response(self, response) -> str:
        """
        Format the LLM response for display.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Formatted response string
        """
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                text = '\n'.join(
                    block.text if hasattr(block, 'text') else str(block) 
                    for block in content
                )
            else:
                text = str(content)
        else:
            text = str(response)
            
        return text.replace('\\n', '\n').strip()

    def update_system_prompt(self, prompt_type: str, new_prompt: str) -> bool:
        """
        Update a system prompt.
        
        Args:
            prompt_type: Type of prompt to update ('pure' or 'rag')
            new_prompt: New prompt text
            
        Returns:
            Boolean indicating success
        """
        if prompt_type not in self.system_prompts:
            logger.error(f"Invalid prompt type: {prompt_type}")
            return False
            
        try:
            self.system_prompts[prompt_type] = new_prompt
            logger.info(f"Successfully updated {prompt_type} prompt")
            return True
        except Exception as e:
            logger.error(f"Error updating prompt: {str(e)}")
            return False

    def get_system_prompts(self) -> Dict[str, str]:
        """
        Get current system prompts.
        
        Returns:
            Dictionary of prompt types and their content
        """
        return self.system_prompts.copy()