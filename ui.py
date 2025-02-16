# ui.py
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
from datetime import datetime
from typing import Dict, NamedTuple
from dotenv import load_dotenv

from .config import SearchParameters
from .rag_manager import RAGManager
from .index_workflow import IndexWorkflowManager, IndexConfig
from .utils import (
    SearchResultFormatter, 
    IndexUtils, 
    ValidationUtils,
    TimeUtils
)
from .chunker import ChunkingMode, get_chunking_strategy
from .elastic_config import ElasticsearchConfig
from .inference_utils import (
    InferenceModelManager,
    ModelDeploymentConfig,
    ModelTask,
    get_inference_manager
)
from .inference_menu import InferenceMenu

logger = logging.getLogger(__name__)

class PresetModel(NamedTuple):
    """Preset model configuration."""
    name: str
    hub_id: str
    task_type: str
    description: str

class ElasticRAGInterface:
    def __init__(self, rag_manager: RAGManager):
        """
        Initialize the interface.
        
        Args:
            rag_manager: RAG system manager instance
        """
        self.rag_manager = rag_manager
        self.search_params = SearchParameters()
        self.conversation_history: List[Dict[str, str]] = []
        self.formatter = SearchResultFormatter()
        
        # Initialize inference model manager
        self.inference_manager = get_inference_manager(rag_manager.enhanced_searcher.es)
        
        # Verify utilities are available
        if not all([SearchResultFormatter, IndexUtils, ValidationUtils, TimeUtils]):
            raise ImportError("Required utilities not available")
            
        logger.info("ElasticRAGInterface initialized")

    def verify_index_exists(self, index_name: str) -> bool:
        """
        Verify if an index exists.
        
        Args:
            index_name: Name of the index to check
            
        Returns:
            Boolean indicating if index exists
        """
        try:
            return self.rag_manager.enhanced_searcher.es.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False

    def run(self):
        """Run the main application loop."""
        while True:
            try:
                self._display_menu()
                choice = input("\nSelect an option: ").strip()
                
                if choice == "8":
                    print("Exiting. Goodbye!")
                    break
                    
                self._handle_menu_choice(choice)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"❌ An error occurred: {str(e)}")
                print("Please try again or select a different option.")
            
            input("\nPress Enter to continue...")

    def _display_menu(self):
        """Display the main menu."""
        print("\n" + "=" * 40)
        print("=== Elastic RAG Demo TUI ===")
        print("=" * 40)
        print("1. Ask a question")
        print("2. Configure Search Parameters")
        print("3. Debug Info & Query Explanation")
        print("4. Data Architecture Menu")
        print("5. Show/Change Prompt Strategy")
        print("6. Index or Re-index Data")
        print("7. Manage Inference Models")
        print("8. About")
        print("9. Exit")
        
    def _handle_menu_choice(self, choice: str):
        """Handle menu selection."""
        handlers = {
            "1": self._handle_question,
            "2": self._handle_config,
            "3": self._handle_debug,
            "4": self._handle_architecture,
            "5": self._handle_prompt_strategy,
            "6": self._handle_indexing,
            "7": self._handle_inference,
            "8": self._handle_about
        }
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("Invalid option. Please try again.")
            
    def _handle_indexing(self):
        """Handle data indexing and re-indexing operations."""
        try:
            # Initialize workflow manager
            workflow_manager = IndexWorkflowManager(self.rag_manager.enhanced_searcher.es)
            
            while True:
                print("\n=== Index or Re-index Data ===")
                print("1. Index PDF Document")
                print("2. Index Text File")
                print("3. Re-index from Existing Index")
                print("4. Return to Main Menu")
                
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == "4":
                    break
                    
                if choice == "1":
                    success, message = self._handle_pdf_indexing(workflow_manager)
                elif choice == "2":
                    success, message = self._handle_text_indexing(workflow_manager)
                elif choice == "3":
                    success, message = self._handle_reindexing(workflow_manager)
                else:
                    print("Invalid option. Please try again.")
                    continue
                    
                print(f"\n{'✅' if success else '❌'} {message}")
                
        except Exception as e:
            logger.error(f"Error in indexing handler: {str(e)}")
            print(f"❌ Error: {str(e)}")

    def _handle_pdf_indexing(self, workflow_manager: IndexWorkflowManager) -> Tuple[bool, str]:
        """Handle PDF document indexing workflow."""
        try:
            # Get PDF file path
            pdf_path = input("\nEnter path to PDF file: ").strip()
            if not os.path.exists(pdf_path):
                return False, "File not found."
                
            if not pdf_path.lower().endswith('.pdf'):
                return False, "File must be a PDF."
            
            # Allow user to choose a chunking mode before processing
            self._configure_chunking_mode(workflow_manager)
            
            # Get destination index details
            dest_config = self._get_destination_config()
            if not dest_config:
                return False, "Invalid destination configuration."
                
            # Process PDF
            return workflow_manager.handle_pdf_workflow(pdf_path, dest_config)
            
        except Exception as e:
            logger.error(f"Error in PDF indexing: {str(e)}")
            return False, str(e)

    def _handle_text_indexing(self, workflow_manager: IndexWorkflowManager) -> Tuple[bool, str]:
        """Handle text file indexing workflow."""
        try:
            # Get text file path
            text_path = input("\nEnter path to text file: ").strip()
            if not os.path.exists(text_path):
                return False, "File not found."
            
            # Allow user to choose a chunking mode before processing
            self._configure_chunking_mode(workflow_manager)
            
            # Get destination index details
            dest_config = self._get_destination_config()
            if not dest_config:
                return False, "Invalid destination configuration."
                
            # Process text file
            return workflow_manager.handle_text_workflow(text_path, dest_config)
            
        except Exception as e:
            logger.error(f"Error in text indexing: {str(e)}")
            return False, str(e)

    def _configure_chunking_mode(self, workflow_manager: IndexWorkflowManager):
        """
        Prompt the user to choose a chunking mode (Fast or Thorough)
        and update the chunking configuration in the indexing manager.
        """
        print("\nSelect Chunking Mode for indexing:")
        print("1. Fast")
        print("2. Thorough")
        mode_choice = input("Enter choice (1 or 2): ").strip()
        if mode_choice == "1":
            selected_mode = ChunkingMode.FAST
        else:
            selected_mode = ChunkingMode.THOROUGH
        
        # Update the chunking mode in the indexing manager's configuration
        workflow_manager.indexing_manager.chunking_config.recursive_config.mode = selected_mode
        print(f"Chunking mode set to: {selected_mode.value}")

    def _handle_reindexing(self, workflow_manager: IndexWorkflowManager) -> Tuple[bool, str]:
        """Handle re-indexing workflow."""
        try:
            # Get source index
            source_index = input("\nEnter source index name: ").strip()
            if not self.verify_index_exists(source_index):
                return False, "Source index not found or invalid."
                
            # Get destination index details
            dest_config = self._get_destination_config(source_index)
            if not dest_config:
                return False, "Invalid destination configuration."
                
            # Get mapping strategy
            mapping_selection = self._get_mapping_strategy(workflow_manager, source_index)
            if not mapping_selection:
                return False, "Invalid mapping strategy."
                
            # Process re-indexing
            return workflow_manager.handle_reindex_workflow(
                source_index, 
                dest_config,
                mapping_selection
            )
            
        except Exception as e:
            logger.error(f"Error in re-indexing: {str(e)}")
            return False, str(e)

    def _handle_document_search(self, query: str, dense_index: Optional[str], elser_index: Optional[str]):
        """Handle document search without LLM."""
        print("\nSelect search type:")
        print("1. ELSER Search")
        print("2. Dense Vector Search")
        
        search_type = input("Enter 1 or 2: ").strip()

        try:
            if search_type == "1" and elser_index:
                print("\n=== ELSER Search Results ===")
                print("-" * 80)
                print(f"Using parameters: size={self.search_params.size}, "
                      f"min_score={self.search_params.min_score}, "
                      f"rank_constant={self.search_params.rank_constant}, "
                      f"rank_window_size={self.search_params.rank_window_size}")
                
                results = self.rag_manager.enhanced_searcher.hybrid_search(
                    query=query,
                    index_name=elser_index,
                    params=self.search_params
                )
                
                if results:
                    # Leverage utils for formatting instead of a duplicate method
                    self.formatter.format_search_results(results)
                else:
                    print("⚠️ No relevant documents found.")
                    
            elif search_type == "2" and dense_index:
                print("\n=== Dense Vector Search Results ===")
                print("-" * 80)
                print(f"Using Dense Top K: {self.search_params.dense_top_k}")
                
                results = self.rag_manager.dense_searcher.search(
                    query=query,
                    index_name=dense_index,
                    top_k=self.search_params.dense_top_k
                )
                
                if results:
                    self.formatter.format_search_results(results)
                else:
                    print("⚠️ No relevant documents found.")
                    
            else:
                print("❌ Invalid search type or index not available.")
                
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            print(f"❌ Error: {str(e)}")

    def _get_destination_config(self, source_index: Optional[str] = None) -> Optional[IndexConfig]:
        """Get destination index configuration from user."""
        try:
            dest_name = input("\nEnter destination index name: ").strip()
            if not ValidationUtils.validate_index_name(dest_name):
                print("❌ Invalid index name format.")
                return None
                
            exists = self.verify_index_exists(dest_name)
            
            if exists:
                proceed = input("\nDestination index exists. Overwrite? (y/n): ").strip().lower()
                if proceed != 'y':
                    return None
            else:
                create = input("\nDestination index doesn't exist. Create it? (y/n): ").strip().lower()
                if create != 'y':
                    return None
                
            print("\nSelect mapping strategy:")
            print("1. " + ("Copy from source index" if source_index else "Use default mapping"))
            print("2. Copy from another index")
            print("3. Load from mapping file")
            print("4. Use custom mapping template")
            
            mapping_choice = input("\nSelect option (1-4): ").strip()
            mapping_info = self._get_mapping_info(mapping_choice, source_index)
            if not mapping_info:
                return None
                
            return IndexConfig(
                name=dest_name,
                mapping_type=mapping_info["type"],
                mapping_source=mapping_info["source"],
                exists=exists
            )
            
        except Exception as e:
            logger.error(f"Error getting destination config: {str(e)}")
            return None

    def _get_mapping_info(self, choice: str, source_index: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get mapping information based on user selection."""
        try:
            if choice == "1":
                if source_index:
                    return {"type": "source_copy", "source": source_index}
                else:
                    return {"type": "default", "source": "default_template"}
                    
            elif choice == "2":
                source = input("\nEnter index name to copy mapping from: ").strip()
                if not self.verify_index_exists(source):
                    print("❌ Source index not found.")
                    return None
                return {"type": "index_copy", "source": source}
                    
            elif choice == "3":
                mapping_file = input("\nEnter path to mapping file: ").strip()
                if not os.path.exists(mapping_file):
                    print("❌ Mapping file not found.")
                    return None
                return {"type": "file", "source": mapping_file}
                    
            elif choice == "4":
                print("\nSelect mapping template:")
                print("1. Basic (text + keyword fields)")
                print("2. RAG (dense vectors + ELSER)")
                print("3. Advanced (custom analyzers + pipelines)")
                
                template_choice = input("\nSelect template (1-3): ").strip()
                return {"type": "template", "source": f"template_{template_choice}"}
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting mapping info: {str(e)}")
            return None

    def _get_mapping_strategy(self, workflow_manager: IndexWorkflowManager, source_index: str) -> Optional[Dict[str, Any]]:
        """Get mapping strategy with suggestions."""
        try:
            suggestions = workflow_manager.suggest_mapping_templates(source_index)
            
            if "error" in suggestions:
                print(f"⚠️ Could not generate suggestions: {suggestions['error']}")
            else:
                print("\nRecommended templates:")
                for template in suggestions["recommended"]:
                    print(f"- {template}")
                print(f"\nReason: {suggestions['reason']}")
                
                print("\nAlternative templates:")
                for template in suggestions["alternatives"]:
                    print(f"- {template}")
            
            return self._get_mapping_info(
                input("\nSelect mapping strategy (1-4): ").strip(),
                source_index
            )
            
        except Exception as e:
            logger.error(f"Error getting mapping strategy: {str(e)}")
            return None

    def _handle_question(self):
        """Handle question answering workflow."""
        try:
            base_index = input("\nEnter the base index name to search: ").strip()
            if not ValidationUtils.validate_index_name(base_index):
                print("❌ Invalid index name format.")
                return

            dense_index = f"{base_index}_dense"
            elser_index = f"{base_index}_elser"
            
            dense_exists = self.verify_index_exists(dense_index)
            elser_exists = self.verify_index_exists(elser_index)
            
            if not (dense_exists or elser_exists):
                print(f"❌ No indices found for base name: {base_index}")
                return

            print("\nSelect search strategy:")
            print("1. Pure LLM (No RAG)")
            if elser_exists:
                print("2. LLM+RAG (Hybrid ELSER)")
            if dense_exists:
                print("3. LLM+RAG (Dense)")
            print("4. Document Search Only (No LLM)")
            
            strategy = input("Enter choice: ").strip()
            
            if strategy == "2" and not elser_exists:
                print("❌ ELSER index not available.")
                return
            if strategy == "3" and not dense_exists:
                print("❌ Dense vector index not available.")
                return

            query = input("\nEnter your question: ").strip()
            if not query:
                print("❌ Empty query.")
                return

            print("\n🚀 Processing your question...")
            print(f"Using base index: {base_index}")
            
            start_time = datetime.now()
            self._process_search_strategy(
                strategy, 
                query, 
                dense_index if dense_exists else None,
                elser_index if elser_exists else None
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            print(f"\n⏱ Query processed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in question handling: {str(e)}")
            print(f"❌ Error processing question: {str(e)}")

    def _process_search_strategy(self, strategy: str, query: str, dense_index: Optional[str], elser_index: Optional[str]):
        """Process search based on selected strategy."""
        try:
            if strategy == "1":
                print("\n=== Pure LLM Response ===")
                print("-" * 80)
                response = self.rag_manager.get_pure_llm_response(query)
                print(response)
                self._update_conversation(query, response)

            elif strategy == "2" and elser_index:
                print("\n=== Hybrid ELSER RAG ===")
                print("-" * 80)
                print(f"Parameters: size={self.search_params.size}, min_score={self.search_params.min_score}")
                
                search_results = self.rag_manager.enhanced_searcher.hybrid_search(
                    query=query,
                    index_name=elser_index,
                    params=self.search_params
                )
                
                if not search_results:
                    print("⚠️ No relevant passages found.")
                    return
                    
                print(f"Retrieved {len(search_results)} passages")
                response = self.rag_manager.get_rag_response(
                    query, 
                    search_results,
                    self.conversation_history
                )
                print(response)
                self._update_conversation(query, response)
                self._offer_result_views(search_results, "hybrid")

            elif strategy == "3" and dense_index:
                print("\n=== Dense RAG ===")
                print("-" * 80)
                print(f"Using Dense Top K: {self.search_params.dense_top_k}")
                
                response = self.rag_manager.get_dense_rag_response(
                    query=query,
                    index_name=dense_index,
                    params=self.search_params,
                    conversation_history=self.conversation_history
                )
                print(response)
                self._update_conversation(query, response)
                self._offer_result_views(None, "dense")

            elif strategy == "4":
                self._handle_document_search(query, dense_index, elser_index)
                
        except Exception as e:
            logger.error(f"Error processing strategy: {str(e)}")
            raise

    def _handle_config(self):
        """Handle search parameter configuration."""
        while True:
            print("\n=== Search Parameter Configuration ===")
            print("\nCurrent Parameters:")
            print("\nHybrid ELSER Parameters:")
            print(f"1. Result Size: {self.search_params.size}")
            print(f"2. Minimum Score: {self.search_params.min_score}")
            print(f"3. RRF Rank Constant: {self.search_params.rank_constant}")
            print(f"4. RRF Window Size: {self.search_params.rank_window_size}")
            print("\nDense Search Parameters:")
            print(f"5. Dense Top K: {self.search_params.dense_top_k}")
            print("\n6. Return to Main Menu")
            
            choice = input("\nSelect parameter to modify (1-6): ").strip()
            if choice == "6":
                break
            try:
                self._modify_search_parameter(choice)
            except ValueError as e:
                print(f"❌ {str(e)}")

    def _modify_search_parameter(self, choice: str):
        """Helper to modify search parameters based on user input."""
        if choice == "1":
            new_size = int(input("Enter new result size (1-20): ").strip())
            self.search_params.size = new_size
        elif choice == "2":
            new_min = float(input("Enter new minimum score (0.0-1.0): ").strip())
            self.search_params.min_score = new_min
        elif choice == "3":
            new_const = int(input("Enter new RRF rank constant (1-100): ").strip())
            self.search_params.rank_constant = new_const
        elif choice == "4":
            new_window = int(input("Enter new RRF window size (10-200): ").strip())
            self.search_params.rank_window_size = new_window
        elif choice == "5":
            new_dense = int(input("Enter new Dense Top K (e.g., 5-50): ").strip())
            self.search_params.dense_top_k = new_dense
        else:
            raise ValueError("Invalid parameter selection.")

    def _handle_prompt_strategy(self):
        """Handle prompt strategy display and modification."""
        print("\n=== Current Prompt Strategies ===")
        print("-" * 80)
        prompts = self.rag_manager.get_system_prompts()
        for key, prompt in prompts.items():
            print(f"\n--- {key.upper()} Prompt ---")
            print(prompt)
            
        choice = input("\nWould you like to modify a prompt? (pure/rag/n): ").strip().lower()
        if choice in ["pure", "rag"]:
            print(f"\nCurrent {choice} prompt:")
            print(prompts[choice])
            new_prompt = input("\nEnter new prompt text (or press Enter to cancel): ").strip()
            if new_prompt:
                if self.rag_manager.update_system_prompt(choice, new_prompt):
                    print("✅ Prompt updated successfully")
                else:
                    print("❌ Failed to update prompt")

    def _handle_debug(self):
        """Handle debug information display."""
        print("\n=== Debug Information ===")
        print("-" * 80)
        print("\n🔍 Search Parameters:")
        for key, value in self.search_params.to_dict().items():
            print(f"- {key}: {value}")
        
        print("\n🔍 Connection Test:")
        try:
            info = self.rag_manager.enhanced_searcher.es.info()
            print("✅ Connected to Elasticsearch")
            print(f"- Version: {info['version']['number']}")
            print(f"- Cluster Name: {info['cluster_name']}")
            
            indices = self.rag_manager.enhanced_searcher.es.cat.indices(format="json")
            print(f"\n🔍 Found {len(indices)} indices:")
            recent_indices = sorted(indices, key=lambda x: x.get('creation.date', '0'), reverse=True)[:5]
            for idx in recent_indices:
                name = idx.get('index')
                docs = idx.get('docs.count', 'N/A')
                size = idx.get('store.size', 'N/A')
                print(f"- {name}: docs={docs}, size={size}")
                
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")

    def _handle_architecture(self):
        """Handle data architecture menu."""
        try:
            print("\n=== Data Architecture Menu ===")
            print("1. List All Indices")
            print("2. Show Index Mapping")
            print("3. List Ingest Pipelines")
            print("4. Search for Index")
            print("5. List Top Indices by Size")
            print("6. List Recent Indices")
            print("7. Return to Main Menu")
            
            choice = input("\nSelect an option (1-7): ").strip()
            if choice == "7":
                return
                
            self._handle_architecture_choice(choice)
                
        except Exception as e:
            logger.error(f"Error in architecture handler: {str(e)}")
            print(f"❌ Error: {str(e)}")

    def _handle_architecture_choice(self, choice: str):
        """Handle detailed actions for the Data Architecture menu."""
        try:
            es = self.rag_manager.enhanced_searcher.es
            if choice == "1":
                indices = es.cat.indices(format="json")
                print(IndexUtils.format_indices_table(indices))
            elif choice == "2":
                index_name = input("Enter index name: ").strip()
                try:
                    mapping = es.indices.get_mapping(index=index_name)
                    settings = es.indices.get_settings(index=index_name)
                    print(self.formatter.format_mapping_info(mapping, settings))
                except Exception as e:
                    print(f"Error fetching mapping: {str(e)}")
            elif choice == "3":
                pipelines = es.ingest.get_pipeline()
                print(self.formatter.format_pipeline_list(pipelines))
            elif choice == "4":
                search_term = input("Enter search term for index name: ").strip()
                indices = es.cat.indices(format="json")
                filtered = [idx for idx in indices if search_term in idx.get('index', '')]
                print(IndexUtils.format_indices_table(filtered))
            elif choice == "5":
                indices = es.cat.indices(format="json")
                print(IndexUtils.format_indices_table(indices, sort_by='size'))
            elif choice == "6":
                indices = es.cat.indices(format="json")
                recent_indices = sorted(indices, key=lambda x: int(x.get('creation.date', 0)), reverse=True)[:5]
                print(IndexUtils.format_indices_table(recent_indices))
            else:
                print("Invalid option.")
        except Exception as e:
            logger.error(f"Error handling architecture choice: {str(e)}")
            print(f"❌ Error: {str(e)}")

    def _update_conversation(self, query: str, response: str):
        """Update conversation history."""
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _offer_result_views(self, search_results: Optional[List[Dict[str, Any]]], search_type: str):
        """Offer to show search results and query details."""
        if search_results and input("\nView source documents? (y/n): ").lower().strip() == 'y':
            self.formatter.format_search_results(search_results)
            
        if input("\nView search query details? (y/n): ").lower().strip() == 'y':
            query_explanation = self.formatter.format_query_explanation(
                search_type,
                self.conversation_history[-2]["content"],  # Last query
                self.search_params
            )
            print(query_explanation)

    def _handle_inference(self):
        """Handle inference model management."""
        inference_menu = InferenceMenu(self.inference_manager)
        inference_menu.show_menu()
        
    def _handle_about(self):
        """Display about information."""
        about_text = """
About This Application
------------------------

This application is a hybrid Retrieval-Augmented Generation (RAG) system that combines 
multiple advanced techniques to deliver context-aware responses from Elasticsearch-loaded 
documentation.

Key Features:
1. Multiple Search Strategies
   - Dense vector search using Instructor-XL embeddings
   - ELSER semantic search
   - Hybrid search with RRF fusion

2. Advanced RAG Implementation
   - Context-aware prompt construction
   - Conversation history support
   - Content type detection and boosting

3. Flexible Configuration
   - Adjustable search parameters
   - Customizable system prompts
   - Multiple indexing strategies

4. Data Management
   - PDF and text file indexing
   - Re-indexing capabilities
   - Flexible mapping management

For support or contributions:
https://github.com/datadingbat/rag-tools
"""
        print(about_text)

if __name__ == "__main__":
    # Initialize Elasticsearch client first
    es_config = ElasticsearchConfig()
    es_client = es_config.get_client()
    
    # Initialize RAG manager with the client
    rag_manager = RAGManager(es_client)
    
    # Initialize and run the interface
    interface = ElasticRAGInterface(rag_manager)
    interface.run()