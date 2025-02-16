# inference_menu.py
from typing import Dict, NamedTuple, Optional, List
import logging
import os
from dotenv import load_dotenv
from .inference_utils import InferenceModelManager, ModelDeploymentConfig, ModelTask

logger = logging.getLogger(__name__)

class PresetModel(NamedTuple):
    """Preset model configuration."""
    name: str
    hub_id: str
    task_type: str
    description: str

class InferenceMenu:
    """Handler for inference model management menu options."""
    
    PRESET_MODELS = {
        "ms-marco": PresetModel(
            name="MS MARCO MiniLM",
            hub_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            task_type="text_similarity",
            description="Efficient model for semantic search and text similarity"
        ),
        "bert-ner": PresetModel(
            name="DistilBERT NER",
            hub_id="elastic/distilbert-base-cased-finetuned-conll03-english",
            task_type="ner",
            description="Named Entity Recognition model fine-tuned on CoNLL-2003"
        ),
        "mpnet": PresetModel(
            name="All-MPNet-Base",
            hub_id="sentence-transformers/all-mpnet-base-v2",
            task_type="text_embedding",
            description="Powerful model for text embeddings and similarity"
        ),
        "e5-small": PresetModel(
            name="E5 Small",
            hub_id="intfloat/multilingual-e5-small",
            task_type="text_embedding",
            description="Efficient multilingual text embedding model"
        ),
        "bert-qa": PresetModel(
            name="BERT Q&A",
            hub_id="deepset/bert-base-cased-squad2",
            task_type="question_answering",
            description="Question answering model fine-tuned on SQuAD2"
        )
    }

    def __init__(self, inference_manager: InferenceModelManager):
        self.inference_manager = inference_manager

    def show_menu(self):
        while True:
            print("\n=== Inference Model Management ===")
            print("1. List Available Models")
            print("2. List Deployed Models")
            print("3. Deploy Model")
            print("4. Undeploy Model")
            print("5. Import Model")
            print("6. Create Inference Pipeline")
            print("7. Return to Main Menu")
            
            choice = input("\nSelect option (1-7): ").strip()
            if choice == "7":
                break
            try:
                if choice == "1":
                    self._list_available_models()
                elif choice == "2":
                    self._list_deployed_models()
                elif choice == "3":
                    self._deploy_model()
                elif choice == "4":
                    self._undeploy_model()
                elif choice == "5":
                    self._import_model()
                elif choice == "6":
                    self._create_inference_pipeline()
                else:
                    print("Invalid option. Please try again.")
            except Exception as e:
                logger.error(f"Error in inference management: {str(e)}")
                print(f"❌ Error: {str(e)}")

    def _list_available_models(self):
        """List all available models."""
        try:
            models = self.inference_manager.list_deployable_models()
            
            if not models:
                print("No deployable models found.")
                return
                
            print("\nAvailable Models:")
            print(f"{'Model ID':<40} {'Task Type':<20} {'Version':<10}")
            print("-" * 70)
            
            for model in models:
                model_id = model.get('model_id', 'N/A')
                task_type = model.get('task_type', 'N/A')
                version = model.get('version', 'N/A')
                print(f"{model_id:<40} {task_type:<20} {str(version):<10}")
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise

    def _list_deployed_models(self):
        """List currently deployed models."""
        try:
            models = self.inference_manager.list_deployed_models()
            
            if not models:
                print("No models currently deployed.")
                return
                
            print("\nDeployed Models:")
            print(f"{'Model ID':<40} {'State':<15} {'Inference Count':<15}")
            print("-" * 70)
            
            for model in models:
                print(f"{model['model_id']:<40} {model['state']:<15} {model['inference_count']:<15}")
                
        except Exception as e:
            logger.error(f"Error listing deployed models: {str(e)}")
            raise

    def _deploy_model(self):
        """Deploy a model for inference."""
        try:
            models = self.inference_manager.list_deployable_models()
            if not models:
                print("No deployable models available.")
                return
            print("\nAvailable Models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['model_id']} ({model['task_type']})")
            selection = input("\nSelect model number to deploy: ").strip()
            try:
                idx = int(selection) - 1
                if not (0 <= idx < len(models)):
                    raise ValueError("Invalid selection")
                model = models[idx]
            except (ValueError, IndexError):
                print("Invalid selection.")
                return

            # Ask whether to use adaptive allocations.
            adaptive = input("Enable adaptive allocations? (y/n) [default=n]: ").strip().lower() == "y"
            if adaptive:
                min_alloc = input("Enter minimum number of allocations: ").strip()
                max_alloc = input("Enter maximum number of allocations: ").strip()
                try:
                    min_alloc = int(min_alloc)
                    max_alloc = int(max_alloc)
                except ValueError:
                    print("Invalid numbers provided for adaptive allocations.")
                    return
                adaptive_allocations = {
                    "enabled": True,
                    "min_number_of_allocations": min_alloc,
                    "max_number_of_allocations": max_alloc
                }
                config = ModelDeploymentConfig(
                    model_id=model['model_id'],
                    adaptive_allocations=adaptive_allocations,
                    priority="normal",
                    timeout="30s"
                )
            else:
                num_allocations = input("Enter number of allocations [default=1]: ").strip()
                num_allocations = int(num_allocations) if num_allocations else 1
                num_threads = input("Enter number of threads [default=1]: ").strip()
                num_threads = int(num_threads) if num_threads else 1
                config = ModelDeploymentConfig(
                    model_id=model['model_id'],
                    num_allocations=num_allocations,
                    num_threads=num_threads,
                    priority="normal",
                    timeout="30s"
                )

            if self.inference_manager.deploy_model(config):
                print(f"✅ Successfully deployed model {model['model_id']}")
            else:
                print(f"❌ Failed to deploy model {model['model_id']}")
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise


    def _undeploy_model(self):
        """Undeploy a model."""
        try:
            # Get deployed models
            models = self.inference_manager.list_deployed_models()
            if not models:
                print("No models currently deployed.")
                return
                
            # Display deployed models
            print("\nDeployed Models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['model_id']} ({model['state']})")
                
            # Get model selection
            selection = input("\nSelect model number to undeploy: ").strip()
            try:
                idx = int(selection) - 1
                if not (0 <= idx < len(models)):
                    raise ValueError("Invalid selection")
                model = models[idx]
            except (ValueError, IndexError):
                print("Invalid selection.")
                return
                
            # Confirm undeployment
            confirm = input(f"Are you sure you want to undeploy {model['model_id']}? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Operation cancelled.")
                return
                
            # Undeploy model
            if self.inference_manager.undeploy_model(model['model_id']):
                print(f"✅ Successfully undeployed model {model['model_id']}")
            else:
                print(f"❌ Failed to undeploy model {model['model_id']}")
                
        except Exception as e:
            logger.error(f"Error undeploying model: {str(e)}")
            raise

    def _import_model(self):
        """Handle model import options."""
        try:
            print("\nImport Model Options:")
            print("1. Import from Hugging Face Hub")
            print("2. Import from Local Files")
            print("3. Import E5 Model (Air-gapped)")
            print("4. Cancel")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "4":
                return
                
            if choice == "1":
                self._handle_huggingface_import()
            elif choice == "2":
                self._handle_local_import()
            elif choice == "3":
                self._handle_e5_import()
            
        except Exception as e:
            logger.error(f"Error importing model: {str(e)}")
            raise

    def _handle_huggingface_import(self):
        """Handle HuggingFace model import with presets and flexible authentication."""
        try:
            print("\nHuggingFace Model Import")
            print("\nSelect Model Source:")
            print("1. Choose from preset models")
            print("2. Specify custom model")
            print("3. Cancel")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "3":
                return
                
            # Get model details
            if choice == "1":
                model_info = self._select_preset_model()
                if not model_info:
                    return
                hub_id = model_info.hub_id
                task_type = model_info.task_type
            else:
                hub_id = input("\nEnter HuggingFace model ID: ").strip()
                task_type = self._select_task_type()
                if not task_type:
                    return
                    
            # Get authentication method
            auth_method = self._get_auth_method()
            if not auth_method:
                return
                
            # Get deployment configuration
            es_model_id = input("\nEnter Elasticsearch model ID [optional]: ").strip()
            
            # Additional options
            print("\nAdditional Options:")
            clear_previous = input("Clear previous model? (y/N): ").strip().lower() == 'y'
            start_deployment = input("Start deployment after import? (y/N): ").strip().lower() == 'y'
            
            # Import model
            if self.inference_manager.import_huggingface_model(
                hub_model_id=hub_id,
                task_type=task_type,
                es_model_id=es_model_id or None,
                clear_previous=clear_previous,
                start=start_deployment,
                **auth_method
            ):
                print(f"\n✅ Successfully imported model: {hub_id}")
                
                # If start wasn't specified, offer deployment
                if not start_deployment:
                    if input("\nDeploy model now? (y/N): ").strip().lower() == 'y':
                        self._deploy_imported_model(es_model_id or hub_id.split('/')[-1])
            else:
                print(f"\n❌ Failed to import model: {hub_id}")
                
        except Exception as e:
            logger.error(f"Error in HuggingFace import: {str(e)}")
            raise

    def _handle_local_import(self):
        """Handle local file import."""
        try:
            model_path = input("Enter path to model files: ").strip()
            es_id = input("Enter Elasticsearch model ID: ").strip()
            
            task_type = self._select_task_type()
            if not task_type:
                return
                
            if self.inference_manager.import_local_model(
                model_path=model_path,
                task_type=task_type,
                es_model_id=es_id
            ):
                print("✅ Successfully imported local model")
                if input("\nDeploy model now? (y/N): ").strip().lower() == 'y':
                    self._deploy_imported_model(es_id)
            else:
                print("❌ Failed to import model")
                
        except Exception as e:
            logger.error(f"Error in local import: {str(e)}")
            raise

    def _handle_e5_import(self):
        """Handle E5 model import for air-gapped environments."""
        try:
            print("\nE5 Model Import Options:")
            print("1. File-based Deployment")
            print("2. Local HuggingFace Clone")
            print("3. Cancel")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "3":
                return
                
            if choice == "1":
                self._handle_e5_file_based()
            elif choice == "2":
                self._handle_e5_local_huggingface()
                
        except Exception as e:
            logger.error(f"Error in E5 import: {str(e)}")
            raise

    def _select_preset_model(self) -> Optional[PresetModel]:
        """Display and select from preset models."""
        try:
            print("\nAvailable Preset Models:")
            for i, (key, model) in enumerate(self.PRESET_MODELS.items(), 1):
                print(f"\n{i}. {model.name}")
                print(f"   Model ID: {model.hub_id}")
                print(f"   Task: {model.task_type}")
                print(f"   Description: {model.description}")
                
            choice = input("\nSelect model number: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.PRESET_MODELS):
                    return list(self.PRESET_MODELS.values())[idx]
            except ValueError:
                pass
                
            print("Invalid selection.")
            return None
            
        except Exception as e:
            logger.error(f"Error selecting preset model: {str(e)}")
            raise

    def _select_task_type(self) -> Optional[str]:
        """Select NLP task type."""
        try:
            print("\nAvailable Task Types:")
            tasks = [
                ("fill_mask", "Masked language modeling"),
                ("ner", "Named Entity Recognition"),
                ("question_answering", "Question Answering"),
                ("text_classification", "Text Classification"),
                ("text_embedding", "Text Embeddings"),
                ("text_expansion", "Text Expansion"),
                ("text_similarity", "Text Similarity"),
                ("zero_shot_classification", "Zero-shot Classification")
            ]
            
            for i, (task, desc) in enumerate(tasks, 1):
                print(f"{i}. {task} - {desc}")
                
            choice = input("\nSelect task type number: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(tasks):
                    return tasks[idx][0]
            except ValueError:
                pass
                
            print("Invalid selection.")
            return None
            
        except Exception as e:
            logger.error(f"Error selecting task type: {str(e)}")
            raise

    def _get_auth_method(self) -> Optional[Dict]:
        """Get authentication method and credentials."""
        try:
            print("\nAuthentication Method:")
            print("1. Use .env configuration")
            print("2. Username/Password")
            print("3. API Key")
            print("4. Cancel")
            
            choice = input("\nSelect auth method (1-4): ").strip()
            
            if choice == "4":
                return None
                
            if choice == "1":
                # Load from .env
                load_dotenv()
                
                # Check for cloud ID first
                cloud_id = os.getenv('ELASTIC_CLOUD_ID')
                if cloud_id:
                    api_key = os.getenv('ELASTIC_API_KEY')
                    if api_key:
                        return {
                            "cloud_id": cloud_id,
                            "api_key": api_key
                        }
                    
                # Fall back to basic auth
                url = os.getenv('ELASTIC_URL')
                username = os.getenv('ELASTIC_USERNAME')
                password = os.getenv('ELASTIC_PASSWORD')
                
                if all([url, username, password]):
                    return {
                        "url": url,
                        "username": username,
                        "password": password
                    }
                    
                print("❌ Required environment variables not found in .env")
                return None
                
            elif choice == "2":
                url = input("Enter Elasticsearch URL: ").strip()
                username = input("Enter username: ").strip()
                password = input("Enter password: ").strip()
                
                if all([url, username, password]):
                    return {
                        "url": url,
                        "username": username,
                        "password": password
                    }
                    
            elif choice == "3":
                url = input("Enter Elasticsearch URL: ").strip()
                api_key = input("Enter API key: ").strip()
                
                if url and api_key:
                    return {
                        "url": url,
                        "api_key": api_key
                    }
                    
            print("❌ Invalid or incomplete authentication information")
            return None
            
        except Exception as e:
            logger.error(f"Error getting authentication method: {str(e)}")
            raise

    def _deploy_imported_model(self, model_id: str):
        """Deploy a newly imported model."""
        try:
            print("\nDeployment Configuration:")
            priority = input("Enter priority (low/normal/high) [default=normal]: ").strip() or "normal"
            
            num_allocations = input("Enter number of allocations [default=1]: ").strip()
            num_allocations = int(num_allocations) if num_allocations else 1
            
            num_threads = input("Enter threads per allocation [default=1]: ").strip()
            num_threads = int(num_threads) if num_threads else 1
            
            config = ModelDeploymentConfig(
                model_id=model_id,
                num_allocations=num_allocations,
                num_threads=num_threads,
                priority=priority
            )
            
            if self.inference_manager.deploy_model(config):
                print(f"✅ Successfully deployed model {model_id}")
            else:
                print(f"❌ Failed to deploy model {model_id}")
                
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise

    def _create_inference_pipeline(self):
        """Create an inference pipeline."""
        try:
            # Get deployed models
            models = self.inference_manager.list_deployed_models()
            if not models:
                print("No models currently deployed. Please deploy a model first.")
                return
                
            # Display deployed models
            print("\nDeployed Models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['model_id']} ({model['state']})")
                
            # Get model selection
            selection = input("\nSelect model number for pipeline: ").strip()
            try:
                idx = int(selection) - 1
                if not (0 <= idx < len(models)):
                    raise ValueError("Invalid selection")
                model = models[idx]
            except (ValueError, IndexError):
                print("Invalid selection.")
                return
                
            # Get pipeline configuration
            pipeline_id = input("Enter pipeline ID: ").strip()
            source_field = input("Enter source field name: ").strip()
            target_field = input("Enter target field name: ").strip()
            
            # Create pipeline
            if self.inference_manager.create_inference_pipeline(
                pipeline_id=pipeline_id,
                model_id=model['model_id'],
                source_field=source_field,
                target_field=target_field
            ):
                print(f"✅ Successfully created inference pipeline {pipeline_id}")
            else:
                print(f"❌ Failed to create inference pipeline")
                
        except Exception as e:
            logger.error(f"Error creating inference pipeline: {str(e)}")
            raise

    def _handle_e5_file_based(self):
        """Handle file-based E5 model deployment."""
        try:
            print("\nE5 Model Versions:")
            print("1. multilingual-e5-small")
            print("2. multilingual-e5-small-linux-x86-64 (optimized)")
            
            version = input("\nSelect version (1-2): ").strip()
            model_id = "multilingual-e5-small"
            if version == "2":
                model_id += "_linux-x86_64"
                
            # Get model directory
            model_dir = input("\nEnter path to model files directory: ").strip()
            
            # Verify required files
            required_files = [
                f"{model_id}.metadata.json",
                f"{model_id}.pt",
                f"{model_id}.vocab.json"
            ]
            
            missing_files = []
            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                print("\n❌ Missing required files:")
                for file in missing_files:
                    print(f"  - {file}")
                print("\nRequired file URLs:")
                base_url = "https://ml-models.elastic.co"
                for file in required_files:
                    print(f"  {base_url}/{file}")
                return
                
            print("\nℹ️ Configuration Instructions:")
            print("1. Place model files in config/models/ directory on all master-eligible nodes")
            print("2. Add to elasticsearch.yml:")
            print('   xpack.ml.model_repository: file://${path.home}/config/models/')
            print("3. Restart master-eligible nodes one by one")
            
            # Get deployment configuration
            if input("\nProceed with import? (y/N): ").strip().lower() != 'y':
                return
                
            # Import model
            if self.inference_manager.import_model_airgapped(
                model_id=model_id,
                model_path=model_dir,
                task_type=ModelTask.TEXT_EMBEDDING,
                metadata_path=os.path.join(model_dir, f"{model_id}.metadata.json")
            ):
                print("✅ Model imported successfully")
                
                # Offer deployment
                if input("\nDeploy model now? (y/N): ").strip().lower() == 'y':
                    self._deploy_imported_model(model_id)
            else:
                print("❌ Failed to import model")
                
        except Exception as e:
            logger.error(f"Error in file-based E5 deployment: {str(e)}")
            raise

    def _handle_e5_local_huggingface(self):
        """Handle E5 model deployment from local HuggingFace clone."""
        try:
            # Get local clone path
            clone_path = input("\nEnter path to cloned multilingual-e5-small directory: ").strip()
            
            if not os.path.exists(clone_path):
                print("\n❌ Directory not found. Please clone the model first:")
                print("git clone https://huggingface.co/intfloat/multilingual-e5-small")
                return
                
            # Import using eland
            if self.inference_manager.import_local_model(
                model_path=clone_path,
                task_type=ModelTask.TEXT_EMBEDDING,
                es_model_id="multilingual-e5-small"
            ):
                print("✅ Model imported successfully")
                
                # Offer deployment
                if input("\nDeploy model now? (y/N): ").strip().lower() == 'y':
                    self._deploy_imported_model("multilingual-e5-small")
            else:
                print("❌ Failed to import model")
                
        except Exception as e:
            logger.error(f"Error in local HuggingFace E5 deployment: {str(e)}")
            raise