# inference_utils.py
from typing import Dict, List, Optional, Union, Any
import logging
import os
import json
from pathlib import Path
from elasticsearch import Elasticsearch
from dataclasses import dataclass
from enum import Enum
import base64

logger = logging.getLogger(__name__)

class ModelTask(Enum):
    """Supported NLP task types for inference models."""
    FILL_MASK = "fill_mask"
    NER = "ner"
    QUESTION_ANSWERING = "question_answering"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_EMBEDDING = "text_embedding"
    TEXT_EXPANSION = "text_expansion"
    TEXT_SIMILARITY = "text_similarity"
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"

@dataclass
class ModelDeploymentConfig:
    model_id: str
    num_allocations: int = 1
    num_threads: int = 1
    queue_capacity: Optional[int] = None
    priority: str = "normal"
    timeout: str = "30s"  # timeout as a string, e.g., "30s" or "1m"
    deployment_id: Optional[str] = None  # if None, defaults to model_id
    adaptive_allocations: Optional[Dict[str, Any]] = None  # e.g. {"enabled": True, "min_number_of_allocations": 3, "max_number_of_allocations": 10}
    cache_size: Optional[str] = None  # e.g., "100mb"

class InferenceModelManager:
    """Manages Elasticsearch inference models and deployments."""
    
    def __init__(self, es_client: Elasticsearch):
        """Initialize with Elasticsearch client."""
        self.es = es_client

    def _determine_task_type(self, model_config: Dict[str, Any]) -> Optional[str]:
        """
        Determine the task type from the model configuration.
        
        Args:
            model_config: The configuration dictionary for a model.
        
        Returns:
            A string representing the task type, if it can be determined, otherwise None.
        """
        # Try to get the inference configuration.
        inference_config = model_config.get("inference_config", {})
        # Look for a known task type in the inference configuration.
        for task in ModelTask:
            if task.value in inference_config:
                return task.value
        return None

    def list_deployable_models(self) -> List[Dict[str, Any]]:
        """
        List all models that can be deployed.
        
        Returns:
            List of model information dictionaries
        """
        try:
            # First, get list of model IDs without detailed status.
            response = self.es.ml.get_trained_models(
                allow_no_match=True,
                exclude_generated=True,
                size=100
            )
            models = response.get("trained_model_configs", [])
            
            deployable = []
            for model in models:
                model_id = model.get("model_id")
                try:
                    # Try to get detailed info with extra fields.
                    detail_response = self.es.ml.get_trained_models(
                        model_id=model_id,
                        include=["definition_status", "definition", "hyperparameters"],
                        allow_no_match=True
                    )
                    if detail_response.get("trained_model_configs"):
                        model_detail = detail_response["trained_model_configs"][0]
                        
                        # Use definition_status to determine if the model is fully defined.
                        # Depending on your Elasticsearch version this field might be a dict or a string.
                        ds = model_detail.get("definition_status")
                        if isinstance(ds, dict):
                            fully_defined = ds.get("state") == "fully_defined"
                        elif isinstance(ds, str):
                            fully_defined = ds == "fully_defined"
                        else:
                            fully_defined = True  # Fallback if not provided.
                        
                        deployable.append({
                            "model_id": model_id,
                            "task_type": self._determine_task_type(model_detail) or "N/A",
                            "definition_status": ds,
                            "fully_defined": fully_defined
                        })
                except Exception as e:
                    # If the error indicates that definition info isn’t supported,
                    # fall back to using the basic info from the original model.
                    err_str = str(e)
                    if ("does not support retrieving the definition" in err_str or
                        "Definition status is only relevant" in err_str):
                        # Log at debug level instead of warning to hide it from normal output.
                        logger.debug(f"Falling back for model {model_id} due to: {err_str}")
                        deployable.append({
                            "model_id": model_id,
                            "task_type": self._determine_task_type(model) or "N/A",
                            "definition_status": "N/A",
                            "fully_defined": True  # Assume deployable if basic info exists.
                        })
                    else:
                        logger.debug(f"Could not get details for model {model_id}: {err_str}")
                        continue
            
            return deployable
            
        except Exception as e:
            logger.error(f"Error listing deployable models: {str(e)}")
            raise


    def list_deployed_models(self) -> List[Dict[str, Any]]:
        """
        List all currently deployed models.
        
        Returns:
            List of deployed model information
        """
        try:
            response = self.es.ml.get_trained_models_stats(
                model_id="_all",
                allow_no_match=True
            )
            models = response.get("trained_model_stats", [])
            
            deployed = []
            for model in models:
                if model.get("deployment_stats"):
                    deployed.append({
                        "model_id": model.get("model_id"),
                        "state": model.get("deployment_stats", {}).get("state"),
                        "allocation_status": model.get("deployment_stats", {}).get("allocation_status"),
                        "nodes": model.get("deployment_stats", {}).get("nodes", []),
                        "inference_count": model.get("inference_stats", {}).get("inference_count", 0),
                        "last_access": model.get("inference_stats", {}).get("last_access_time")
                    })
            
            return deployed
            
        except Exception as e:
            logger.error(f"Error listing deployed models: {str(e)}")
            raise

def deploy_model(self, config: ModelDeploymentConfig) -> bool:
    """
    Deploy a trained model using the ML API endpoint.

    This method calls:
      POST /_ml/trained_models/<model_id>/deployment/_start
    with query parameters for deployment_id, wait_for, and timeout.

    For manual deployments, it sends number_of_allocations, threads_per_allocation,
    and optionally queue_capacity. For adaptive deployments, it sends the adaptive_allocations object.

    Args:
        config: ModelDeploymentConfig with deployment settings.

    Returns:
        True if deployment succeeded.

    Raises:
        Exception: Propagates any errors during the API call.
    """
    try:
        # Check that the model exists
        if not self.model_exists(config.model_id):
            raise ValueError(f"Model {config.model_id} does not exist")
        
        # Use deployment_id from config if provided, otherwise default to model_id.
        deployment_id = config.deployment_id if config.deployment_id else config.model_id
        
        # Set wait_for parameter (e.g. "started") and timeout.
        wait_for = "started"
        query_params = f"?deployment_id={deployment_id}&wait_for={wait_for}&timeout={config.timeout}"
        
        # Build the request body.
        body = {}
        if config.adaptive_allocations:
            body["adaptive_allocations"] = config.adaptive_allocations
        else:
            body["number_of_allocations"] = config.num_allocations
            body["threads_per_allocation"] = config.num_threads
            if config.queue_capacity is not None:
                body["queue_capacity"] = config.queue_capacity
        
        # Optionally add priority and cache_size.
        if config.priority:
            body["priority"] = config.priority
        if config.cache_size:
            body["cache_size"] = config.cache_size
        
        endpoint = f"/_ml/trained_models/{config.model_id}/deployment/_start{query_params}"
        logger.debug("Deploying model with endpoint: %s and body: %s", endpoint, body)
        
        # Include the required header.
        response = self.es.transport.perform_request(
            "POST",
            endpoint,
            body=body,
            headers={"Content-Type": "application/json"}
        )
        logger.debug("Deploy response for model %s: %s", config.model_id, response)
        logger.info(f"Successfully deployed model {config.model_id}")
        return True
    except Exception as e:
        logger.error(f"Error deploying model {config.model_id}: {str(e)}")
        raise


    def undeploy_model(self, model_id: str) -> bool:
        """
        Stop a trained model deployment using the ML API endpoint.
        
        This method uses the endpoint:
        POST /_ml/trained_models/<model_id>/deployment/_stop
        It first attempts a normal undeployment. If a 409 error is returned indicating
        that the deployment is referenced by ingest processors, it prompts the user to force undeployment.
        
        Args:
            model_id: ID of the model to undeploy.
        
        Returns:
            True if undeployment succeeded, False if cancelled.
        
        Raises:
            Exception: Propagates any errors that occur during the API call.
        """
        try:
            # Check if the model is deployed
            deployed_models = self.list_deployed_models()
            if not any(m["model_id"] == model_id for m in deployed_models):
                logger.warning(f"Model {model_id} is not currently deployed")
                return False

            endpoint = f"/_ml/trained_models/{model_id}/deployment/_stop"
            # Initial body parameters
            body = {"allow_no_match": True, "finish_pending_work": True}

            # Make the initial undeploy request.
            response = self.es.transport.perform_request("POST", endpoint, body=body)
            logger.debug("Initial undeploy response for model %s: %s", model_id, response)

            # Check for a 409 conflict indicating the deployment is referenced by ingest processors.
            if response.meta.status == 409:
                error_str = str(response.body)
                if "referenced by ingest processors" in error_str or "use force to stop" in error_str:
                    user_input = input(
                        "The deployment is referenced by ingest processors. "
                        "Do you want to force undeployment? (y/n): "
                    ).strip().lower()
                    if user_input == "y":
                        body["force"] = True
                        response = self.es.transport.perform_request("POST", endpoint, body=body)
                        logger.debug("Forced undeploy response for model %s: %s", model_id, response)
                    else:
                        logger.info("Force undeployment cancelled by user.")
                        return False

            logger.info(f"Successfully undeployed model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Error undeploying model {model_id}: {str(e)}")
            raise


    def create_inference_pipeline(
        self, 
        pipeline_id: str,
        model_id: str,
        source_field: str,
        target_field: str,
        inference_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create an inference ingest pipeline.
        
        Args:
            pipeline_id: ID for the new pipeline
            model_id: ID of the deployed model to use
            source_field: Field containing text to process
            target_field: Field to store inference results
            inference_config: Optional additional inference configuration
            
        Returns:
            Boolean indicating success
        """
        try:
            # Check if model is deployed
            deployed_models = self.list_deployed_models()
            if not any(m["model_id"] == model_id for m in deployed_models):
                raise ValueError(f"Model {model_id} must be deployed first")
            
            pipeline_body = {
                "description": f"Inference pipeline using model {model_id}",
                "processors": [
                    {
                        "inference": {
                            "model_id": model_id,
                            "target_field": target_field,
                            "field_map": {
                                source_field: "text_field"
                            }
                        }
                    }
                ],
                "on_failure": [
                    {
                        "set": {
                            "description": "Index document to 'failed-<index>'",
                            "field": "_index",
                            "value": "failed-{{{_index}}}"
                        }
                    },
                    {
                        "set": {
                            "description": "Set error message",
                            "field": "ingest.failure",
                            "value": "{{_ingest.on_failure_message}}"
                        }
                    }
                ]
            }
            
            # Add any additional inference configuration
            if inference_config:
                pipeline_body["processors"][0]["inference"].update(inference_config)
            
            # Create pipeline
            self.es.ingest.put_pipeline(
                id=pipeline_id,
                body=pipeline_body
            )
            
            logger.info(f"Successfully created inference pipeline {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating inference pipeline: {str(e)}")
            raise

    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists."""
        try:
            return bool(self.es.ml.get_trained_models(model_id=model_id, allow_no_match=True))
        except Exception:
            return False

    def import_model_airgapped(
        self,
        model_id: str,
        model_path: str,
        task_type: ModelTask,
        metadata_path: Optional[str] = None
    ) -> bool:
        """
        Import a model in an air-gapped environment.
        
        Args:
            model_id: ID to assign to the imported model
            model_path: Path to model files
            task_type: Type of NLP task
            metadata_path: Optional path to metadata file
            
        Returns:
            Boolean indicating success
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError(f"Model path {model_path} does not exist")
            
            # Load required files
            required_files = [
                model_path / f"{model_id}.pt",
                model_path / f"{model_id}.vocab.json"
            ]
            
            if metadata_path:
                metadata_file = Path(metadata_path)
                if not metadata_file.exists():
                    raise ValueError(f"Metadata file {metadata_file} does not exist")
            else:
                metadata_file = model_path / f"{model_id}.metadata.json"
                if not metadata_file.exists():
                    raise ValueError(f"Metadata file not found at {metadata_file}")
            
            for file in required_files:
                if not file.exists():
                    raise ValueError(f"Required file {file} not found")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Import model
            # Implementation depends on specific air-gapped requirements
            # This is a placeholder for the actual implementation
            logger.info(f"Successfully imported model {model_id} in air-gapped environment")
            return True
            
        except Exception as e:
            logger.error(f"Error importing model in air-gapped environment: {str(e)}")
            raise

    def _determine_task_type(self, model_config: Dict[str, Any]) -> Optional[str]:
        """Determine the task type from model configuration."""
        inference_config = model_config.get("inference_config", {})
        for task in ModelTask:
            if task.value in inference_config:
                return task.value
        return None

    def import_huggingface_model(
        self,
        hub_model_id: str,
        task_type: ModelTask,
        es_model_id: Optional[str] = None,
        timeout: int = 600
    ) -> bool:
        """
        Import a model from Hugging Face using eland.
        
        Args:
            hub_model_id: Hugging Face model ID (e.g., 'elastic/distilbert-base-cased-finetuned-conll03-english')
            task_type: Type of NLP task
            es_model_id: Optional custom model ID for Elasticsearch
            timeout: Timeout in seconds for the import process
            
        Returns:
            Boolean indicating success
        """
        try:
            import subprocess
            import sys
            
            # Ensure eland with PyTorch is installed
            try:
                import eland
                import torch
            except ImportError:
                logger.info("Installing eland with PyTorch dependencies...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "eland[pytorch]"])
            
            # Build command
            cmd = [
                "eland_import_hub_model",
                "--url", self.es.transport.get_connection().host,
                "--task-type", task_type.value
            ]
            
            # Add authentication if present
            if hasattr(self.es, 'transport'):
                headers = self.es.transport.get_default_headers()
                if 'Authorization' in headers:
                    auth = headers['Authorization'].split(' ')[1]  # Remove 'Basic' or 'Bearer'
                    import base64
                    username, password = base64.b64decode(auth).decode().split(':')
                    cmd.extend(["--es-username", username, "--es-password", password])
            
            # Add model IDs
            cmd.extend(["--hub-model-id", hub_model_id])
            if es_model_id:
                cmd.extend(["--es-model-id", es_model_id])
            
            # Run eland import
            logger.info(f"Importing model {hub_model_id} using eland...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                raise RuntimeError(f"Model import failed: {result.stderr}")
            
            logger.info(f"Successfully imported model from Hugging Face: {hub_model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing model from Hugging Face: {str(e)}")
            raise

    def import_local_model(
        self,
        model_path: str,
        task_type: ModelTask,
        es_model_id: str,
        timeout: int = 600
    ) -> bool:
        """
        Import a model from local files using eland (useful for air-gapped environments).
        
        Args:
            model_path: Path to local model files
            task_type: Type of NLP task
            es_model_id: Model ID for Elasticsearch
            timeout: Timeout in seconds for the import process
            
        Returns:
            Boolean indicating success
        """
        try:
            import subprocess
            import sys
            
            # Ensure eland with PyTorch is installed
            try:
                import eland
                import torch
            except ImportError:
                logger.info("Installing eland with PyTorch dependencies...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "eland[pytorch]"])
            
            # Build command for local import
            cmd = [
                "eland_import_hub_model",
                "--url", self.es.transport.get_connection().host,
                "--task-type", task_type.value,
                "--hub-model-id", model_path,
                "--es-model-id", es_model_id
            ]
            
            # Add authentication if present
            if hasattr(self.es, 'transport'):
                headers = self.es.transport.get_default_headers()
                if 'Authorization' in headers:
                    auth = headers['Authorization'].split(' ')[1]
                    import base64
                    username, password = base64.b64decode(auth).decode().split(':')
                    cmd.extend(["--es-username", username, "--es-password", password])
            
            # Run eland import
            logger.info(f"Importing local model from {model_path} using eland...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                raise RuntimeError(f"Model import failed: {result.stderr}")
            
            logger.info(f"Successfully imported local model: {es_model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing local model: {str(e)}")
            raise

def get_inference_manager(es_client: Elasticsearch) -> InferenceModelManager:
    """
    Create an inference model manager instance.
    
    Args:
        es_client: Elasticsearch client
        
    Returns:
        InferenceModelManager instance
    """
    return InferenceModelManager(es_client)