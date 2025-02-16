# elastic_config.py
import os
from typing import Optional, Dict
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Silence elastic_transport INFO logs
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

class ElasticsearchConfig:
    def __init__(self):
        """Initialize Elasticsearch configuration by loading environment variables."""
        load_dotenv()  # Load environment variables from .env file
        
        # Configuration preference
        self.prefer_cloud = os.getenv('ELASTIC_PREFER_CLOUD', 'true').lower() == 'true'
        
        # Cloud configuration
        self.cloud_id = os.getenv('ELASTIC_CLOUD_ID')
        self.cloud_api_key = os.getenv('ELASTIC_API_KEY')
        
        # Self-hosted configuration
        self.hosts = os.getenv('ELASTIC_HOSTS')  # Can be comma-separated list
        self.api_key = os.getenv('ELASTIC_SELF_HOSTED_API_KEY')
        self.username = os.getenv('ELASTIC_USERNAME')
        self.password = os.getenv('ELASTIC_PASSWORD')
        self.ca_certs = os.getenv('ELASTIC_CA_CERTS')  # Path to CA certificate
        self.verify_certs = os.getenv('ELASTIC_VERIFY_CERTS', 'true').lower() == 'true'

    def validate_license(self, client: Elasticsearch) -> bool:
        """
        Check if cluster has an active Enterprise license.
        
        Args:
            client: Elasticsearch client
            
        Returns:
            Boolean indicating if Enterprise features are available
        """
        try:
            response = client.license.get()
            license_info = response.get('license', {})
            
            if license_info.get('status') != 'active':
                logger.warning("⚠️  Elasticsearch license is not active")
                return False
                
            license_type = license_info.get('type', '').lower()
            if license_type != 'enterprise':
                logger.warning(
                    f"⚠️  Current license type '{license_type}' does not support inference features. "
                    "Enterprise license is required."
                )
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking license status: {str(e)}")
            return False
        
    def get_client(self, timeout: int = 60, prefer_cloud: Optional[bool] = None, validate_enterprise: bool = True) -> Elasticsearch:
        """
        Create and return an Elasticsearch client based on available configuration.
        
        Args:
            timeout: Request timeout in seconds
            prefer_cloud: If True, use cloud configuration when both are available
            
        Returns:
            Elasticsearch client configured for either cloud or self-hosted
            
        Raises:
            ValueError: If neither cloud nor self-hosted configuration is complete
        """
        has_cloud = bool(self.cloud_id and self.cloud_api_key)
        has_self_hosted = bool(self.hosts and (self.api_key or (self.username and self.password)))
        
        # Log available configurations and their details
        print("\n🔍 Available Elasticsearch configurations:")
        if has_cloud:
            print("- Cloud configuration detected")
            print(f"  • Cloud ID: {self.cloud_id[:25]}...")
        if has_self_hosted:
            print("- Self-hosted configuration detected")
            # Parse hosts string into list and create summary
            host_list = [host.strip() for host in self.hosts.split(',')]
            host_summary = f"{host_list[0]}"
            if len(host_list) > 1:
                host_summary += f" (+{len(host_list)-1} more)"
            print(f"  • Hosts: {host_summary}")
            if self.api_key:
                print("  • Authentication: API Key")
            elif self.username:
                print("  • Authentication: Basic Auth")
            if self.ca_certs:
                print("  • CA Certificate: Configured")
            
        if has_cloud and has_self_hosted:
            # Use parameter if provided, otherwise use environment setting
            should_use_cloud = prefer_cloud if prefer_cloud is not None else self.prefer_cloud
            if should_use_cloud:
                print("\nℹ️  Using cloud configuration (preferred)")
                client = Elasticsearch(
                    cloud_id=self.cloud_id,
                    api_key=self.cloud_api_key,
                    request_timeout=timeout
                )
                
                # Validate enterprise license if requested
                if validate_enterprise and not self.validate_license(client):
                    logger.warning(
                        "⚠️  Some features requiring Enterprise license may not be available. "
                        "Please contact your administrator if you need inference capabilities."
                    )
                
                return client
            else:
                print("\nℹ️  Using self-hosted configuration (preferred)")
                config: Dict[str, any] = {
                    'hosts': [host.strip() for host in self.hosts.split(',')],
                    'request_timeout': timeout,
                    'verify_certs': self.verify_certs
                }
                
                # Add API key if provided
                if self.api_key:
                    config['api_key'] = self.api_key
                # Otherwise use basic auth if provided
                elif self.username and self.password:
                    config['basic_auth'] = (self.username, self.password)
                
                # Add CA certificate if provided
                if self.ca_certs:
                    config['ca_certs'] = self.ca_certs
                
                return Elasticsearch(**config)
        
        # Try cloud configuration
        if has_cloud:
            print("\nℹ️  Using cloud configuration")
            client = Elasticsearch(
                cloud_id=self.cloud_id,
                api_key=self.cloud_api_key,
                request_timeout=timeout
            )
            
            # Validate enterprise license if requested
            if validate_enterprise and not self.validate_license(client):
                logger.warning(
                    "⚠️  Some features requiring Enterprise license may not be available. "
                    "Please contact your administrator if you need inference capabilities."
                )
            
            return client
        
        # Try self-hosted configuration
        elif has_self_hosted:
            config: Dict[str, any] = {
                'hosts': [host.strip() for host in self.hosts.split(',')],
                'request_timeout': timeout,
                'verify_certs': self.verify_certs
            }
            
            # Add API key if provided
            if self.api_key:
                config['api_key'] = self.api_key
            # Otherwise use basic auth if provided
            elif self.username and self.password:
                config['basic_auth'] = (self.username, self.password)
            
            # Add CA certificate if provided
            if self.ca_certs:
                config['ca_certs'] = self.ca_certs
            
            print("\nℹ️  Using self-hosted configuration")
            client = Elasticsearch(**config)
            
            # Validate enterprise license if requested
            if validate_enterprise and not self.validate_license(client):
                logger.warning(
                    "⚠️  Some features requiring Enterprise license may not be available. "
                    "Please contact your administrator if you need inference capabilities."
                )
            
            return client
        
        raise ValueError(
            "Incomplete Elasticsearch configuration. Please provide either:\n"
            "1. Cloud configuration (ELASTIC_CLOUD_ID and ELASTIC_API_KEY) or\n"
            "2. Self-hosted configuration (ELASTIC_HOSTS and authentication)"
        )

    def validate_connection(self, client: Optional[Elasticsearch] = None, validate_enterprise: bool = True) -> bool:
        """
        Validate Elasticsearch connection and optionally check license status.
        
        Args:
            client: Optional Elasticsearch client (creates new one if not provided)
            validate_enterprise: Whether to validate Enterprise license
            
        Returns:
            bool: True if connection is successful
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            es = client or self.get_client(validate_enterprise=validate_enterprise)
            if not es.ping():
                raise ConnectionError("Failed to ping Elasticsearch")
            
            # Get cluster info
            info = es.info()
            print(f"✅ Connected to Elasticsearch {info['version']['number']}")
            print(f"🔹 Cluster: {info['cluster_name']}")
            
            # Validate license if requested and not already done by get_client
            if validate_enterprise and client:  # Only check if using provided client
                if not self.validate_license(es):
                    logger.warning(
                        "⚠️  Some features requiring Enterprise license may not be available. "
                        "Please contact your administrator if you need inference capabilities."
                    )
            
            return True
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Elasticsearch: {str(e)}")