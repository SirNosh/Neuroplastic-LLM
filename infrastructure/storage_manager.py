"""Storage management for neuroplastic Qwen system."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import asdict
from datetime import datetime

import structlog
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import torch

logger = structlog.get_logger(__name__)


class StorageManager:
    """Manages cloud storage operations for model artifacts and training data."""
    
    def __init__(self, config):
        self.config = config
        self.storage_config = config.storage
        self.s3_client = None
        self.bucket_name = self.storage_config.bucket
        self.region = self.storage_config.region
        self.local_cache_dir = Path(self.storage_config.local_cache_dir)
        self.running = False
        
        self.provider = self.storage_config.type.lower()
        
    async def initialize(self) -> bool:
        """Initialize storage connections and verify access."""
        try:
            logger.info(
                "Initializing storage manager", 
                provider=self.provider,
                bucket=self.bucket_name, 
                region=self.region
            )
            
            if self.provider == "s3" or self.provider == "minio":
                session_kwargs = {}
                if self.region:
                    session_kwargs['region_name'] = self.region
                if self.storage_config.aws_access_key_id:
                    session_kwargs['aws_access_key_id'] = self.storage_config.aws_access_key_id
                if self.storage_config.aws_secret_access_key:
                    session_kwargs['aws_secret_access_key'] = self.storage_config.aws_secret_access_key
                
                session = boto3.Session(**session_kwargs)
                
                client_kwargs = {}
                if self.provider == "minio" and self.storage_config.endpoint_url:
                    client_kwargs['endpoint_url'] = self.storage_config.endpoint_url
                
                self.s3_client = session.client('s3', **client_kwargs)
                await self._verify_s3_bucket_access()
            
            # elif self.provider == "gcs":
            #     # Initialize GCS client
            #     # self.gcs_client = gcs_storage.Client()
            #     # await self._verify_gcs_bucket_access()
            #     logger.error("GCS provider not yet fully implemented")
            #     return False
            # elif self.provider == "azure":
            #     # Initialize Azure client
            #     # self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_config.azure_connection_string)
            #     # await self._verify_azure_container_access()
            #     logger.error("Azure provider not yet fully implemented")
            #     return False
            else:
                logger.error(f"Unsupported storage provider: {self.provider}")
                return False
            
            # Create local cache directory
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.running = True
            logger.info("Storage manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize storage manager", error=str(e))
            return False
    
    async def _verify_s3_bucket_access(self):
        """Verify access to the S3 bucket."""
        try:
            # Try to list objects to verify access
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            logger.info("Bucket access verified", bucket=self.bucket_name)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                logger.warning("Bucket does not exist, attempting to create", bucket=self.bucket_name)
                await self._create_s3_bucket()
            else:
                logger.error("Failed to access bucket", bucket=self.bucket_name, error=str(e))
                raise
        except NoCredentialsError:
            logger.error("AWS credentials not found for S3")
            raise
    
    async def _create_s3_bucket(self):
        """Create the S3 bucket if it doesn't exist."""
        try:
            create_kwargs = {'Bucket': self.bucket_name}
            if self.region and self.region != 'us-east-1':
                create_kwargs['CreateBucketConfiguration'] = {
                    'LocationConstraint': self.region
                }
            
            self.s3_client.create_bucket(**create_kwargs)
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            logger.info("S3 Bucket created successfully", bucket=self.bucket_name)
            
        except ClientError as e:
            logger.error("Failed to create S3 bucket", bucket=self.bucket_name, error=str(e))
            raise
    
    def _get_object_key(self, path: str, prefix: Optional[str] = None) -> str:
        """Generate S3 object key from path."""
        base_prefix = self.storage_config.prefix.strip('/')
        user_prefix = prefix.strip('/') if prefix else ""
        actual_path = path.strip('/')

        parts = []
        if base_prefix:
            parts.append(base_prefix)
        if user_prefix:
            parts.append(user_prefix)
        parts.append(actual_path)
        
        return "/".join(parts)
    
    def _get_local_cache_path(self, object_key: str) -> Path:
        """Get local cache path for an object."""
        return self.local_cache_dir / object_key.replace('/', '_')
    
    async def upload_file(
        self, 
        local_path: Union[str, Path], 
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> bool:
        """Upload a file to S3."""
        if not self.running:
            logger.warning("Storage manager not initialized")
            return False
        
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error("Local file does not exist", path=str(local_path))
                return False
            
            object_key = self._get_object_key(remote_path)
            
            if self.provider == "s3" or self.provider == "minio":
                try:
                    upload_args = {}
                    if metadata:
                        upload_args['Metadata'] = metadata
                    if content_type:
                        upload_args['ContentType'] = content_type
                    
                    self.s3_client.upload_file(
                        Filename=str(local_path),
                        Bucket=self.bucket_name,
                        Key=object_key,
                        ExtraArgs=upload_args if upload_args else None
                    )
                    logger.info("File uploaded to S3 successfully", 
                               local_path=str(local_path), 
                               remote_path=object_key)
                    return True
                except Exception as e:
                    logger.error("Failed to upload file to S3", 
                                local_path=str(local_path), 
                                remote_path=object_key, 
                                error=str(e))
                    return False
            else:
                logger.error(f"Upload not implemented for provider: {self.provider}")
                return False
            
        except Exception as e:
            logger.error("Failed to upload file", 
                        local_path=str(local_path), 
                        remote_path=remote_path, 
                        error=str(e))
            return False
    
    async def download_file(
        self, 
        remote_path: str, 
        local_path: Optional[Union[str, Path]] = None,
        use_cache: bool = True
    ) -> Optional[Path]:
        """Download a file from S3."""
        if not self.running:
            logger.warning("Storage manager not initialized")
            return None
        
        try:
            object_key = self._get_object_key(remote_path)
            
            # Determine local path
            if local_path:
                local_path = Path(local_path)
            else:
                local_path = self._get_local_cache_path(object_key)
            
            # Check cache if enabled
            if use_cache and local_path.exists():
                # Verify file integrity with S3 (if S3 provider)
                if self.provider == "s3" or self.provider == "minio":
                    try:
                        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
                        s3_etag = response['ETag'].strip('"') # ETag includes quotes
                        local_size = local_path.stat().st_size
                        s3_size = response['ContentLength']
                        
                        # Basic check: if size matches, assume it's okay. ETag check can be added.
                        if local_size == s3_size:
                            logger.debug("Using cached file", path=str(local_path))
                            return local_path
                        else:
                            logger.debug("Cache miss due to size mismatch", 
                                        local_size=local_size, s3_size=s3_size)
                    except ClientError as e:
                        if e.response['Error']['Code'] == '404':
                            logger.warning("File not found in S3 for cache validation", key=object_key)
                            # If file not in S3 but exists in cache, it's stale. Redownload.
                        else:
                            logger.warning("Error checking S3 for cache validation", key=object_key, error=str(e))
                            # Uncertain state, proceed to download to be safe
                else:
                    # For other providers, simple existence check for now
                    logger.debug("Using cached file (non-S3 provider)", path=str(local_path))
                    return local_path
            
            # Ensure directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if self.provider == "s3" or self.provider == "minio":
                try:
                    self.s3_client.download_file(
                        Bucket=self.bucket_name,
                        Key=object_key,
                        Filename=str(local_path)
                    )
                    logger.info("File downloaded from S3 successfully", 
                               remote_path=object_key, 
                               local_path=str(local_path))
                    return local_path
                except Exception as e:
                    logger.error("Failed to download file from S3", 
                                remote_path=object_key, 
                                local_path=str(local_path),
                                error=str(e))
                    return None
            else:
                logger.error(f"Download not implemented for provider: {self.provider}")
                return None
            
        except Exception as e:
            logger.error("Failed to download file", 
                        remote_path=remote_path, 
                        error=str(e))
            return None
    
    async def upload_lora_adapter(
        self, 
        adapter_path: Union[str, Path], 
        adapter_id: str,
        version: str = "latest"
    ) -> bool:
        """Upload a LoRA adapter to storage."""
        adapter_path = Path(adapter_path)
        
        if not adapter_path.exists():
            logger.error("LoRA adapter path does not exist", path=str(adapter_path))
            return False
        
        try:
            # Create metadata
            metadata = {
                'adapter_id': adapter_id,
                'version': version,
                'upload_time': datetime.utcnow().isoformat(),
                'type': 'lora_adapter'
            }
            
            if adapter_path.is_file():
                # Single file adapter
                remote_path = f"lora_adapters/{adapter_id}/{version}/{adapter_path.name}"
                return await self.upload_file(adapter_path, remote_path, metadata)
            
            else:
                # Directory with multiple files
                success = True
                for file_path in adapter_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(adapter_path)
                        remote_path = f"lora_adapters/{adapter_id}/{version}/{relative_path}"
                        if not await self.upload_file(file_path, remote_path, metadata):
                            success = False
                
                return success
                
        except Exception as e:
            logger.error("Failed to upload LoRA adapter", 
                        adapter_id=adapter_id, 
                        error=str(e))
            return False
    
    async def download_lora_adapter(
        self, 
        adapter_id: str, 
        version: str = "latest",
        local_dir: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """Download a LoRA adapter from storage."""
        try:
            # Determine local directory
            if local_dir:
                local_dir = Path(local_dir)
            else:
                local_dir = self.local_cache_dir / "lora_adapters" / adapter_id / version
            
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # List all files for this adapter
            s3_prefix = self._get_object_key(f"lora_adapters/{adapter_id}/{version}/")

            if self.provider == "s3" or self.provider == "minio":
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=s3_prefix
                    )
                    
                    if 'Contents' not in response or not response['Contents']:
                        logger.error("LoRA adapter not found in S3", adapter_id=adapter_id, version=version, prefix=s3_prefix)
                        return None
                    
                    download_tasks = []
                    for obj in response['Contents']:
                        object_key = obj['Key']
                        # Construct relative path for local storage from the s3_prefix
                        relative_file_path = Path(object_key).relative_to(Path(s3_prefix))
                        local_file_path = local_dir / relative_file_path
                        local_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure sub-dirs exist
                        
                        # Download each file (can be parallelized if needed)
                        download_tasks.append(self.download_file(object_key, local_file_path, use_cache=True))
                    
                    results = await asyncio.gather(*download_tasks)
                    if not all(results):
                        logger.error("Failed to download one or more adapter files", adapter_id=adapter_id, version=version)
                        return None

                    logger.info("LoRA adapter downloaded successfully from S3", 
                               adapter_id=adapter_id, 
                               version=version,
                               local_dir=str(local_dir))
                    return local_dir
                except Exception as e:
                    logger.error("Failed to download LoRA adapter from S3", adapter_id=adapter_id, error=str(e))
                    return None
            else:
                logger.error(f"Download LoRA not implemented for provider: {self.provider}")
                return None
            
        except Exception as e:
            logger.error("Failed to download LoRA adapter", 
                        adapter_id=adapter_id, 
                        error=str(e))
            return None
    
    async def save_checkpoint(
        self, 
        model_state: Dict[str, Any], 
        checkpoint_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save a model checkpoint to storage."""
        try:
            # Create temporary file for checkpoint
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            # Save checkpoint locally first
            checkpoint_data = {
                'model_state': model_state,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'checkpoint_id': checkpoint_id
            }
            
            torch.save(checkpoint_data, tmp_path)
            
            # Upload to S3
            remote_path = f"checkpoints/{checkpoint_id}.pt"
            upload_metadata = {
                'checkpoint_id': checkpoint_id,
                'timestamp': str(time.time()),
                'type': 'model_checkpoint'
            }
            
            success = await self.upload_file(tmp_path, remote_path, upload_metadata)
            
            # Cleanup temporary file
            tmp_path.unlink()
            
            if success:
                logger.info("Checkpoint saved successfully", checkpoint_id=checkpoint_id)
            
            return success
            
        except Exception as e:
            logger.error("Failed to save checkpoint", 
                        checkpoint_id=checkpoint_id, 
                        error=str(e))
            return False
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint from storage."""
        try:
            remote_path = f"checkpoints/{checkpoint_id}.pt"
            local_path = await self.download_file(remote_path)
            
            if not local_path or not local_path.exists():
                logger.error("Failed to download checkpoint", checkpoint_id=checkpoint_id)
                return None
            
            # Load checkpoint
            checkpoint_data = torch.load(local_path, map_location='cpu')
            
            logger.info("Checkpoint loaded successfully", checkpoint_id=checkpoint_id)
            return checkpoint_data
            
        except Exception as e:
            logger.error("Failed to load checkpoint", 
                        checkpoint_id=checkpoint_id, 
                        error=str(e))
            return None
    
    async def list_lora_adapters(self) -> List[Dict[str, Any]]:
        """List all available LoRA adapters."""
        try:
            adapters = []
            # Use the storage_config.prefix as the base for LoRA adapters
            lora_base_prefix = self._get_object_key("lora_adapters/")
            
            if self.provider == "s3" or self.provider == "minio":
                paginator = self.s3_client.get_paginator('list_objects_v2')
                # List adapter IDs (first level directories under lora_base_prefix)
                adapter_id_pages = paginator.paginate(
                    Bucket=self.bucket_name, 
                    Prefix=lora_base_prefix, 
                    Delimiter='/'
                )
                
                for page in adapter_id_pages:
                    if 'CommonPrefixes' in page:
                        for adapter_folder_info in page['CommonPrefixes']:
                            adapter_full_prefix = adapter_folder_info['Prefix']
                            # Extract adapter_id: last part of prefix, removing trailing slash
                            adapter_id = Path(adapter_full_prefix.rstrip('/')).name
                            
                            # List versions for this adapter
                            version_pages = paginator.paginate(
                                Bucket=self.bucket_name, 
                                Prefix=adapter_full_prefix, # adapter_full_prefix already has trailing slash 
                                Delimiter='/'
                            )
                            
                            versions = []
                            for version_page in version_pages:
                                if 'CommonPrefixes' in version_page:
                                    for version_folder_info in version_page['CommonPrefixes']:
                                        version_full_prefix = version_folder_info['Prefix']
                                        version = Path(version_full_prefix.rstrip('/')).name
                                        versions.append(version)
                            
                            if versions:
                                adapters.append({
                                    'adapter_id': adapter_id,
                                    'versions': sorted(versions, reverse=True), # Sort for latest first
                                    'latest_version': sorted(versions, reverse=True)[0]
                                })
                return adapters
            else:
                logger.warning(f"List LoRA adapters not implemented for provider: {self.provider}")
                return []
            
        except Exception as e:
            logger.error("Failed to list LoRA adapters", error=str(e))
            return []
    
    async def delete_object(self, object_key: str) -> bool:
        """Delete an object from S3."""
        # Note: object_key here should be the full S3 key including storage_config.prefix
        if self.provider == "s3" or self.provider == "minio":
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
                logger.info("Object deleted successfully from S3", object_key=object_key)
                return True
            except Exception as e:
                logger.error("Failed to delete object from S3", object_key=object_key, error=str(e))
                return False
        else:
            logger.error(f"Delete object not implemented for provider: {self.provider}")
            return False
    
    async def delete_lora_adapter(self, adapter_id: str, version: Optional[str] = None) -> bool:
        """Delete a LoRA adapter (specific version or all versions)."""
        try:
            if version:
                # Delete a specific version
                s3_prefix_to_delete = self._get_object_key(f"lora_adapters/{adapter_id}/{version}/")
            else:
                # Delete all versions of an adapter
                s3_prefix_to_delete = self._get_object_key(f"lora_adapters/{adapter_id}/")
            
            if self.provider == "s3" or self.provider == "minio":
                # List all objects with the prefix
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix_to_delete)
                
                objects_to_delete = []
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            objects_to_delete.append({'Key': obj['Key']})
            
            if not objects_to_delete:
                logger.warning("No S3 objects found to delete for LoRA adapter", 
                               adapter_id=adapter_id, version=version, prefix=s3_prefix_to_delete)
                return True # Considered success if nothing to delete
            
            # Delete objects in batches (S3 limit is 1000 per request)
            batch_size = 1000  # S3 limit
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i:i + batch_size]
                response = self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': batch, 'Quiet': True} # Quiet mode suppresses success responses for each key
                )
                # Check for errors in response
                if response.get('Errors'):
                    for err in response['Errors']:
                        logger.error("Error deleting S3 object during LoRA cleanup", 
                                     key=err['Key'], code=err['Code'], message=err['Message'])
                    # Decide if this constitutes overall failure
                    # For now, log errors but continue trying to delete others / report success if some batches worked
            
            logger.info("LoRA adapter S3 objects deleted successfully", 
                       adapter_id=adapter_id, 
                       version=version,
                       objects_deleted=len(objects_to_delete))
            return True
            
        except Exception as e:
            logger.error("Failed to delete LoRA adapter", 
                        adapter_id=adapter_id, 
                        version=version,
                        error=str(e))
            return False
    
    async def get_object_metadata(self, object_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an S3 object."""
        # Note: object_key here should be the full S3 key including storage_config.prefix
        if self.provider == "s3" or self.provider == "minio":
            try:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
                
                return {
                    'content_length': response.get('ContentLength'),
                    'last_modified': response.get('LastModified'),
                    'etag': response.get('ETag', '').strip('"'),
                    'content_type': response.get('ContentType'),
                    'metadata': response.get('Metadata', {}),
                    'version_id': response.get('VersionId')
                }
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.debug("Object not found for metadata retrieval", object_key=object_key)
                    return None
                logger.error("Failed to get S3 object metadata", object_key=object_key, error=str(e))
                return None
            except Exception as e:
                logger.error("Unexpected error getting S3 object metadata", object_key=object_key, error=str(e))
                return None
        else:
            logger.warning(f"Get object metadata not implemented for provider: {self.provider}")
            return None
    
    async def cleanup_cache(self, max_age_hours: Optional[int] = None): # Use config default if None
        """Clean up old files from local cache."""
        resolved_max_age_hours = max_age_hours if max_age_hours is not None else self.storage_config.cache_max_age_hours
        
        if not self.local_cache_dir.exists():
            logger.debug("Local cache directory does not exist, skipping cleanup.")
            return
        try:
            current_time = time.time()
            deleted_count = 0
            
            for file_path in self.local_cache_dir.rglob('*'):
                if file_path.is_file():
                    file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                    if file_age_hours > resolved_max_age_hours:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except OSError as e:
                            logger.warning(f"Error deleting cached file {file_path}: {e}")
            
            # Remove empty directories
            for dir_path in sorted(list(self.local_cache_dir.rglob('*')), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                    except OSError as e:
                        logger.warning(f"Error deleting empty cache directory {dir_path}: {e}")
            
            logger.info("Cache cleanup completed", 
                       deleted_files=deleted_count, 
                       max_age_hours=resolved_max_age_hours)
            
        except Exception as e:
            logger.error("Failed to cleanup cache", error=str(e))
    
    async def shutdown(self):
        """Shutdown storage manager."""
        logger.info("Shutting down storage manager")
        self.running = False
        
        # Close S3 client connections
        if self.s3_client:
            # boto3 clients are generally thread-safe and don't need explicit close unless using underlying connections
            # For now, no explicit close operation is standard for boto3.client
            pass 
        # if self.gcs_client: self.gcs_client.close() # Example
        # if self.blob_service_client: self.blob_service_client.close() # Example

        logger.info("Storage manager shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the storage manager."""
        cache_size = 0
        cache_files = 0
        if self.local_cache_dir.exists():
            try:
                cache_size = sum(
                    f.stat().st_size for f in self.local_cache_dir.rglob('*') if f.is_file()
                )
                cache_files = sum(1 for f in self.local_cache_dir.rglob('*') if f.is_file())
            except Exception as e:
                logger.warning("Could not calculate cache size/files", error=str(e))

        return {
            "running": self.running,
            "provider": self.provider,
            "bucket_name": self.bucket_name,
            "region": self.region,
            "prefix": self.storage_config.prefix,
            "local_cache_dir": str(self.local_cache_dir),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "cache_files": cache_files,
            "client_available": (
                (self.s3_client is not None) if (self.provider == 's3' or self.provider == 'minio') 
                # or (self.gcs_client is not None if self.provider == 'gcs') 
                # or (self.blob_service_client is not None if self.provider == 'azure')
                else False
            )
        } 