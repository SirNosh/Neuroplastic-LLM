"""Kafka management for neuroplastic Qwen system."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import asdict

import structlog
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError

logger = structlog.get_logger(__name__)


class KafkaManager:
    """Manages Kafka connections, topics, and message streaming."""
    
    def __init__(self, config):
        self.config = config
        self.kafka_config = config.kafka
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.admin_client: Optional[KafkaAdminClient] = None
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize Kafka connections and create topics."""
        try:
            logger.info("Initializing Kafka manager", servers=self.kafka_config.bootstrap_servers)
            
            # Create admin client
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                client_id="neuroplastic-qwen-admin"
            )
            
            # Create topics if they don't exist
            await self._create_topics()
            
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                client_id="neuroplastic-qwen-producer",
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True,
                compression_type='gzip',
                linger_ms=10,  # Batch messages for better throughput
                batch_size=16384,
                buffer_memory=33554432,  # 32MB buffer
            )
            
            self.running = True
            logger.info("Kafka manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Kafka manager", error=str(e))
            return False
    
    async def _create_topics(self):
        """Create required topics if they don't exist."""
        topics_to_create = [
            NewTopic(
                name=self.kafka_config.topics.traces,
                num_partitions=3,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'compression.type': 'gzip',
                    'cleanup.policy': 'delete'
                }
            ),
            NewTopic(
                name=self.kafka_config.topics.feedback,
                num_partitions=2,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(30 * 24 * 60 * 60 * 1000),  # 30 days
                    'compression.type': 'gzip',
                    'cleanup.policy': 'delete'
                }
            ),
            NewTopic(
                name=self.kafka_config.topics.ewc_samples,
                num_partitions=2,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(24 * 60 * 60 * 1000),  # 1 day
                    'compression.type': 'gzip',
                    'cleanup.policy': 'delete'
                }
            ),
            NewTopic(
                name=self.kafka_config.topics.tot_traces,
                num_partitions=2,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'compression.type': 'gzip',
                    'cleanup.policy': 'delete'
                }
            ),
            NewTopic(
                name=self.kafka_config.topics.metrics,
                num_partitions=1,
                replication_factor=1,
                topic_configs={
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'compression.type': 'gzip',
                    'cleanup.policy': 'delete'
                }
            )
        ]
        
        try:
            result = self.admin_client.create_topics(topics_to_create, validate_only=False)
            
            # Wait for topic creation
            for topic_name, future in result.topic_futures.items():
                try:
                    future.result()  # Block until topic is created
                    logger.info("Topic created successfully", topic=topic_name)
                except TopicAlreadyExistsError:
                    logger.info("Topic already exists", topic=topic_name)
                except Exception as e:
                    logger.error("Failed to create topic", topic=topic_name, error=str(e))
                    
        except Exception as e:
            logger.error("Failed to create topics", error=str(e))
            raise
    
    async def send_message(
        self, 
        topic: str, 
        message: Dict[str, Any], 
        key: Optional[str] = None,
        partition: Optional[int] = None
    ) -> bool:
        """Send a message to a Kafka topic."""
        if not self.running or not self.producer:
            logger.warning("Kafka producer not available")
            return False
        
        try:
            # Convert dataclass to dict if needed
            if hasattr(message, '__dataclass_fields__'):
                message_dict = asdict(message)
            else:
                message_dict = message
            
            # Add timestamp if not present
            if 'timestamp' not in message_dict:
                message_dict['timestamp'] = time.time()
            
            future = self.producer.send(
                topic=topic,
                value=message_dict,
                key=key,
                partition=partition
            )
            
            # Don't block on the result for async operation
            logger.debug("Message sent to Kafka", topic=topic, key=key)
            return True
            
        except KafkaError as e:
            logger.error("Failed to send message to Kafka", topic=topic, error=str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error sending message", topic=topic, error=str(e))
            return False
    
    async def send_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Send a trace message to the traces topic."""
        return await self.send_message(
            topic=self.kafka_config.topics.traces,
            message=trace_data,
            key=trace_data.get('session_id')
        )
    
    async def send_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Send feedback data to the feedback topic."""
        return await self.send_message(
            topic=self.kafka_config.topics.feedback,
            message=feedback_data,
            key=feedback_data.get('session_id')
        )
    
    async def send_ewc_sample(self, sample_data: Dict[str, Any]) -> bool:
        """Send EWC sample data to the EWC samples topic."""
        return await self.send_message(
            topic=self.kafka_config.topics.ewc_samples,
            message=sample_data,
            key=sample_data.get('request_id')
        )
    
    async def send_tot_trace(self, tot_data: Dict[str, Any]) -> bool:
        """Send Tree-of-Thought trace to the ToT traces topic."""
        return await self.send_message(
            topic=self.kafka_config.topics.tot_traces,
            message=tot_data,
            key=tot_data.get('problem_id')
        )
    
    async def send_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Send metrics data to the metrics topic."""
        return await self.send_message(
            topic=self.kafka_config.topics.metrics,
            message=metrics_data,
            key=metrics_data.get('component', 'system')
        )
    
    def create_consumer(
        self, 
        topic: str, 
        consumer_group: Optional[str] = None,
        auto_offset_reset: Optional[str] = None
    ) -> KafkaConsumer:
        """Create a Kafka consumer for a specific topic."""
        if not self.running:
            raise RuntimeError("Kafka manager not initialized")
        
        group_id = consumer_group or self.kafka_config.consumer_group
        offset_reset = auto_offset_reset or self.kafka_config.auto_offset_reset
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=offset_reset,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            max_poll_records=100,
            max_poll_interval_ms=300000,  # 5 minutes
            session_timeout_ms=30000,    # 30 seconds
            heartbeat_interval_ms=10000, # 10 seconds
        )
        
        consumer_id = f"{topic}_{group_id}"
        self.consumers[consumer_id] = consumer
        
        logger.info("Consumer created", topic=topic, group_id=group_id)
        return consumer
    
    async def consume_messages(
        self, 
        topic: str, 
        message_handler: Callable[[Dict[str, Any]], None],
        consumer_group: Optional[str] = None,
        max_messages: Optional[int] = None
    ):
        """Consume messages from a topic with a handler function."""
        consumer = self.create_consumer(topic, consumer_group)
        
        try:
            message_count = 0
            logger.info("Starting message consumption", topic=topic)
            
            for message in consumer:
                if not self.running:
                    break
                
                try:
                    # Process the message
                    await message_handler(message.value)
                    message_count += 1
                    
                    if max_messages and message_count >= max_messages:
                        break
                        
                except Exception as e:
                    logger.error(
                        "Error processing message", 
                        topic=topic, 
                        error=str(e),
                        message_key=message.key
                    )
                    
        except Exception as e:
            logger.error("Error in message consumption", topic=topic, error=str(e))
        finally:
            consumer.close()
            consumer_id = f"{topic}_{consumer_group or self.kafka_config.consumer_group}"
            if consumer_id in self.consumers:
                del self.consumers[consumer_id]
    
    async def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get information about a specific topic."""
        try:
            metadata = self.admin_client.describe_topics([topic])
            topic_metadata = metadata[topic]
            
            return {
                "topic": topic,
                "partitions": len(topic_metadata.partitions),
                "replication_factor": len(topic_metadata.partitions[0].replicas) if topic_metadata.partitions else 0,
                "partition_info": [
                    {
                        "partition": p.partition,
                        "leader": p.leader,
                        "replicas": p.replicas,
                        "isr": p.isr
                    }
                    for p in topic_metadata.partitions
                ]
            }
        except Exception as e:
            logger.error("Failed to get topic info", topic=topic, error=str(e))
            return {"error": str(e)}
    
    async def get_consumer_group_info(self, group_id: str) -> Dict[str, Any]:
        """Get information about a consumer group."""
        try:
            # Note: This requires kafka-python with admin support for consumer groups
            # For now, return basic info
            return {
                "group_id": group_id,
                "active_consumers": len([c for c in self.consumers.values() if c._group_id == group_id])
            }
        except Exception as e:
            logger.error("Failed to get consumer group info", group_id=group_id, error=str(e))
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown Kafka connections."""
        logger.info("Shutting down Kafka manager")
        self.running = False
        
        # Close all consumers
        for consumer_id, consumer in self.consumers.items():
            try:
                consumer.close()
                logger.debug("Consumer closed", consumer_id=consumer_id)
            except Exception as e:
                logger.error("Error closing consumer", consumer_id=consumer_id, error=str(e))
        
        self.consumers.clear()
        
        # Close producer
        if self.producer:
            try:
                self.producer.flush(timeout=10)  # Wait up to 10 seconds for pending messages
                self.producer.close()
                logger.debug("Producer closed")
            except Exception as e:
                logger.error("Error closing producer", error=str(e))
        
        # Close admin client
        if self.admin_client:
            try:
                self.admin_client.close()
                logger.debug("Admin client closed")
            except Exception as e:
                logger.error("Error closing admin client", error=str(e))
        
        logger.info("Kafka manager shutdown complete")
    
    def flush_producer(self, timeout: float = 10.0):
        """Flush pending messages in the producer."""
        if self.producer:
            self.producer.flush(timeout=timeout)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Kafka manager."""
        active_consumer_topics = []
        if self.consumers:
            for consumer_id, consumer_instance in self.consumers.items():
                try:
                    # KafkaConsumer.subscription() returns a set of topics
                    # or None if not subscribed (e.g. if manually assigned partitions)
                    sub = consumer_instance.subscription()
                    if sub:
                        active_consumer_topics.extend(list(sub))
                    else:
                        # If subscription() is None, it might be manually assigned partitions.
                        # KafkaConsumer.assignment() returns a set of TopicPartition instances.
                        assignments = consumer_instance.assignment()
                        if assignments:
                            active_consumer_topics.extend(list(set(tp.topic for tp in assignments)))
                except Exception as e:
                    logger.debug("Could not get subscription/assignment for consumer", consumer_id=consumer_id, error=str(e))
        
        return {
            "running": self.running,
            "producer_available": self.producer is not None,
            "admin_client_available": self.admin_client is not None,
            "active_consumers_count": len(self.consumers),
            "active_consumer_topics": sorted(list(set(active_consumer_topics))),
            "bootstrap_servers": self.kafka_config.bootstrap_servers,
            "client_id_producer": self.producer.config.get('client_id') if self.producer else None,
            "client_id_admin": self.admin_client.config.get('client_id') if self.admin_client else None,
        } 