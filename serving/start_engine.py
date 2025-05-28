#!/usr/bin/env python3
"""Standalone script to start the vLLM serving engine.

This script initializes and runs the vLLM serving engine without the API gateway,
which is useful for services that need to access the engine directly.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import load_config
from serving import NeuroplasticVLLMEngine

logger = structlog.get_logger(__name__)

async def main(config_path: str):
    """Initialize and run the vLLM serving engine."""
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    log_level = config.logging.level.lower()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger.info("Starting vLLM serving engine", config_path=config_path)
    
    # Create the engine
    engine = NeuroplasticVLLMEngine(config)
    
    # Initialize the engine
    if not await engine.initialize():
        logger.error("Failed to initialize vLLM engine")
        return 1
    
    # Log model information
    logger.info("Model information", 
               model_name=config.model.name,
               model_dtype=config.model.dtype,
               max_model_len=config.model.max_model_len)
    
    # Start the engine
    await engine.start_server()
    
    # Run until interrupted
    try:
        logger.info("vLLM engine running", 
                   host=config.serving.host,
                   port=config.serving.port)
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
    finally:
        # Shutdown the engine
        await engine.shutdown()
        logger.info("vLLM engine shutdown complete")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start vLLM serving engine")
    parser.add_argument("--config", type=str, default="config/base.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Run the main function
    exit_code = asyncio.run(main(args.config))
    sys.exit(exit_code) 