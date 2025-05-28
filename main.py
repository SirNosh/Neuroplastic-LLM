#!/usr/bin/env python3
"""Main entry point for neuroplastic Qwen system."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import structlog
import uvicorn
import click

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from serving import NeuroplasticVLLMEngine
from api import NeuroplasticAPI


def setup_logging(config):
    """Setup structured logging."""
    # Create logs directory
    log_file = Path(config.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if config.logging.format == "json" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    log_level = getattr(logging, config.logging.level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.logging.file)
        ]
    )


async def create_app(config_path: str = None):
    """Create and configure the application."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    
    logger = structlog.get_logger(__name__)
    logger.info("Starting Neuroplastic Qwen System", config_path=config_path)
    
    # Create serving engine
    if config.serving.engine.lower() == "vllm":
        serving_engine = NeuroplasticVLLMEngine(config)
    else:
        raise ValueError(f"Unsupported serving engine: {config.serving.engine}")
    
    # Create API gateway
    api_gateway = NeuroplasticAPI(config, serving_engine)
    
    logger.info("Application created successfully")
    return api_gateway.app, config


@click.group()
def cli():
    """Neuroplastic Qwen CLI."""
    pass


@cli.command()
@click.option(
    "--config", 
    "-c", 
    default=None, 
    help="Path to configuration file"
)
@click.option(
    "--host", 
    default=None, 
    help="Host to bind to (overrides config)"
)
@click.option(
    "--port", 
    default=None, 
    type=int, 
    help="Port to bind to (overrides config)"
)
@click.option(
    "--workers", 
    default=1, 
    type=int, 
    help="Number of worker processes"
)
@click.option(
    "--reload", 
    is_flag=True, 
    help="Enable auto-reload for development"
)
@click.option(
    "--log-level", 
    default=None, 
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Log level (overrides config)"
)
def serve(config, host, port, workers, reload, log_level):
    """Start the API server."""
    
    async def run_server():
        app, app_config = await create_app(config)
        
        # Override config with CLI options
        final_host = host or app_config.api.host
        final_port = port or app_config.api.port
        final_log_level = log_level or app_config.api.log_level
        
        logger = structlog.get_logger(__name__)
        logger.info(
            "Starting server",
            host=final_host,
            port=final_port,
            workers=workers,
            reload=reload
        )
        
        # Run the server
        uvicorn_config = uvicorn.Config(
            app,
            host=final_host,
            port=final_port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level=final_log_level,
            access_log=True,
            loop="asyncio"
        )
        
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    
    # Run the server
    asyncio.run(run_server())


@cli.command()
@click.option(
    "--config", 
    "-c", 
    default=None, 
    help="Path to configuration file"
)
def validate_config(config):
    """Validate configuration file."""
    try:
        config_obj = load_config(config)
        click.echo("✅ Configuration is valid!")
        
        # Print some key info
        click.echo(f"Model: {config_obj.model.name}")
        click.echo(f"Serving engine: {config_obj.serving.engine}")
        click.echo(f"API port: {config_obj.api.port}")
        click.echo(f"Kafka servers: {config_obj.kafka.bootstrap_servers}")
        click.echo(f"Storage bucket: {config_obj.storage.bucket}")
        
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--config", 
    "-c", 
    default=None, 
    help="Path to configuration file"
)
@click.option(
    "--prompt", 
    "-p", 
    required=True, 
    help="Test prompt"
)
@click.option(
    "--lora-id", 
    default=None, 
    help="LoRA adapter ID to use"
)
def test_generation(config, prompt, lora_id):
    """Test text generation."""
    
    async def run_test():
        app, app_config = await create_app(config)
        
        # Get the serving engine from the app
        # This is a bit hacky, but works for testing
        api_gateway = None
        for route in app.routes:
            if hasattr(route, 'dependant') and hasattr(route.dependant, 'call'):
                func = route.dependant.call
                if hasattr(func, '__self__') and hasattr(func.__self__, 'serving_engine'):
                    api_gateway = func.__self__
                    break
        
        if not api_gateway:
            click.echo("❌ Could not find serving engine")
            return
        
        serving_engine = api_gateway.serving_engine
        
        try:
            click.echo(f"🤖 Generating response for: '{prompt}'")
            if lora_id:
                click.echo(f"📦 Using LoRA adapter: {lora_id}")
            
            start_time = asyncio.get_event_loop().time()
            
            response = await serving_engine.generate(
                prompt=prompt,
                lora_id=lora_id,
                max_tokens=100,
                temperature=0.7
            )
            
            end_time = asyncio.get_event_loop().time()
            latency = end_time - start_time
            
            click.echo(f"\n✅ Response ({latency:.2f}s):")
            click.echo(f"'{response}'")
            
        except Exception as e:
            click.echo(f"❌ Generation failed: {e}")
        finally:
            await serving_engine.shutdown()
    
    asyncio.run(run_test())


@cli.command()
@click.option(
    "--config", 
    "-c", 
    default=None, 
    help="Path to configuration file"
)
def health_check(config):
    """Check system health."""
    
    async def run_health_check():
        app, app_config = await create_app(config)
        
        # Similar hack as above to get the serving engine
        api_gateway = None
        for route in app.routes:
            if hasattr(route, 'dependant') and hasattr(route.dependant, 'call'):
                func = route.dependant.call
                if hasattr(func, '__self__') and hasattr(func.__self__, 'serving_engine'):
                    api_gateway = func.__self__
                    break
        
        if not api_gateway:
            click.echo("❌ Could not find serving engine")
            return
        
        serving_engine = api_gateway.serving_engine
        
        try:
            health_data = await serving_engine.health_check()
            
            status = health_data.get("status", "unknown")
            if status == "healthy":
                click.echo("✅ System is healthy")
            elif status == "degraded":
                click.echo("⚠️ System is degraded")
            else:
                click.echo("❌ System is unhealthy")
            
            click.echo(f"Model loaded: {health_data.get('model_loaded', False)}")
            click.echo(f"Active LoRAs: {health_data.get('active_loras', 0)}")
            
            stats = health_data.get("stats", {})
            if stats:
                click.echo(f"Total requests: {stats.get('total_requests', 0)}")
                click.echo(f"Avg latency: {stats.get('avg_latency', 0):.3f}s")
            
        except Exception as e:
            click.echo(f"❌ Health check failed: {e}")
        finally:
            await serving_engine.shutdown()
    
    asyncio.run(run_health_check())


if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    cli() 