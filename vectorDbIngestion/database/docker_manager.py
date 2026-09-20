"""
Docker container lifecycle manager for Qdrant vector database.
Manages the 'vectordb-cdss' container: creation, cleanup, and health checks.
"""

import logging
import time
import sys

import config

logger = logging.getLogger(__name__)


def _get_docker_client():
    """
    Get a Docker client instance.
    Raises a clear error if Docker is not installed or not running.
    """
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return client
    except ImportError:
        logger.error(
            "The 'docker' Python package is not installed. "
            "Run: pip install docker"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Cannot connect to Docker. Is Docker Desktop running? "
            f"Error: {e}"
        )
        sys.exit(1)


def ensure_container_running():
    """
    Ensure the Qdrant Docker container is running if using localhost.
    Skips Docker management if using Qdrant Cloud.
    """
    if "cloud.qdrant.io" in config.QDRANT_URL or config.QDRANT_API_KEY:
        logger.info(f"Using remote Qdrant Cloud instance ({config.QDRANT_URL}). Skipping local Docker container setup.")
        return

    client = _get_docker_client()
    container_name = config.QDRANT_CONTAINER_NAME
    image = config.QDRANT_DOCKER_IMAGE
    port = config.QDRANT_HOST_PORT
    volume_name = config.QDRANT_STORAGE_VOLUME

    # --- Cleanup existing container ---
    try:
        existing = client.containers.get(container_name)
        logger.info(f"Found existing container '{container_name}'. Stopping and removing...")
        existing.stop(timeout=10)
        existing.remove(force=True)
        logger.info(f"Container '{container_name}' removed.")
    except Exception:
        logger.info(f"No existing container '{container_name}' found. Creating new one.")

    # --- Cleanup existing volume ---
    try:
        vol = client.volumes.get(volume_name)
        vol.remove(force=True)
        logger.info(f"Removed existing volume '{volume_name}'.")
    except Exception:
        pass  # Volume doesn't exist, that's fine

    # --- Pull the Qdrant image ---
    logger.info(f"Pulling Docker image '{image}'...")
    try:
        client.images.pull(image)
        logger.info(f"Image '{image}' pulled successfully.")
    except Exception as e:
        logger.error(f"Failed to pull image '{image}': {e}")
        sys.exit(1)

    # --- Create and start the container ---
    logger.info(f"Creating container '{container_name}' on port {port}...")
    try:
        container = client.containers.run(
            image,
            name=container_name,
            ports={"6333/tcp": port, "6334/tcp": port + 1},
            volumes={volume_name: {"bind": "/qdrant/storage", "mode": "rw"}},
            detach=True,
            restart_policy={"Name": "unless-stopped"},
        )
        logger.info(f"Container '{container_name}' started (ID: {container.short_id}).")
    except Exception as e:
        logger.error(f"Failed to start container: {e}")
        sys.exit(1)

    # --- Wait for Qdrant to become healthy ---
    logger.info("Waiting for Qdrant to become ready...")
    import requests
    health_url = f"http://localhost:{port}/readyz"
    for attempt in range(30):
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                logger.info("Qdrant is healthy and ready.")
                return
        except Exception:
            pass
        time.sleep(1)

    logger.error("Qdrant did not become healthy within 30 seconds.")
    sys.exit(1)


def stop_container():
    """
    Gracefully stop the Qdrant container.
    """
    client = _get_docker_client()
    try:
        container = client.containers.get(config.QDRANT_CONTAINER_NAME)
        container.stop(timeout=10)
        logger.info(f"Container '{config.QDRANT_CONTAINER_NAME}' stopped.")
    except Exception as e:
        logger.warning(f"Could not stop container: {e}")
