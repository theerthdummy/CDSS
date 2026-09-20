"""HTTP REST Client for communicating with the Agent 4 Clinical Data Fusion Service."""

import logging
from typing import Any, Dict, Optional
import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.input_models import UnifiedClinicalContext

logger = logging.getLogger("agent5.clients.agent4")


class Agent4ClientError(Exception):
    """Base exception raised for Agent 4 client communication or processing errors."""
    pass


class Agent4UnavailableError(Agent4ClientError):
    """Raised when the Agent 4 service is unreachable, timed out, or returns a 5xx error."""
    pass


class Agent4ValidationError(Agent4ClientError):
    """Raised when Agent 4 returns malformed JSON or an invalid UnifiedClinicalContext payload."""
    pass


class Agent4Client:
    """HTTP Client for sending evidence payloads to Agent 4 (/api/v1/fuse) and receiving UnifiedClinicalContext."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.agent_4_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.upstream_timeout_seconds

    def check_health() -> bool:
        """Check if Agent 4 service is available via GET /api/v1/."""
        url = f"{self.base_url}/api/v1/"
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(url)
                return response.status_code == 200
        except Exception as exc:
            logger.debug(f"Agent 4 health check failed: {str(exc)}")
            return False

    def fuse_context(self, payload: Dict[str, Any], correlation_id: Optional[str] = None) -> UnifiedClinicalContext:
        """Send raw Agent 2 and Agent 3 evidence to Agent 4 /api/v1/fuse and return validated UnifiedClinicalContext.

        Args:
            payload: FusionRequest dict containing agent2_output and agent3_output.
            correlation_id: Optional correlation ID string for request tracing.

        Returns:
            UnifiedClinicalContext: Validated Pydantic context model.

        Raises:
            Agent4UnavailableError: If Agent 4 service is down, times out, or returns 5xx error.
            Agent4ValidationError: If request/response JSON is invalid or fails schema validation.
            Agent4ClientError: For general client HTTP errors.
        """
        url = f"{self.base_url}/api/v1/fuse"
        headers = {}
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                logger.info(f"Posting clinical data fusion request to Agent 4 at '{url}'...")
                response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error(f"Agent 4 service request timed out ({self.timeout_seconds}s): {str(exc)}")
            raise Agent4UnavailableError(f"Agent 4 service timed out after {self.timeout_seconds}s.") from exc
        except httpx.RequestError as exc:
            logger.error(f"Failed to connect to Agent 4 service at '{url}': {str(exc)}")
            raise Agent4UnavailableError(f"Agent 4 service unavailable at '{self.base_url}'.") from exc

        if response.status_code >= 500:
            logger.error(f"Agent 4 returned server error HTTP {response.status_code}: {response.text}")
            raise Agent4UnavailableError(f"Agent 4 returned server error HTTP {response.status_code}.")

        if response.status_code >= 400:
            logger.warning(f"Agent 4 returned client error HTTP {response.status_code}: {response.text}")
            raise Agent4ValidationError(f"Agent 4 returned client error HTTP {response.status_code}: {response.text}")

        try:
            response_data = response.json()
        except Exception as exc:
            logger.error(f"Failed to parse JSON response from Agent 4: {str(exc)}")
            raise Agent4ValidationError("Agent 4 returned malformed non-JSON response.") from exc

        return self.extract_unified_context_from_response(response_data)

    @staticmethod
    def extract_unified_context_from_response(response_data: Dict[str, Any]) -> UnifiedClinicalContext:
        """Extract and validate UnifiedClinicalContext from Agent 4 FusionResponse JSON dictionary."""
        if not isinstance(response_data, dict):
            raise Agent4ValidationError("Agent 4 response payload must be a JSON dictionary.")

        if "unified_context" in response_data:
            raw_context = response_data["unified_context"]
        elif "merged_findings" in response_data:
            raw_context = response_data
        else:
            raise Agent4ValidationError("Agent 4 response payload missing 'unified_context' field.")

        try:
            return UnifiedClinicalContext.model_validate(raw_context)
        except ValidationError as val_err:
            logger.error(f"UnifiedClinicalContext schema validation failed for Agent 4 output: {str(val_err)}")
            raise Agent4ValidationError(f"Agent 4 output failed schema validation: {str(val_err)}") from val_err
