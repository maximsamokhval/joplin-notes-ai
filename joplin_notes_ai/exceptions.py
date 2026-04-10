"""Custom exceptions used across the application."""


class AppError(Exception):
    """Base application error."""


class ConfigurationError(AppError):
    """Raised when settings are invalid."""


class IntegrationError(AppError):
    """Raised when a remote integration fails."""


class JoplinApiError(IntegrationError):
    """Raised for Joplin API failures."""


class LlmApiError(IntegrationError):
    """Raised for LLM API failures."""


class LlmResponseValidationError(LlmApiError):
    """Raised when LLM returns invalid JSON payload."""


class VectorStoreError(IntegrationError):
    """Raised for vector store failures."""
