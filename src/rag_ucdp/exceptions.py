"""Custom exception hierarchy for rag_ucdp domain errors.

All errors raised by ``rag_ucdp`` derive from :class:`RagUcdpError`, so callers
at the pipeline boundary can catch the package's errors with a single
``except`` clause and surface clean messages to the UI.
"""


class RagUcdpError(Exception):
    """Base exception for all errors raised by the rag_ucdp package."""


class ConfigValidationError(RagUcdpError):
    """The configuration file or config object failed validation."""


class DataValidationError(RagUcdpError):
    """The input data did not conform to the expected schema or content."""


class RouterUncertainError(RagUcdpError):
    """The router could not confidently classify a query into a path."""


class RetrievalFailedError(RagUcdpError):
    """The retrieval subsystem failed to return any candidates."""
