from importlib import metadata

from langchain_scaledown._client import ScaledownAPIError, ScaledownClient
from langchain_scaledown.middleware import (
    ScaledownCompressionMiddleware,
    ScaledownExtractionMiddleware,
    ScaledownSummarizationMiddleware,
    ScaledownToolCompressionMiddleware,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ScaledownAPIError",
    "ScaledownClient",
    "ScaledownCompressionMiddleware",
    "ScaledownExtractionMiddleware",
    "ScaledownSummarizationMiddleware",
    "ScaledownToolCompressionMiddleware",
    "__version__",
]
