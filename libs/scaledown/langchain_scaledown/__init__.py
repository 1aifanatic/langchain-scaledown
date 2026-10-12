from importlib import metadata

from langchain_scaledown._client import ScaledownAPIError, ScaledownClient
from langchain_scaledown.middleware import (
    ScaledownCompressionMiddleware,
    ScaledownExtractionMiddleware,
    ScaledownSummarizationMiddleware,
)
from langchain_scaledown.summarizer import ScaledownSummarizer

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
    "ScaledownSummarizer",
    "__version__",
]
