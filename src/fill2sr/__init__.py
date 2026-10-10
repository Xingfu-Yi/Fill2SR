"""Fill2SR inference package."""

__all__ = ["Fill2SRPipeline"]
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "Fill2SRPipeline":
        from .core import Fill2SRPipeline
        return Fill2SRPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
