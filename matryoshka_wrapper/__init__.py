
"""
Matryoshka Embeddings Library
============================

A Python library for creating and using Matryoshka embeddings with multiple dimensions.

Basic usage:
    >>> from matryoshka_embeddings import load_model
    >>> model, tokenizer = load_model("your-repo-id")
    >>> embedding = model.get_embedding("مرحبا بالعالم", tokenizer, dim="256")
"""

from .core import MatryoshkaWrapper, load_model

__version__ = "0.1.0"
__author__ = "Abdalrahman54"
__email__ = "abdalrahmankamel6@gmail.com"

__all__ = [
    "MatryoshkaWrapper",
    "load_model"
]
