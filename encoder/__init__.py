# encoder/__init__.py
"""Encoders from expression data to SUBIT distributions."""

from .universal import UniversalEncoder, encode_to_subit

__all__ = ["UniversalEncoder", "encode_to_subit"]