"""
Data Management Module
Handles dataset loading, preprocessing, and management for RAG evaluation
"""

from .dataset_manager import DatasetManager
from .ms_marco_loader import MSMarcoLoader
from .document_processor import DocumentProcessor

__all__ = [
    "DatasetManager",
    "MSMarcoLoader", 
    "DocumentProcessor"
] 