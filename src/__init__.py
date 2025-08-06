"""
RAG Evaluation POC with DeepEval
Production-ready RAG evaluation from development to CI/CD
"""

__version__ = "1.0.0"
__author__ = "RAG Evaluation Team"
__email__ = "team@rag-eval.com"

from .rag import RAGSystem
from .evaluation import DeepEvalFramework
from .data import DatasetManager

__all__ = ["RAGSystem", "DeepEvalFramework", "DatasetManager"] 