"""
Custom RAG Components for Haystack Integration
Includes OpenAI generator and other custom components
"""

import requests
import json
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import os

from haystack import component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Optional dependencies for OpenAI") as openai_import:
    import openai


@component
class OpenAIGenerator:
    """
    OpenAI generator component for Haystack pipelines
    
    Integrates with OpenAI API for text generation using GPT models
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 30,
    ):
        """
        Initialize OpenAI generator
        
        Args:
            model: OpenAI model name (e.g., gpt-4o-mini, gpt-3.5-turbo)
            api_key: OpenAI API key
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize OpenAI client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Successfully initialized OpenAI client with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary"""
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIGenerator":
        """Deserialize component from dictionary"""
        return default_from_dict(cls, data)
    
    @component.output_types(replies=List[str])
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate text using OpenAI
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Dictionary containing generated replies
        """
        logger.info(f"Generating response with model {self.model}")
        
        if not self.client:
            logger.error("OpenAI client not initialized")
            return {"replies": ["Error: OpenAI client not initialized"]}
        
        try:
            # Make request to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            logger.info("Successfully generated response")
            return {"replies": [generated_text]}
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"replies": [f"Error: {str(e)}"]}


@component
class OllamaGenerator:
    """
    Ollama generator component for Haystack pipelines
    
    Integrates with local Ollama instance for text generation
    """
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 30,
    ):
        """
        Initialize Ollama generator
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Validate connection
        self._validate_connection()
    
    def _validate_connection(self):
        """Validate connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary"""
        return default_to_dict(
            self,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaGenerator":
        """Deserialize component from dictionary"""
        return default_from_dict(cls, data)
    
    @component.output_types(replies=List[str])
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate text using Ollama
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Dictionary containing generated replies
        """
        logger.info(f"Generating response with model {self.model}")
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.info("Successfully generated response")
            return {"replies": [generated_text]}
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"replies": [f"Error: {str(e)}"]}


@component
class QdrantRetriever:
    """
    Qdrant vector database retriever component
    
    Retrieves relevant documents based on embedding similarity
    """
    
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        Initialize Qdrant retriever
        
        Args:
            collection_name: Name of Qdrant collection
            url: Qdrant server URL
            api_key: Optional API key for authentication
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # Initialize Qdrant client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client connection"""
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
            
            logger.info(f"Connected to Qdrant at {self.url}")
            
        except ImportError:
            logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary"""
        return default_to_dict(
            self,
            collection_name=self.collection_name,
            url=self.url,
            api_key=self.api_key,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdrantRetriever":
        """Deserialize component from dictionary"""
        return default_from_dict(cls, data)
    
    @component.output_types(documents=List[Dict[str, Any]])
    def run(self, query_embedding: List[float]) -> Dict[str, Any]:
        """
        Retrieve documents based on embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Dictionary containing retrieved documents
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return {"documents": []}
        
        try:
            # Search for similar documents
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.top_k,
                score_threshold=self.score_threshold,
            )
            
            # Convert results to Haystack document format
            documents = []
            for point in search_result:
                doc = {
                    "content": point.payload.get("content", ""),
                    "meta": point.payload.get("meta", {}),
                    "score": point.score,
                    "id": str(point.id),
                }
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents")
            return {"documents": documents}
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {"documents": []}


@dataclass
class DocumentProcessor:
    """
    Document processing utilities for RAG system
    """
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the end position for this chunk
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # Try to break at a natural separator
            if end_pos < len(text):
                for separator in self.separators:
                    last_sep = text.rfind(separator, current_pos, end_pos)
                    if last_sep > current_pos:
                        end_pos = last_sep + len(separator)
                        break
            
            # Extract chunk
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next position with overlap
            current_pos = max(end_pos - self.chunk_overlap, current_pos + 1)
            
            # Prevent infinite loop
            if current_pos >= len(text):
                break
        
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents into chunks
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed document chunks
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                chunks = self.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc_chunk = {
                        "content": chunk,
                        "meta": {
                            "source": file_path,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    }
                    all_chunks.append(doc_chunk)
                    
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return all_chunks


class HaystackRetriever:
    """
    Wrapper for Haystack retriever components
    Provides compatibility with different retriever types
    """
    
    def __init__(self, retriever_component):
        """
        Initialize with a Haystack retriever component
        
        Args:
            retriever_component: Any Haystack retriever component
        """
        self.retriever = retriever_component
    
    def retrieve(self, query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using the wrapped retriever
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve (optional override)
            
        Returns:
            List of retrieved documents
        """
        try:
            # Update top_k if provided
            if top_k and hasattr(self.retriever, 'top_k'):
                original_top_k = self.retriever.top_k
                self.retriever.top_k = top_k
            
            # Run retrieval
            result = self.retriever.run(query_embedding=query_embedding)
            documents = result.get("documents", [])
            
            # Restore original top_k
            if top_k and hasattr(self.retriever, 'top_k'):
                self.retriever.top_k = original_top_k
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return [] 