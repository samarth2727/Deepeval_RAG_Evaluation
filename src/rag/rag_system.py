"""
Production-ready RAG System using Haystack
Integrates vector database with OpenAI GPT models
"""

import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import os

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

from .components import OpenAIGenerator, QdrantRetriever
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG system response with metadata"""
    query: str
    answer: str
    retrieved_contexts: List[str]
    retrieval_scores: List[float]
    metadata: Dict[str, Any]


class RAGSystem:
    """
    Production-ready RAG system using Haystack framework
    
    Features:
    - Vector database integration (In-Memory or Qdrant)
    - OpenAI GPT models for generation
    - Configurable retrieval and generation
    - Comprehensive logging and monitoring
    """

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Initialize RAG system with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.document_store = None
        self.pipeline = None
        self.indexing_pipeline = None
        
        self._setup_logging()
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default config with OpenAI settings
            return {
                'llm': {
                    'provider': 'openai',
                    'model_name': 'gpt-4o-mini',
                    'api_key': os.getenv('OPENAI_API_KEY'),  # Get from environment variable
                    'temperature': 0.1,
                    'max_tokens': 512
                },
                'embeddings': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'batch_size': 32
                },
                'retrieval': {
                    'top_k': 5,
                    'score_threshold': 0.7
                },
                'document_processing': {
                    'chunk_size': 500,
                    'chunk_overlap': 50
                }
            }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/rag_system.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize RAG components"""
        logger.info("Initializing RAG system components...")
        
        # Initialize document store
        self.document_store = InMemoryDocumentStore()
        
        # Initialize embedding models
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=self.config['embeddings']['model_name'],
            batch_size=self.config['embeddings'].get('batch_size', 32)
        )
        
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=self.config['embeddings']['model_name']
        )
        
        # Initialize retriever
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store
        )
        
        # Initialize OpenAI generator
        api_key = self.config['llm'].get('api_key') or os.getenv('OPENAI_API_KEY')
        self.generator = OpenAIGenerator(
            model=self.config['llm']['model_name'],
            api_key=api_key,
            temperature=self.config['llm'].get('temperature', 0.1),
            max_tokens=self.config['llm'].get('max_tokens', 512)
        )
        
        # Setup pipelines
        self._setup_pipelines()
        
        logger.info("RAG system components initialized successfully")
    
    def _setup_pipelines(self):
        """Setup indexing and query pipelines"""
        
        # Indexing pipeline
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("converter", TextFileToDocument())
        self.indexing_pipeline.add_component("splitter", DocumentSplitter(
            split_by="sentence",
            split_length=self.config['document_processing']['chunk_size'],
            split_overlap=self.config['document_processing']['chunk_overlap']
        ))
        self.indexing_pipeline.add_component("embedder", self.document_embedder)
        self.indexing_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
        
        self.indexing_pipeline.connect("converter", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")
        
        # Query pipeline with RAG prompt
        rag_prompt = """
        Given the following context information, please answer the question as accurately as possible.
        
        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}
        
        Question: {{ query }}
        
        Answer: Provide a comprehensive answer based on the context above. If the context doesn't contain enough information, please say so.
        """
        
        self.pipeline = Pipeline()
        self.pipeline.add_component("text_embedder", self.text_embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", PromptBuilder(template=rag_prompt))
        self.pipeline.add_component("generator", self.generator)
        
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "generator")
    
    def index_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Index documents into the vector database
        
        Args:
            document_paths: List of paths to documents to index
            
        Returns:
            Dictionary with indexing results and statistics
        """
        logger.info(f"Starting document indexing for {len(document_paths)} files")
        
        try:
            results = []
            total_docs = 0
            
            for doc_path in document_paths:
                logger.info(f"Processing document: {doc_path}")
                
                result = self.indexing_pipeline.run({
                    "converter": {"sources": [doc_path]}
                })
                
                docs_written = len(result["writer"]["documents_written"])
                total_docs += docs_written
                results.append({
                    "path": doc_path,
                    "chunks_created": docs_written
                })
                
                logger.info(f"Indexed {docs_written} chunks from {doc_path}")
            
            logger.info(f"Document indexing completed. Total chunks: {total_docs}")
            
            return {
                "success": True,
                "total_documents": len(document_paths),
                "total_chunks": total_docs,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG system
        
        Args:
            question: User question to answer
            
        Returns:
            RAGResponse with answer and retrieved contexts
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Run the pipeline
            result = self.pipeline.run({
                "text_embedder": {"text": question},
                "prompt_builder": {"query": question}
            })
            
            # Extract results
            answer = result["generator"]["replies"][0] if result["generator"]["replies"] else "No answer generated"
            retrieved_docs = result["retriever"]["documents"]
            
            # Prepare response
            response = RAGResponse(
                query=question,
                answer=answer,
                retrieved_contexts=[doc.content for doc in retrieved_docs],
                retrieval_scores=[doc.score for doc in retrieved_docs],
                metadata={
                    "num_retrieved": len(retrieved_docs),
                    "pipeline_run_id": result.get("run_id"),
                    "model": self.config['llm']['model_name']
                }
            )
            
            logger.info(f"Query processed successfully. Retrieved {len(retrieved_docs)} contexts")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return RAGResponse(
                query=question,
                answer=f"Error processing query: {str(e)}",
                retrieved_contexts=[],
                retrieval_scores=[],
                metadata={"error": str(e)}
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        return {
            "model": self.config['llm']['model_name'],
            "embedding_model": self.config['embeddings']['model_name'],
            "vector_db": "in_memory",
            "document_count": self.document_store.count_documents(),
            "config": self.config
        } 