"""
RAG Pipeline Management and Orchestration
Handles the complete RAG workflow from indexing to query processing
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder

from .components import OpenAIGenerator, DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    run_id: str
    query: str
    execution_time: float
    num_retrieved_docs: int
    retrieval_scores: List[float]
    answer_length: int
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class RAGPipeline:
    """
    Complete RAG pipeline orchestration
    
    Manages the entire RAG workflow including:
    - Document indexing
    - Query processing
    - Response generation
    - Metrics collection
    """
    
    def __init__(
        self,
        document_store,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        openai_api_key: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        """
        Initialize RAG pipeline
        
        Args:
            document_store: Haystack document store
            embedding_model: Name of embedding model
            llm_model: Name of LLM model
            openai_api_key: OpenAI API key
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        self.metrics_history: List[PipelineMetrics] = []
        
        # Initialize components
        self._initialize_components()
        
        # Build pipelines
        self._build_indexing_pipeline()
        self._build_query_pipeline()
        
        logger.info("RAG pipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        
        # Embedding components
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embedding_model
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=self.embedding_model
        )
        
        # Retriever
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k
        )
        
        # Generator (using OpenAI instead of Ollama)
        self.generator = OpenAIGenerator(
            model=self.llm_model,
            api_key=self.openai_api_key
        )
        
        # Document processor
        self.doc_processor = DocumentProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _build_indexing_pipeline(self):
        """Build document indexing pipeline"""
        self.indexing_pipeline = Pipeline()
        
        # Add components
        self.indexing_pipeline.add_component("converter", TextFileToDocument())
        self.indexing_pipeline.add_component("splitter", DocumentSplitter(
            split_by="sentence",
            split_length=self.chunk_size,
            split_overlap=self.chunk_overlap
        ))
        self.indexing_pipeline.add_component("embedder", self.document_embedder)
        self.indexing_pipeline.add_component("writer", DocumentWriter(
            document_store=self.document_store
        ))
        
        # Connect components
        self.indexing_pipeline.connect("converter", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")
        
        logger.info("Indexing pipeline built")
    
    def _build_query_pipeline(self):
        """Build query processing pipeline"""
        # RAG prompt template
        rag_template = """
        You are a helpful AI assistant. Answer the question based on the provided context.
        
        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}
        
        Question: {{ query }}
        
        Instructions:
        - Provide a comprehensive and accurate answer based on the context
        - If the context doesn't contain enough information, say so clearly
        - Use specific details from the context when possible
        - Be concise but thorough
        
        Answer:
        """
        
        self.query_pipeline = Pipeline()
        
        # Add components
        self.query_pipeline.add_component("text_embedder", self.text_embedder)
        self.query_pipeline.add_component("retriever", self.retriever)
        self.query_pipeline.add_component("prompt_builder", PromptBuilder(
            template=rag_template
        ))
        self.query_pipeline.add_component("generator", self.generator)
        
        # Connect components
        self.query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "generator")
        
        logger.info("Query pipeline built")
    
    def index_documents(
        self,
        file_paths: List[str],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Index documents into the vector database
        
        Args:
            file_paths: List of file paths to index
            batch_size: Number of files to process in each batch
            
        Returns:
            Indexing results and statistics
        """
        logger.info(f"Starting indexing of {len(file_paths)} documents")
        start_time = time.time()
        
        results = {
            "total_files": len(file_paths),
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "processing_time": 0,
            "errors": []
        }
        
        try:
            # Process files in batches
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        logger.info(f"Processing: {file_path}")
                        
                        # Run indexing pipeline
                        result = self.indexing_pipeline.run({
                            "converter": {"sources": [file_path]}
                        })
                        
                        # Count documents written
                        docs_written = len(result["writer"]["documents_written"])
                        results["total_chunks"] += docs_written
                        results["successful_files"] += 1
                        
                        logger.info(f"Indexed {docs_written} chunks from {file_path}")
                        
                    except Exception as e:
                        error_msg = f"Failed to index {file_path}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        results["failed_files"] += 1
            
            results["processing_time"] = time.time() - start_time
            
            logger.info(
                f"Indexing completed: {results['successful_files']} successful, "
                f"{results['failed_files']} failed, "
                f"{results['total_chunks']} total chunks"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Indexing pipeline failed: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            results["processing_time"] = time.time() - start_time
            return results
    
    def query(
        self,
        question: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User question
            include_metadata: Whether to include execution metadata
            
        Returns:
            Query results with answer and metadata
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Processing query [{run_id}]: {question}")
        
        try:
            # Run query pipeline
            result = self.query_pipeline.run({
                "text_embedder": {"text": question},
                "prompt_builder": {"query": question}
            })
            
            # Extract results
            answer = result["generator"]["replies"][0] if result["generator"]["replies"] else "No answer generated"
            retrieved_docs = result["retriever"]["documents"]
            retrieval_scores = [doc.score for doc in retrieved_docs]
            
            execution_time = time.time() - start_time
            
            # Create response
            response = {
                "query": question,
                "answer": answer,
                "retrieved_contexts": [doc.content for doc in retrieved_docs],
                "success": True
            }
            
            # Add metadata if requested
            if include_metadata:
                response["metadata"] = {
                    "run_id": run_id,
                    "execution_time": execution_time,
                    "num_retrieved_docs": len(retrieved_docs),
                    "retrieval_scores": retrieval_scores,
                    "answer_length": len(answer),
                    "model": self.llm_model,
                    "embedding_model": self.embedding_model
                }
            
            # Record metrics
            metrics = PipelineMetrics(
                run_id=run_id,
                query=question,
                execution_time=execution_time,
                num_retrieved_docs=len(retrieved_docs),
                retrieval_scores=retrieval_scores,
                answer_length=len(answer),
                timestamp=time.time(),
                success=True
            )
            self.metrics_history.append(metrics)
            
            logger.info(f"Query processed successfully [{run_id}] in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Query processing failed: {str(e)}"
            
            logger.error(f"Query failed [{run_id}]: {error_msg}")
            
            # Record error metrics
            metrics = PipelineMetrics(
                run_id=run_id,
                query=question,
                execution_time=execution_time,
                num_retrieved_docs=0,
                retrieval_scores=[],
                answer_length=0,
                timestamp=time.time(),
                success=False,
                error_message=error_msg
            )
            self.metrics_history.append(metrics)
            
            return {
                "query": question,
                "answer": f"Error: {error_msg}",
                "retrieved_contexts": [],
                "success": False,
                "metadata": {
                    "run_id": run_id,
                    "execution_time": execution_time,
                    "error": error_msg
                }
            }
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated pipeline metrics
        
        Returns:
            Dictionary with pipeline performance metrics
        """
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        successful_runs = [m for m in self.metrics_history if m.success]
        failed_runs = [m for m in self.metrics_history if not m.success]
        
        if successful_runs:
            avg_execution_time = sum(m.execution_time for m in successful_runs) / len(successful_runs)
            avg_retrieved_docs = sum(m.num_retrieved_docs for m in successful_runs) / len(successful_runs)
            avg_answer_length = sum(m.answer_length for m in successful_runs) / len(successful_runs)
        else:
            avg_execution_time = avg_retrieved_docs = avg_answer_length = 0
        
        return {
            "total_queries": len(self.metrics_history),
            "successful_queries": len(successful_runs),
            "failed_queries": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.metrics_history) if self.metrics_history else 0,
            "avg_execution_time": avg_execution_time,
            "avg_retrieved_docs": avg_retrieved_docs,
            "avg_answer_length": avg_answer_length,
            "last_24h_queries": len([m for m in self.metrics_history if time.time() - m.timestamp < 86400]),
            "document_count": self.document_store.count_documents(),
        }
    
    def export_metrics(self, file_path: str):
        """
        Export metrics to JSON file
        
        Args:
            file_path: Path to save metrics file
        """
        import json
        
        metrics_data = {
            "pipeline_config": {
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k
            },
            "aggregated_metrics": self.get_pipeline_metrics(),
            "detailed_metrics": [asdict(m) for m in self.metrics_history]
        }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics exported to {file_path}")
    
    def clear_metrics(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        logger.info("Metrics history cleared") 