"""
Enhanced Document Processor with Evaluation Integration
Combines document processing with chunking evaluation metrics
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from dataclasses import dataclass

# Use absolute imports instead of relative imports
from src.data.document_processor import DocumentProcessor
from src.evaluation.chunking_evaluator import ChunkingEvaluator, ChunkingMetrics, ProcessingMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of enhanced document processing with evaluations"""
    chunks: List[Dict[str, Any]]
    chunking_metrics: ChunkingMetrics
    processing_metrics: ProcessingMetrics
    processing_time: float
    evaluation_time: float
    total_documents: int
    total_chunks: int


class EnhancedDocumentProcessor:
    """
    Enhanced document processor with integrated evaluations
    
    Features:
    - Real-time chunking quality evaluation
    - Processing quality assessment
    - Performance monitoring
    - Quality-based chunk optimization
    - Detailed evaluation reporting
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        enable_evaluations: bool = True,
        evaluation_model: str = "gpt-4o-mini",
        quality_threshold: float = 0.7
    ):
        """
        Initialize enhanced document processor
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            enable_evaluations: Whether to enable evaluations
            evaluation_model: Model for evaluations
            quality_threshold: Minimum quality threshold
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_evaluations = enable_evaluations
        self.quality_threshold = quality_threshold
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if enable_evaluations:
            self.evaluator = ChunkingEvaluator(evaluation_model)
        else:
            self.evaluator = None
        
        self.processing_history = []
        
        logger.info("Enhanced document processor initialized")
    
    def process_documents_with_evaluation(
        self,
        file_paths: List[str],
        optimize_chunks: bool = True
    ) -> ProcessingResult:
        """
        Process documents with integrated evaluations
        
        Args:
            file_paths: List of file paths to process
            optimize_chunks: Whether to optimize chunks based on quality
            
        Returns:
            Processing result with evaluations
        """
        start_time = time.time()
        logger.info(f"Processing {len(file_paths)} documents with evaluations")
        
        # Step 1: Process documents normally
        chunks = self.document_processor.process_files(file_paths)
        
        processing_time = time.time() - start_time
        evaluation_time = 0.0
        
        chunking_metrics = None
        processing_metrics = None
        
        # Step 2: Evaluate processing if enabled
        if self.enable_evaluations and self.evaluator:
            eval_start_time = time.time()
            
            # Evaluate processing quality
            processing_metrics = self.evaluator.evaluate_processing_quality(
                chunks, file_paths
            )
            
            # Extract original text for chunking evaluation
            original_texts = self._extract_original_texts(chunks)
            
            # Evaluate chunking quality
            chunking_metrics = self.evaluator.evaluate_chunking_quality(
                [chunk['content'] for chunk in chunks],
                '\n'.join(original_texts),
                self.chunk_size,
                self.chunk_overlap
            )
            
            evaluation_time = time.time() - eval_start_time
            
            # Step 3: Optimize chunks if requested and quality is low
            if optimize_chunks and chunking_metrics.overall_chunk_quality < self.quality_threshold:
                logger.info("Chunking quality below threshold, attempting optimization")
                chunks = self._optimize_chunks(chunks, chunking_metrics)
                
                # Re-evaluate after optimization
                chunking_metrics = self.evaluator.evaluate_chunking_quality(
                    [chunk['content'] for chunk in chunks],
                    '\n'.join(original_texts),
                    self.chunk_size,
                    self.chunk_overlap
                )
        
        # Create processing result
        result = ProcessingResult(
            chunks=chunks,
            chunking_metrics=chunking_metrics,
            processing_metrics=processing_metrics,
            processing_time=processing_time,
            evaluation_time=evaluation_time,
            total_documents=len(file_paths),
            total_chunks=len(chunks)
        )
        
        # Store in history
        self.processing_history.append(result)
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        if evaluation_time > 0:
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        
        return result
    
    def process_single_document_with_evaluation(
        self,
        file_path: str,
        optimize_chunks: bool = True
    ) -> ProcessingResult:
        """
        Process a single document with evaluation
        
        Args:
            file_path: Path to document
            optimize_chunks: Whether to optimize chunks
            
        Returns:
            Processing result with evaluations
        """
        return self.process_documents_with_evaluation([file_path], optimize_chunks)
    
    def _extract_original_texts(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract original text content from chunks"""
        # This is a simplified version - in practice, you'd want to reconstruct
        # the original text more carefully
        texts = []
        current_source = None
        current_text = ""
        
        for chunk in chunks:
            source = chunk.get('meta', {}).get('source', 'unknown')
            
            if source != current_source:
                if current_text:
                    texts.append(current_text)
                current_source = source
                current_text = ""
            
            current_text += chunk['content'] + " "
        
        if current_text:
            texts.append(current_text)
        
        return texts
    
    def _optimize_chunks(self, chunks: List[Dict[str, Any]], metrics: ChunkingMetrics) -> List[Dict[str, Any]]:
        """
        Optimize chunks based on evaluation metrics
        
        Args:
            chunks: Original chunks
            metrics: Chunking quality metrics
            
        Returns:
            Optimized chunks
        """
        logger.info("Optimizing chunks based on evaluation metrics")
        
        optimized_chunks = []
        
        # Strategy 1: Improve chunk boundaries if semantic boundary score is low
        if metrics.semantic_boundary_respect < 0.6:
            optimized_chunks = self._improve_semantic_boundaries(chunks)
        
        # Strategy 2: Adjust chunk sizes if size consistency is low
        elif metrics.chunk_size_consistency < 0.6:
            optimized_chunks = self._adjust_chunk_sizes(chunks)
        
        # Strategy 3: Improve overlap if overlap quality is low
        elif metrics.chunk_overlap_quality < 0.6:
            optimized_chunks = self._improve_overlaps(chunks)
        
        # Strategy 4: General optimization
        else:
            optimized_chunks = self._general_chunk_optimization(chunks)
        
        return optimized_chunks if optimized_chunks else chunks
    
    def _improve_semantic_boundaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improve semantic boundaries in chunks"""
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            
            # Try to find better sentence boundaries
            sentences = content.split('. ')
            if len(sentences) > 1:
                # Ensure chunk ends with complete sentence
                if not content.strip().endswith(('.', '!', '?')):
                    # Find the last complete sentence
                    last_complete = '. '.join(sentences[:-1]) + '.'
                    if len(last_complete) >= self.chunk_size * 0.7:  # At least 70% of target size
                        content = last_complete
            
            # Update chunk
            optimized_chunk = chunk.copy()
            optimized_chunk['content'] = content
            optimized_chunk['meta'] = chunk['meta'].copy()
            optimized_chunk['meta']['optimized'] = True
            optimized_chunk['meta']['optimization_type'] = 'semantic_boundaries'
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def _adjust_chunk_sizes(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adjust chunk sizes for better consistency"""
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            
            # If chunk is too small, try to merge with next chunk
            if len(content) < self.chunk_size * 0.6 and i < len(chunks) - 1:
                next_content = chunks[i + 1]['content']
                combined = content + " " + next_content
                
                if len(combined) <= self.chunk_size * 1.2:
                    # Create merged chunk
                    optimized_chunk = chunk.copy()
                    optimized_chunk['content'] = combined
                    optimized_chunk['meta'] = chunk['meta'].copy()
                    optimized_chunk['meta']['optimized'] = True
                    optimized_chunk['meta']['optimization_type'] = 'size_adjustment'
                    optimized_chunk['meta']['merged_chunks'] = [i, i + 1]
                    
                    optimized_chunks.append(optimized_chunk)
                    # Skip next chunk as it's now merged
                    continue
            
            # If chunk is too large, try to split it
            elif len(content) > self.chunk_size * 1.4:
                # Split at sentence boundaries
                sentences = content.split('. ')
                mid_point = len(sentences) // 2
                
                first_half = '. '.join(sentences[:mid_point]) + '.'
                second_half = '. '.join(sentences[mid_point:])
                
                if len(first_half) >= self.chunk_size * 0.6:
                    # Create first half chunk
                    first_chunk = chunk.copy()
                    first_chunk['content'] = first_half
                    first_chunk['meta'] = chunk['meta'].copy()
                    first_chunk['meta']['optimized'] = True
                    first_chunk['meta']['optimization_type'] = 'size_adjustment'
                    first_chunk['meta']['split_part'] = 1
                    
                    optimized_chunks.append(first_chunk)
                    
                    # Create second half chunk
                    second_chunk = chunk.copy()
                    second_chunk['content'] = second_half
                    second_chunk['meta'] = chunk['meta'].copy()
                    second_chunk['meta']['optimized'] = True
                    second_chunk['meta']['optimization_type'] = 'size_adjustment'
                    second_chunk['meta']['split_part'] = 2
                    
                    optimized_chunks.append(second_chunk)
                    continue
            
            # Keep original chunk if no optimization needed
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _improve_overlaps(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improve chunk overlaps"""
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            
            # Add overlap with previous chunk if available
            if i > 0:
                prev_content = chunks[i - 1]['content']
                
                # Find common words at the end of previous chunk
                prev_words = prev_content.split()[-5:]  # Last 5 words
                current_words = content.split()[:5]     # First 5 words
                
                # Find overlap
                overlap_words = []
                for j in range(min(len(prev_words), len(current_words))):
                    if prev_words[-(j+1):] == current_words[:j+1]:
                        overlap_words = current_words[:j+1]
                
                if overlap_words:
                    # Add overlap to current chunk
                    overlap_text = ' '.join(overlap_words)
                    content = overlap_text + " " + content
            
            # Update chunk
            optimized_chunk = chunk.copy()
            optimized_chunk['content'] = content
            optimized_chunk['meta'] = chunk['meta'].copy()
            optimized_chunk['meta']['optimized'] = True
            optimized_chunk['meta']['optimization_type'] = 'overlap_improvement'
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def _general_chunk_optimization(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """General chunk optimization"""
        optimized_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            
            # Clean up excessive whitespace
            content = ' '.join(content.split())
            
            # Ensure proper sentence endings
            if content and not content.strip().endswith(('.', '!', '?')):
                content = content.strip() + '.'
            
            # Update chunk
            optimized_chunk = chunk.copy()
            optimized_chunk['content'] = content
            optimized_chunk['meta'] = chunk['meta'].copy()
            optimized_chunk['meta']['optimized'] = True
            optimized_chunk['meta']['optimization_type'] = 'general_cleanup'
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def get_processing_history(self) -> List[ProcessingResult]:
        """Get processing history"""
        return self.processing_history
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all processing evaluations"""
        if not self.evaluator:
            return {"message": "Evaluations not enabled"}
        
        return self.evaluator.get_evaluation_summary()
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics from recent processing"""
        if not self.processing_history:
            return {"message": "No processing history available"}
        
        latest_result = self.processing_history[-1]
        
        metrics = {
            "total_documents": latest_result.total_documents,
            "total_chunks": latest_result.total_chunks,
            "processing_time": latest_result.processing_time,
            "evaluation_time": latest_result.evaluation_time,
            "chunks_per_document": latest_result.total_chunks / latest_result.total_documents if latest_result.total_documents > 0 else 0
        }
        
        if latest_result.chunking_metrics:
            metrics["chunking_quality"] = {
                "overall": latest_result.chunking_metrics.overall_chunk_quality,
                "coherence": latest_result.chunking_metrics.chunk_coherence,
                "completeness": latest_result.chunking_metrics.chunk_completeness,
                "overlap_quality": latest_result.chunking_metrics.chunk_overlap_quality,
                "size_consistency": latest_result.chunking_metrics.chunk_size_consistency,
                "semantic_boundaries": latest_result.chunking_metrics.semantic_boundary_respect
            }
        
        if latest_result.processing_metrics:
            metrics["processing_quality"] = {
                "overall": latest_result.processing_metrics.overall_processing_quality,
                "text_cleanliness": latest_result.processing_metrics.text_cleanliness,
                "content_preservation": latest_result.processing_metrics.content_preservation,
                "format_handling": latest_result.processing_metrics.format_handling,
                "metadata_quality": latest_result.processing_metrics.metadata_quality,
                "processing_efficiency": latest_result.processing_metrics.processing_efficiency
            }
        
        return metrics 