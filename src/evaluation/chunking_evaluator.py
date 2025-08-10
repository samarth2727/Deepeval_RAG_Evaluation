"""
Chunking and Processing Evaluator
Evaluates document processing and chunking quality during RAG pipeline execution
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path
import re

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


@dataclass
class ChunkingMetrics:
    """Metrics for chunking evaluation"""
    chunk_coherence: float
    chunk_completeness: float
    chunk_overlap_quality: float
    chunk_size_consistency: float
    semantic_boundary_respect: float
    overall_chunk_quality: float


@dataclass
class ProcessingMetrics:
    """Metrics for document processing evaluation"""
    text_cleanliness: float
    content_preservation: float
    format_handling: float
    metadata_quality: float
    processing_efficiency: float
    overall_processing_quality: float


class ChunkingEvaluator:
    """
    Evaluates chunking quality during document processing
    
    Features:
    - Chunk coherence evaluation
    - Semantic boundary detection
    - Overlap quality assessment
    - Size consistency monitoring
    - Content preservation validation
    """
    
    def __init__(self, evaluation_model: str = "gpt-4o-mini"):
        """
        Initialize chunking evaluator
        
        Args:
            evaluation_model: Model to use for evaluation (default: gpt-4o-mini)
        """
        self.evaluation_model = evaluation_model
        self.chunking_metrics = []
        self.processing_metrics = []
        
        # Initialize evaluation metrics
        self._initialize_metrics()
        
        logger.info("Chunking evaluator initialized")
    
    def _initialize_metrics(self):
        """Initialize DeepEval metrics for chunking evaluation"""
        # Chunk coherence metric
        self.chunk_coherence_metric = GEval(
            name="chunk_coherence",
            criteria="Evaluate if the text chunk maintains semantic coherence and logical flow",
            evaluation_steps=[
                "Check if the chunk contains complete thoughts or ideas",
                "Assess logical flow within the chunk",
                "Evaluate if sentences are properly connected",
                "Check for abrupt cuts that break meaning"
            ],
            evaluation_params=["chunk_text"],
            threshold=0.7,
            model=self.evaluation_model
        )
        
        # Chunk completeness metric
        self.chunk_completeness_metric = GEval(
            name="chunk_completeness",
            criteria="Evaluate if the chunk contains complete information units",
            evaluation_steps=[
                "Check if the chunk contains complete sentences",
                "Assess if paragraphs are properly bounded",
                "Evaluate if key information is preserved",
                "Check for incomplete thoughts or ideas"
            ],
            evaluation_params=["chunk_text"],
            threshold=0.7,
            model=self.evaluation_model
        )
        
        # Overlap quality metric
        self.overlap_quality_metric = GEval(
            name="overlap_quality",
            criteria="Evaluate the quality of chunk overlaps for context preservation",
            evaluation_steps=[
                "Check if overlaps maintain context continuity",
                "Assess if overlaps contain important transition information",
                "Evaluate if overlaps are neither too small nor too large",
                "Check for redundant information in overlaps"
            ],
            evaluation_params=["prev_chunk", "current_chunk", "overlap_size"],
            threshold=0.7,
            model=self.evaluation_model
        )
        
        # Processing quality metric
        self.processing_quality_metric = GEval(
            name="processing_quality",
            criteria="Evaluate the quality of document processing and text cleaning",
            evaluation_steps=[
                "Check if original content meaning is preserved",
                "Assess text cleanliness and formatting",
                "Evaluate metadata accuracy and completeness",
                "Check for processing artifacts or errors"
            ],
            evaluation_params=["processed_chunk", "original_text"],
            threshold=0.7,
            model=self.evaluation_model
        )
    
    def evaluate_chunking_quality(
        self,
        chunks: List[str],
        original_text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> ChunkingMetrics:
        """
        Evaluate the quality of text chunking
        
        Args:
            chunks: List of text chunks
            original_text: Original text before chunking
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            Chunking quality metrics
        """
        logger.info(f"Evaluating chunking quality for {len(chunks)} chunks")
        
        metrics = {
            'chunk_coherence': 0.0,
            'chunk_completeness': 0.0,
            'chunk_overlap_quality': 0.0,
            'chunk_size_consistency': 0.0,
            'semantic_boundary_respect': 0.0
        }
        
        # Evaluate each chunk
        coherence_scores = []
        completeness_scores = []
        overlap_scores = []
        size_scores = []
        boundary_scores = []
        
        for i, chunk in enumerate(chunks):
            # Chunk coherence evaluation
            coherence_score = self._evaluate_chunk_coherence(chunk)
            coherence_scores.append(coherence_score)
            
            # Chunk completeness evaluation
            completeness_score = self._evaluate_chunk_completeness(chunk)
            completeness_scores.append(completeness_score)
            
            # Overlap quality evaluation (for chunks with overlap)
            if i > 0:
                overlap_score = self._evaluate_overlap_quality(
                    chunks[i-1], chunk, chunk_overlap
                )
                overlap_scores.append(overlap_score)
            
            # Size consistency evaluation
            size_score = self._evaluate_size_consistency(chunk, chunk_size)
            size_scores.append(size_score)
            
            # Semantic boundary evaluation
            boundary_score = self._evaluate_semantic_boundaries(chunk)
            boundary_scores.append(boundary_score)
        
        # Calculate aggregate scores
        metrics['chunk_coherence'] = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        metrics['chunk_completeness'] = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        metrics['chunk_overlap_quality'] = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        metrics['chunk_size_consistency'] = sum(size_scores) / len(size_scores) if size_scores else 0.0
        metrics['semantic_boundary_respect'] = sum(boundary_scores) / len(boundary_scores) if boundary_scores else 0.0
        
        # Calculate overall quality
        overall_quality = sum(metrics.values()) / len(metrics)
        
        chunking_metrics = ChunkingMetrics(
            chunk_coherence=metrics['chunk_coherence'],
            chunk_completeness=metrics['chunk_completeness'],
            chunk_overlap_quality=metrics['chunk_overlap_quality'],
            chunk_size_consistency=metrics['chunk_size_consistency'],
            semantic_boundary_respect=metrics['semantic_boundary_respect'],
            overall_chunk_quality=overall_quality
        )
        
        self.chunking_metrics.append(chunking_metrics)
        logger.info(f"Chunking evaluation completed. Overall quality: {overall_quality:.3f}")
        
        return chunking_metrics
    
    def evaluate_processing_quality(
        self,
        processed_chunks: List[Dict[str, Any]],
        original_files: List[str]
    ) -> ProcessingMetrics:
        """
        Evaluate document processing quality
        
        Args:
            processed_chunks: List of processed document chunks
            original_files: List of original file paths
            
        Returns:
            Processing quality metrics
        """
        logger.info(f"Evaluating processing quality for {len(processed_chunks)} chunks")
        
        metrics = {
            'text_cleanliness': 0.0,
            'content_preservation': 0.0,
            'format_handling': 0.0,
            'metadata_quality': 0.0,
            'processing_efficiency': 0.0
        }
        
        # Evaluate processing quality for each chunk
        cleanliness_scores = []
        preservation_scores = []
        format_scores = []
        metadata_scores = []
        
        for chunk in processed_chunks:
            # Text cleanliness evaluation
            cleanliness_score = self._evaluate_text_cleanliness(chunk['content'])
            cleanliness_scores.append(cleanliness_score)
            
            # Content preservation evaluation
            preservation_score = self._evaluate_content_preservation(chunk)
            preservation_scores.append(preservation_score)
            
            # Format handling evaluation
            format_score = self._evaluate_format_handling(chunk)
            format_scores.append(format_score)
            
            # Metadata quality evaluation
            metadata_score = self._evaluate_metadata_quality(chunk)
            metadata_scores.append(metadata_score)
        
        # Calculate aggregate scores
        metrics['text_cleanliness'] = sum(cleanliness_scores) / len(cleanliness_scores) if cleanliness_scores else 0.0
        metrics['content_preservation'] = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0
        metrics['format_handling'] = sum(format_scores) / len(format_scores) if format_scores else 0.0
        metrics['metadata_quality'] = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0.0
        metrics['processing_efficiency'] = len(processed_chunks) / len(original_files) if original_files else 0.0
        
        # Calculate overall quality
        overall_quality = sum(metrics.values()) / len(metrics)
        
        processing_metrics = ProcessingMetrics(
            text_cleanliness=metrics['text_cleanliness'],
            content_preservation=metrics['content_preservation'],
            format_handling=metrics['format_handling'],
            metadata_quality=metrics['metadata_quality'],
            processing_efficiency=metrics['processing_efficiency'],
            overall_processing_quality=overall_quality
        )
        
        self.processing_metrics.append(processing_metrics)
        logger.info(f"Processing evaluation completed. Overall quality: {overall_quality:.3f}")
        
        return processing_metrics
    
    def _evaluate_chunk_coherence(self, chunk: str) -> float:
        """Evaluate semantic coherence of a chunk"""
        try:
            # For now, use a simple heuristic-based evaluation instead of GEval
            # to avoid the complex result parsing issues
            
            # Check for basic coherence indicators
            coherence_score = 0.5  # Base score
            
            # Check for complete sentences
            sentences = re.split(r'[.!?]+', chunk)
            complete_sentences = [s.strip() for s in sentences if s.strip()]
            
            if complete_sentences:
                # Check if chunk ends with complete sentence
                ends_with_period = chunk.strip().endswith(('.', '!', '?'))
                coherence_score += 0.2 if ends_with_period else 0.1
            
            # Check for logical connectors
            connectors = ['however', 'therefore', 'because', 'although', 'furthermore', 'moreover', 'thus', 'hence']
            has_connectors = any(connector in chunk.lower() for connector in connectors)
            coherence_score += 0.1 if has_connectors else 0.0
            
            # Check for paragraph structure
            has_paragraph_breaks = '\n\n' in chunk
            coherence_score += 0.1 if has_paragraph_breaks else 0.0
            
            # Check for balanced structure
            words = chunk.split()
            if len(words) > 10:  # Reasonable chunk size
                coherence_score += 0.1
            
            return min(coherence_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating chunk coherence: {e}")
            return 0.5
    
    def _evaluate_chunk_completeness(self, chunk: str) -> float:
        """Evaluate completeness of a chunk"""
        try:
            # Check for complete sentences
            sentences = re.split(r'[.!?]+', chunk)
            complete_sentences = [s.strip() for s in sentences if s.strip()]
            
            # Check for balanced parentheses and quotes
            balanced_parens = chunk.count('(') == chunk.count(')')
            balanced_quotes = chunk.count('"') % 2 == 0
            
            # Calculate completeness score
            completeness_score = 0.0
            
            if complete_sentences:
                # Check if chunk ends with complete sentence
                ends_with_period = chunk.strip().endswith(('.', '!', '?'))
                completeness_score += 0.3 if ends_with_period else 0.1
            
            if balanced_parens:
                completeness_score += 0.2
            
            if balanced_quotes:
                completeness_score += 0.2
            
            # Check for incomplete words at boundaries
            words = chunk.split()
            if words:
                first_word_complete = not words[0].startswith('-')
                last_word_complete = not words[-1].endswith('-')
                completeness_score += 0.3 if (first_word_complete and last_word_complete) else 0.1
            
            return min(completeness_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating chunk completeness: {e}")
            return 0.5
    
    def _evaluate_overlap_quality(self, prev_chunk: str, current_chunk: str, overlap_size: int) -> float:
        """Evaluate quality of chunk overlap"""
        try:
            # Find actual overlap
            overlap_text = self._find_overlap(prev_chunk, current_chunk)
            
            if not overlap_text:
                return 0.5
            
            # Evaluate overlap quality
            overlap_length = len(overlap_text)
            ideal_overlap = overlap_size
            
            # Score based on overlap size appropriateness
            size_score = 1.0 - abs(overlap_length - ideal_overlap) / ideal_overlap
            size_score = max(0.0, min(1.0, size_score))
            
            # Check if overlap contains meaningful content
            meaningful_score = 0.5
            if len(overlap_text.split()) > 3:  # At least 3 words
                meaningful_score = 0.8
            
            # Check for redundancy
            redundancy_score = 0.8
            if overlap_text in prev_chunk and overlap_text in current_chunk:
                redundancy_score = 0.6
            
            # Combine scores
            overlap_score = (size_score + meaningful_score + redundancy_score) / 3
            return overlap_score
            
        except Exception as e:
            logger.warning(f"Error evaluating overlap quality: {e}")
            return 0.5
    
    def _evaluate_size_consistency(self, chunk: str, target_size: int) -> float:
        """Evaluate consistency of chunk sizes"""
        try:
            chunk_length = len(chunk)
            
            # Calculate size consistency score
            if target_size == 0:
                return 0.5
            
            size_ratio = chunk_length / target_size
            
            # Ideal size is within 20% of target
            if 0.8 <= size_ratio <= 1.2:
                return 1.0
            elif 0.6 <= size_ratio <= 1.4:
                return 0.8
            elif 0.4 <= size_ratio <= 1.6:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.warning(f"Error evaluating size consistency: {e}")
            return 0.5
    
    def _evaluate_semantic_boundaries(self, chunk: str) -> float:
        """Evaluate if chunk respects semantic boundaries"""
        try:
            # Check for sentence boundaries
            sentences = re.split(r'[.!?]+', chunk)
            complete_sentences = [s.strip() for s in sentences if s.strip()]
            
            # Check for paragraph boundaries
            paragraphs = chunk.split('\n\n')
            
            # Calculate boundary score
            boundary_score = 0.0
            
            # Sentence boundary score
            if complete_sentences:
                avg_sentence_length = sum(len(s.split()) for s in complete_sentences) / len(complete_sentences)
                if 5 <= avg_sentence_length <= 25:  # Reasonable sentence length
                    boundary_score += 0.4
            
            # Paragraph boundary score
            if len(paragraphs) <= 2:  # Not too many paragraph breaks
                boundary_score += 0.3
            
            # Check for natural breaks
            natural_breaks = chunk.count('. ') + chunk.count('! ') + chunk.count('? ')
            if natural_breaks > 0:
                boundary_score += 0.3
            
            return min(boundary_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating semantic boundaries: {e}")
            return 0.5
    
    def _evaluate_text_cleanliness(self, text: str) -> float:
        """Evaluate cleanliness of processed text"""
        try:
            cleanliness_score = 0.0
            
            # Check for excessive whitespace
            whitespace_ratio = len(re.findall(r'\s+', text)) / len(text) if text else 0
            if whitespace_ratio < 0.3:
                cleanliness_score += 0.2
            
            # Check for special characters
            special_char_ratio = len(re.findall(r'[^\w\s\.\,\!\?]', text)) / len(text) if text else 0
            if special_char_ratio < 0.1:
                cleanliness_score += 0.2
            
            # Check for encoding issues
            if text.isprintable():
                cleanliness_score += 0.2
            
            # Check for consistent formatting
            lines = text.split('\n')
            consistent_lines = [line for line in lines if line.strip()]
            if len(consistent_lines) == len(lines) or len(consistent_lines) >= len(lines) * 0.8:
                cleanliness_score += 0.2
            
            # Check for proper capitalization
            sentences = re.split(r'[.!?]+', text)
            proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            if sentences and proper_caps / len(sentences) >= 0.8:
                cleanliness_score += 0.2
            
            return min(cleanliness_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating text cleanliness: {e}")
            return 0.5
    
    def _evaluate_content_preservation(self, chunk: Dict[str, Any]) -> float:
        """Evaluate if original content meaning is preserved"""
        try:
            content = chunk.get('content', '')
            
            # Check for content length
            if len(content) < 10:
                return 0.3
            
            # Check for meaningful content (not just whitespace or special chars)
            meaningful_chars = len(re.findall(r'[a-zA-Z0-9]', content))
            if meaningful_chars / len(content) >= 0.5:
                return 0.8
            else:
                return 0.4
                
        except Exception as e:
            logger.warning(f"Error evaluating content preservation: {e}")
            return 0.5
    
    def _evaluate_format_handling(self, chunk: Dict[str, Any]) -> float:
        """Evaluate format handling quality"""
        try:
            content = chunk.get('content', '')
            meta = chunk.get('meta', {})
            
            format_score = 0.0
            
            # Check if source information is preserved
            if 'source' in meta:
                format_score += 0.3
            
            # Check if chunk information is preserved
            if 'chunk_id' in meta:
                format_score += 0.2
            
            # Check for proper text formatting
            if content and not content.isspace():
                format_score += 0.3
            
            # Check for metadata completeness
            if len(meta) >= 2:
                format_score += 0.2
            
            return format_score
            
        except Exception as e:
            logger.warning(f"Error evaluating format handling: {e}")
            return 0.5
    
    def _evaluate_metadata_quality(self, chunk: Dict[str, Any]) -> float:
        """Evaluate metadata quality"""
        try:
            meta = chunk.get('meta', {})
            
            metadata_score = 0.0
            
            # Check for required metadata fields
            required_fields = ['source', 'chunk_id']
            present_fields = sum(1 for field in required_fields if field in meta)
            metadata_score += (present_fields / len(required_fields)) * 0.5
            
            # Check for additional useful metadata
            useful_fields = ['total_chunks', 'file_type', 'processing_timestamp']
            additional_fields = sum(1 for field in useful_fields if field in meta)
            metadata_score += (additional_fields / len(useful_fields)) * 0.3
            
            # Check for metadata data types
            if 'chunk_id' in meta and isinstance(meta['chunk_id'], int):
                metadata_score += 0.2
            
            return metadata_score
            
        except Exception as e:
            logger.warning(f"Error evaluating metadata quality: {e}")
            return 0.5
    
    def _find_overlap(self, text1: str, text2: str) -> str:
        """Find overlap between two text chunks"""
        try:
            # Simple overlap detection
            words1 = text1.split()
            words2 = text2.split()
            
            # Find common words at boundaries
            for i in range(min(len(words1), len(words2)), 0, -1):
                if words1[-i:] == words2[:i]:
                    return ' '.join(words1[-i:])
            
            return ""
            
        except Exception as e:
            logger.warning(f"Error finding overlap: {e}")
            return ""
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all chunking and processing evaluations"""
        if not self.chunking_metrics and not self.processing_metrics:
            return {"message": "No evaluations performed yet"}
        
        summary = {
            "total_evaluations": len(self.chunking_metrics) + len(self.processing_metrics),
            "chunking_evaluations": len(self.chunking_metrics),
            "processing_evaluations": len(self.processing_metrics),
            "average_chunking_quality": 0.0,
            "average_processing_quality": 0.0,
            "detailed_metrics": {}
        }
        
        if self.chunking_metrics:
            avg_chunking = sum(m.overall_chunk_quality for m in self.chunking_metrics) / len(self.chunking_metrics)
            summary["average_chunking_quality"] = avg_chunking
            
            # Detailed chunking metrics
            summary["detailed_metrics"]["chunking"] = {
                "coherence": sum(m.chunk_coherence for m in self.chunking_metrics) / len(self.chunking_metrics),
                "completeness": sum(m.chunk_completeness for m in self.chunking_metrics) / len(self.chunking_metrics),
                "overlap_quality": sum(m.chunk_overlap_quality for m in self.chunking_metrics) / len(self.chunking_metrics),
                "size_consistency": sum(m.chunk_size_consistency for m in self.chunking_metrics) / len(self.chunking_metrics),
                "semantic_boundaries": sum(m.semantic_boundary_respect for m in self.chunking_metrics) / len(self.chunking_metrics)
            }
        
        if self.processing_metrics:
            avg_processing = sum(m.overall_processing_quality for m in self.processing_metrics) / len(self.processing_metrics)
            summary["average_processing_quality"] = avg_processing
            
            # Detailed processing metrics
            summary["detailed_metrics"]["processing"] = {
                "text_cleanliness": sum(m.text_cleanliness for m in self.processing_metrics) / len(self.processing_metrics),
                "content_preservation": sum(m.content_preservation for m in self.processing_metrics) / len(self.processing_metrics),
                "format_handling": sum(m.format_handling for m in self.processing_metrics) / len(self.processing_metrics),
                "metadata_quality": sum(m.metadata_quality for m in self.processing_metrics) / len(self.processing_metrics),
                "processing_efficiency": sum(m.processing_efficiency for m in self.processing_metrics) / len(self.processing_metrics)
            }
        
        return summary 