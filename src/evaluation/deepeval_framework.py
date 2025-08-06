"""
Official DeepEval RAG Evaluation Framework
Implements component-level evaluation following DeepEval's methodology
"""

import os
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval
)
from deepeval import evaluate

from .metrics import RetrieverMetrics, GeneratorMetrics, EvaluationResults
from .test_cases import TestCaseGenerator, RAGTestCase
from .synthetic_data import SyntheticDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for DeepEval framework"""
    model: str = "gpt-4"
    max_test_cases_per_run: int = 100
    timeout_per_test_case: int = 60
    parallel_execution: bool = True
    max_workers: int = 4
    retriever_threshold: float = 0.7
    generator_threshold: float = 0.7
    save_results: bool = True
    output_format: List[str] = None
    
    def __post_init__(self):
        if self.output_format is None:
            self.output_format = ["html", "json", "csv"]


class DeepEvalFramework:
    """
    Production-ready DeepEval evaluation framework
    
    Features:
    - Component-level evaluation (retriever + generator)
    - 14+ evaluation metrics
    - Synthetic data generation
    - Batch evaluation processing
    - Comprehensive reporting
    - CI/CD integration ready
    """
    
    def __init__(self, config_path: str = "config/eval_config.yaml"):
        """
        Initialize DeepEval framework
        
        Args:
            config_path: Path to evaluation configuration file
        """
        self.config = self._load_config(config_path)
        self.eval_config = EvaluationConfig(**self.config.get('evaluation', {}))
        
        self._setup_logging()
        self._initialize_metrics()
        self._setup_environment()
        
        # Results storage
        self.evaluation_results: List[EvaluationResults] = []
        
        logger.info("DeepEval framework initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Evaluation config loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging for evaluation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/deepeval.log'),
                logging.StreamHandler()
            ]
        )
    
    def _setup_environment(self):
        """Setup environment variables for DeepEval"""
        # Set OpenAI API key if available for evaluation metrics
        if 'OPENAI_API_KEY' in os.environ:
            logger.info("OpenAI API key found for evaluation metrics")
        else:
            logger.warning("OpenAI API key not set - some metrics may not work")
        
        # Set DeepEval model
        os.environ.setdefault('DEEPEVAL_MODEL', self.eval_config.model)
    
    def _initialize_metrics(self):
        """Initialize all evaluation metrics"""
        # Retriever Metrics (Official DeepEval)
        self.retriever_metrics = RetrieverMetrics(
            config=self.config.get('retriever_metrics', {}),
            model=self.eval_config.model
        )
        
        # Generator Metrics (Custom GEval)
        self.generator_metrics = GeneratorMetrics(
            config=self.config.get('generator_metrics', {}),
            model=self.eval_config.model
        )
        
        logger.info("All evaluation metrics initialized")
    
    def create_test_cases_from_rag_responses(
        self,
        rag_responses: List[Dict[str, Any]],
        expected_answers: Optional[List[str]] = None
    ) -> List[LLMTestCase]:
        """
        Create DeepEval test cases from RAG system responses
        
        Args:
            rag_responses: List of RAG system responses
            expected_answers: Optional expected answers for comparison
            
        Returns:
            List of LLMTestCase objects
        """
        test_cases = []
        
        for i, response in enumerate(rag_responses):
            try:
                # Extract components from RAG response
                query = response.get('query', '')
                actual_output = response.get('answer', '')
                retrieval_context = response.get('retrieved_contexts', [])
                
                # Use expected answer if provided, otherwise None
                expected_output = expected_answers[i] if expected_answers and i < len(expected_answers) else None
                
                # Create test case
                test_case = LLMTestCase(
                    input=query,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context,
                    expected_output=expected_output
                )
                
                test_cases.append(test_case)
                
            except Exception as e:
                logger.error(f"Error creating test case {i}: {e}")
        
        logger.info(f"Created {len(test_cases)} test cases from RAG responses")
        return test_cases
    
    def evaluate_retriever_component(
        self,
        test_cases: List[LLMTestCase],
        batch_size: int = None
    ) -> EvaluationResults:
        """
        Evaluate retriever component using official DeepEval metrics
        
        Args:
            test_cases: List of test cases to evaluate
            batch_size: Optional batch size for processing
            
        Returns:
            EvaluationResults with retriever metrics
        """
        logger.info(f"Starting retriever evaluation with {len(test_cases)} test cases")
        start_time = time.time()
        
        if batch_size is None:
            batch_size = self.eval_config.max_test_cases_per_run
        
        try:
            # Get retriever metrics
            metrics = self.retriever_metrics.get_all_metrics()
            
            # Run evaluation
            results = []
            for i in range(0, len(test_cases), batch_size):
                batch = test_cases[i:i + batch_size]
                logger.info(f"Evaluating retriever batch {i//batch_size + 1}")
                
                batch_results = evaluate(batch, metrics)
                results.extend(batch_results)
            
            # Calculate aggregate scores
            aggregate_scores = self._calculate_aggregate_scores(results, metrics)
            
            evaluation_time = time.time() - start_time
            
            eval_results = EvaluationResults(
                component="retriever",
                test_cases_count=len(test_cases),
                aggregate_scores=aggregate_scores,
                individual_results=results,
                execution_time=evaluation_time,
                timestamp=time.time(),
                config=asdict(self.eval_config)
            )
            
            logger.info(f"Retriever evaluation completed in {evaluation_time:.2f}s")
            return eval_results
            
        except Exception as e:
            logger.error(f"Retriever evaluation failed: {e}")
            return EvaluationResults(
                component="retriever",
                test_cases_count=len(test_cases),
                aggregate_scores={},
                individual_results=[],
                execution_time=time.time() - start_time,
                timestamp=time.time(),
                config=asdict(self.eval_config),
                error=str(e)
            )
    
    def evaluate_generator_component(
        self,
        test_cases: List[LLMTestCase],
        batch_size: int = None
    ) -> EvaluationResults:
        """
        Evaluate generator component using custom GEval metrics
        
        Args:
            test_cases: List of test cases to evaluate
            batch_size: Optional batch size for processing
            
        Returns:
            EvaluationResults with generator metrics
        """
        logger.info(f"Starting generator evaluation with {len(test_cases)} test cases")
        start_time = time.time()
        
        if batch_size is None:
            batch_size = self.eval_config.max_test_cases_per_run
        
        try:
            # Get generator metrics
            metrics = self.generator_metrics.get_all_metrics()
            
            # Run evaluation
            results = []
            for i in range(0, len(test_cases), batch_size):
                batch = test_cases[i:i + batch_size]
                logger.info(f"Evaluating generator batch {i//batch_size + 1}")
                
                batch_results = evaluate(batch, metrics)
                results.extend(batch_results)
            
            # Calculate aggregate scores
            aggregate_scores = self._calculate_aggregate_scores(results, metrics)
            
            evaluation_time = time.time() - start_time
            
            eval_results = EvaluationResults(
                component="generator",
                test_cases_count=len(test_cases),
                aggregate_scores=aggregate_scores,
                individual_results=results,
                execution_time=evaluation_time,
                timestamp=time.time(),
                config=asdict(self.eval_config)
            )
            
            logger.info(f"Generator evaluation completed in {evaluation_time:.2f}s")
            return eval_results
            
        except Exception as e:
            logger.error(f"Generator evaluation failed: {e}")
            return EvaluationResults(
                component="generator",
                test_cases_count=len(test_cases),
                aggregate_scores={},
                individual_results=[],
                execution_time=time.time() - start_time,
                timestamp=time.time(),
                config=asdict(self.eval_config),
                error=str(e)
            )
    
    def evaluate_complete_rag_system(
        self,
        test_cases: List[LLMTestCase],
        batch_size: int = None
    ) -> Dict[str, EvaluationResults]:
        """
        Evaluate complete RAG system with both components
        
        Args:
            test_cases: List of test cases to evaluate
            batch_size: Optional batch size for processing
            
        Returns:
            Dictionary with results for both components
        """
        logger.info("Starting complete RAG system evaluation")
        
        results = {}
        
        # Evaluate retriever component
        logger.info("Evaluating retriever component...")
        retriever_results = self.evaluate_retriever_component(test_cases, batch_size)
        results['retriever'] = retriever_results
        
        # Evaluate generator component
        logger.info("Evaluating generator component...")
        generator_results = self.evaluate_generator_component(test_cases, batch_size)
        results['generator'] = generator_results
        
        # Store results
        self.evaluation_results.extend([retriever_results, generator_results])
        
        # Save results if configured
        if self.eval_config.save_results:
            self._save_evaluation_results(results)
        
        logger.info("Complete RAG system evaluation finished")
        return results
    
    def _calculate_aggregate_scores(
        self,
        results: List[Any],
        metrics: List[Any]
    ) -> Dict[str, float]:
        """Calculate aggregate scores from individual results"""
        aggregate_scores = {}
        
        try:
            for metric in metrics:
                metric_name = metric.__class__.__name__
                scores = []
                
                for result in results:
                    if hasattr(result, 'metrics_metadata'):
                        for metric_result in result.metrics_metadata:
                            if metric_result.metric == metric_name:
                                scores.append(metric_result.score)
                
                if scores:
                    aggregate_scores[metric_name] = {
                        'average': sum(scores) / len(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'count': len(scores)
                    }
                else:
                    aggregate_scores[metric_name] = {
                        'average': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0
                    }
        
        except Exception as e:
            logger.error(f"Error calculating aggregate scores: {e}")
        
        return aggregate_scores
    
    def _save_evaluation_results(self, results: Dict[str, EvaluationResults]):
        """Save evaluation results to files"""
        try:
            timestamp = int(time.time())
            
            for component, eval_results in results.items():
                # JSON format
                if 'json' in self.eval_config.output_format:
                    json_path = f"reports/eval_{component}_{timestamp}.json"
                    eval_results.save_to_json(json_path)
                
                # HTML format
                if 'html' in self.eval_config.output_format:
                    html_path = f"reports/eval_{component}_{timestamp}.html"
                    eval_results.save_to_html(html_path)
                
                # CSV format
                if 'csv' in self.eval_config.output_format:
                    csv_path = f"reports/eval_{component}_{timestamp}.csv"
                    eval_results.save_to_csv(csv_path)
            
            logger.info("Evaluation results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_synthetic_test_data(
        self,
        documents: List[str],
        num_samples: int = 100,
        domains: List[str] = None
    ) -> List[LLMTestCase]:
        """
        Generate synthetic test data using DeepEval Synthesizer
        
        Args:
            documents: List of document contents
            num_samples: Number of synthetic samples to generate
            domains: Domains for synthetic data generation
            
        Returns:
            List of synthetic test cases
        """
        logger.info(f"Generating {num_samples} synthetic test cases")
        
        try:
            generator = SyntheticDataGenerator()
            test_cases = generator.generate_from_documents(
                documents=documents,
                num_samples=num_samples,
                domains=domains or ["general"]
            )
            
            logger.info(f"Generated {len(test_cases)} synthetic test cases")
            return test_cases
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return []
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed"""
        if not self.evaluation_results:
            return {"message": "No evaluations performed yet"}
        
        retriever_results = [r for r in self.evaluation_results if r.component == "retriever"]
        generator_results = [r for r in self.evaluation_results if r.component == "generator"]
        
        summary = {
            "total_evaluations": len(self.evaluation_results),
            "retriever_evaluations": len(retriever_results),
            "generator_evaluations": len(generator_results),
            "total_test_cases": sum(r.test_cases_count for r in self.evaluation_results),
            "total_execution_time": sum(r.execution_time for r in self.evaluation_results),
            "latest_results": {}
        }
        
        # Latest results for each component
        if retriever_results:
            latest_retriever = max(retriever_results, key=lambda x: x.timestamp)
            summary["latest_results"]["retriever"] = latest_retriever.aggregate_scores
        
        if generator_results:
            latest_generator = max(generator_results, key=lambda x: x.timestamp)
            summary["latest_results"]["generator"] = latest_generator.aggregate_scores
        
        return summary 