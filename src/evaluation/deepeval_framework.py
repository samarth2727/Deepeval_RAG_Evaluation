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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Continue without dotenv if not installed

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval
)
from deepeval import evaluate

# Use absolute imports instead of relative imports
from src.evaluation.metrics import RetrieverMetrics, GeneratorMetrics, EvaluationResults
from src.evaluation.test_cases import TestCaseGenerator, RAGTestCase
from src.evaluation.synthetic_data import SyntheticDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for DeepEval framework"""
    model: str = "gpt-4o-mini"  # Changed from "gpt-4" to valid model
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
    - 13 specialized evaluation metrics
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
            # Create a mapping of metric names
            metric_names = [metric.__class__.__name__ for metric in metrics]
            logger.info(f"Looking for metrics: {metric_names}")
            
            # Create a mapping from DeepEval metric names to our metric names
            metric_mapping = {
                'Answer Correctness [GEval]': 'GEval',
                'Citation Accuracy [GEval]': 'GEval',
                'Contextual Relevancy': 'ContextualRelevancyMetric',
                'Contextual Recall': 'ContextualRecallMetric',
                'Contextual Precision': 'ContextualPrecisionMetric'
            }
            
            for metric in metrics:
                metric_name = metric.__class__.__name__
                scores = []
                
                for result in results:
                    logger.info(f"Processing result: {type(result)}")
                    
                    # Handle different result types
                    if hasattr(result, 'metrics_metadata'):
                        # Object with metrics_metadata
                        logger.info(f"Result has metrics_metadata: {len(result.metrics_metadata)} items")
                        for i, metric_result in enumerate(result.metrics_metadata):
                            logger.info(f"Metric {i}: {metric_result.metric} (type: {type(metric_result.metric)})")
                            if hasattr(metric_result, 'score'):
                                logger.info(f"  Score: {metric_result.score}")
                            else:
                                logger.info(f"  No score attribute")
                            
                            # Check if this metric result matches our metric
                            if (metric_result.metric == metric_name or 
                                metric_result.metric in metric_mapping and metric_mapping[metric_result.metric] == metric_name or
                                metric_result.metric.startswith(metric_name) or
                                metric_name in metric_result.metric):
                                if hasattr(metric_result, 'score') and metric_result.score is not None:
                                    scores.append(metric_result.score)
                                    logger.info(f"Found score for {metric_name} (from {metric_result.metric}): {metric_result.score}")
                    elif isinstance(result, tuple):
                        # Tuple result from DeepEval
                        logger.info(f"Processing tuple result with {len(result)} items")
                        for i, item in enumerate(result):
                            logger.info(f"Tuple item {i}: {type(item)} - {item}")
                            
                            # Handle TestResult objects with metrics_data
                            if hasattr(item, 'metrics_data') and item.metrics_data:
                                logger.info(f"Found TestResult with {len(item.metrics_data)} metrics")
                                for metric_data in item.metrics_data:
                                    if hasattr(metric_data, 'name') and hasattr(metric_data, 'score'):
                                        metric_name_from_data = metric_data.name
                                        metric_score = metric_data.score
                                        logger.info(f"Found metric: {metric_name_from_data} = {metric_score}")
                                        
                                        # Check if this matches our metric
                                        if (metric_name_from_data == metric_name or 
                                            metric_name_from_data in metric_mapping and metric_mapping[metric_name_from_data] == metric_name or
                                            metric_name_from_data.startswith(metric_name) or
                                            metric_name in metric_name_from_data):
                                            scores.append(metric_score)
                                            logger.info(f"Found score for {metric_name} (from {metric_name_from_data}): {metric_score}")
                            
                            # Handle direct score attributes
                            elif hasattr(item, 'score') and item.score is not None:
                                # Extract metric name from the item
                                metric_name_from_item = getattr(item, 'metric', str(type(item)))
                                logger.info(f"Found score in tuple item: {metric_name_from_item} = {item.score}")
                                
                                # Check if this matches our metric
                                if (metric_name_from_item == metric_name or 
                                    metric_name_from_item in metric_mapping and metric_mapping[metric_name_from_item] == metric_name or
                                    metric_name_from_item.startswith(metric_name) or
                                    metric_name in metric_name_from_item):
                                    scores.append(item.score)
                                    logger.info(f"Found score for {metric_name} (from {metric_name_from_item}): {item.score}")
                    else:
                        logger.info(f"Result does not have metrics_metadata attribute and is not a tuple")
                
                if scores:
                    aggregate_scores[metric_name] = {
                        'average': sum(scores) / len(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'count': len(scores)
                    }
                    logger.info(f"Aggregate scores for {metric_name}: {aggregate_scores[metric_name]}")
                else:
                    aggregate_scores[metric_name] = {
                        'average': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0
                    }
                    logger.warning(f"No scores found for metric: {metric_name}")
        
        except Exception as e:
            logger.error(f"Error calculating aggregate scores: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
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