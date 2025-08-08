"""
Metrics Management for RAG Evaluation
Implements both official DeepEval metrics and custom GEval metrics
"""

import json
import csv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    component: str
    test_cases_count: int
    aggregate_scores: Dict[str, Any]
    individual_results: List[Any]
    execution_time: float
    timestamp: float
    config: Dict[str, Any]
    error: Optional[str] = None
    
    def save_to_json(self, file_path: str):
        """Save results to JSON file"""
        try:
            # Convert results to serializable format
            serializable_data = asdict(self)
            # Remove individual_results as they may not be serializable
            serializable_data['individual_results'] = len(self.individual_results)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Results saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def save_to_html(self, file_path: str):
        """Save results to HTML report"""
        try:
            html_content = self._generate_html_report()
            
            with open(file_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to HTML: {e}")
    
    def save_to_csv(self, file_path: str):
        """Save aggregate scores to CSV"""
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Average', 'Min', 'Max', 'Count'])
                
                for metric, scores in self.aggregate_scores.items():
                    if isinstance(scores, dict):
                        writer.writerow([
                            metric,
                            scores.get('average', 0),
                            scores.get('min', 0),
                            scores.get('max', 0),
                            scores.get('count', 0)
                        ])
            
            logger.info(f"CSV report saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report - {self.component.title()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .success {{ background-color: #d4edda; }}
                .warning {{ background-color: #fff3cd; }}
                .error {{ background-color: #f8d7da; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RAG Evaluation Report</h1>
                <h2>Component: {self.component.title()}</h2>
                <p><strong>Test Cases:</strong> {self.test_cases_count}</p>
                <p><strong>Execution Time:</strong> {self.execution_time:.2f} seconds</p>
                <p><strong>Timestamp:</strong> {self.timestamp}</p>
            </div>
            
            <h3>Aggregate Scores</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Average</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Count</th>
                </tr>
        """
        
        for metric, scores in self.aggregate_scores.items():
            if isinstance(scores, dict):
                avg_score = scores.get('average', 0)
                css_class = 'success' if avg_score >= 0.7 else 'warning' if avg_score >= 0.5 else 'error'
                
                html += f"""
                <tr class="{css_class}">
                    <td>{metric}</td>
                    <td>{avg_score:.3f}</td>
                    <td>{scores.get('min', 0):.3f}</td>
                    <td>{scores.get('max', 0):.3f}</td>
                    <td>{scores.get('count', 0)}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h3>Configuration</h3>
            <pre>{}</pre>
        </body>
        </html>
        """.format(json.dumps(self.config, indent=2))
        
        return html


class RetrieverMetrics:
    """
    Official DeepEval retriever metrics
    Implements contextual relevancy, recall, and precision
    """
    
    def __init__(self, config: Dict[str, Any], model: str = "gpt-4"):
        """
        Initialize retriever metrics
        
        Args:
            config: Configuration for retriever metrics
            model: Model to use for evaluation
        """
        self.config = config
        self.model = model
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all retriever metrics"""
        # Contextual Relevancy Metric
        relevancy_config = self.config.get('contextual_relevancy', {})
        self.contextual_relevancy = ContextualRelevancyMetric(
            threshold=relevancy_config.get('threshold', 0.7),
            model=relevancy_config.get('model', self.model),
            include_reason=True
        )
        
        # Contextual Recall Metric
        recall_config = self.config.get('contextual_recall', {})
        self.contextual_recall = ContextualRecallMetric(
            threshold=recall_config.get('threshold', 0.7),
            model=recall_config.get('model', self.model),
            include_reason=True
        )
        
        # Contextual Precision Metric
        precision_config = self.config.get('contextual_precision', {})
        self.contextual_precision = ContextualPrecisionMetric(
            threshold=precision_config.get('threshold', 0.7),
            model=precision_config.get('model', self.model),
            include_reason=True
        )
        
        logger.info("Retriever metrics initialized")
    
    def get_all_metrics(self) -> List[Any]:
        """Get list of all retriever metrics"""
        metrics = []
        
        if self.config.get('contextual_relevancy', {}).get('enabled', True):
            metrics.append(self.contextual_relevancy)
        
        if self.config.get('contextual_recall', {}).get('enabled', True):
            metrics.append(self.contextual_recall)
        
        if self.config.get('contextual_precision', {}).get('enabled', True):
            metrics.append(self.contextual_precision)
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all retriever metrics"""
        return {
            "ContextualRelevancyMetric": "Measures how relevant the retrieved context is to the given query",
            "ContextualRecallMetric": "Measures whether the retrieved context contains enough information to answer the query",
            "ContextualPrecisionMetric": "Measures whether the retrieved context is precise without unnecessary information"
        }


class GeneratorMetrics:
    """
    Custom GEval generator metrics
    Implements answer correctness and citation accuracy
    """
    
    def __init__(self, config: Dict[str, Any], model: str = "gpt-4"):
        """
        Initialize generator metrics
        
        Args:
            config: Configuration for generator metrics
            model: Model to use for evaluation
        """
        self.config = config
        self.model = model
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all generator metrics"""
        # Answer Correctness Metric
        correctness_config = self.config.get('answer_correctness', {})
        self.answer_correctness = GEval(
            name="Answer Correctness",
            criteria=correctness_config.get(
                'evaluation_criteria',
                "Evaluate if the actual output's answer is correct and complete from the input and retrieved context"
            ),
            evaluation_steps=correctness_config.get('evaluation_steps', [
                "Check if the answer directly addresses the question",
                "Verify factual accuracy against the retrieved context",
                "Assess completeness of the response"
            ]),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            threshold=correctness_config.get('threshold', 0.7),
            model=correctness_config.get('model', self.model)
        )
        
        # Citation Accuracy Metric
        citation_config = self.config.get('citation_accuracy', {})
        self.citation_accuracy = GEval(
            name="Citation Accuracy",
            criteria=citation_config.get(
                'evaluation_criteria',
                "Check if citations are correct and relevant based on input and retrieved context"
            ),
            evaluation_steps=citation_config.get('evaluation_steps', [
                "Verify that cited sources exist in the retrieved context",
                "Check if citations are properly formatted",
                "Assess relevance of cited content to the answer"
            ]),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            threshold=citation_config.get('threshold', 0.8),
            model=citation_config.get('model', self.model)
        )
        
        logger.info("Generator metrics initialized")
    
    def get_all_metrics(self) -> List[Any]:
        """Get list of all generator metrics"""
        metrics = []
        
        if self.config.get('answer_correctness', {}).get('enabled', True):
            metrics.append(self.answer_correctness)
        
        if self.config.get('citation_accuracy', {}).get('enabled', True):
            metrics.append(self.citation_accuracy)
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all generator metrics"""
        return {
            "Answer Correctness": "Evaluates if the generated answer is correct and complete based on the input and retrieved context",
            "Citation Accuracy": "Checks if citations in the answer are correct and relevant to the retrieved context"
        }


# Import required for GEval parameters
try:
    from deepeval.test_case import LLMTestCaseParams
except ImportError:
    # Fallback if the import structure is different
    logger.warning("Could not import LLMTestCaseParams, using string fallbacks")
    class LLMTestCaseParams:
        INPUT = "input"
        ACTUAL_OUTPUT = "actual_output"
        RETRIEVAL_CONTEXT = "retrieval_context"


class MetricsManager:
    """
    Centralized metrics management
    Orchestrates both retriever and generator metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics manager
        
        Args:
            config: Complete evaluation configuration
        """
        self.config = config
        self.model = config.get('evaluation', {}).get('model', 'gpt-4')
        
        # Initialize component metrics
        self.retriever_metrics = RetrieverMetrics(
            config.get('retriever_metrics', {}),
            self.model
        )
        self.generator_metrics = GeneratorMetrics(
            config.get('generator_metrics', {}),
            self.model
        )
        
        logger.info("Metrics manager initialized")
    
    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """Get all metrics organized by component"""
        return {
            'retriever': self.retriever_metrics.get_all_metrics(),
            'generator': self.generator_metrics.get_all_metrics()
        }
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """Get information about all available metrics"""
        retriever_metrics = self.retriever_metrics.get_all_metrics()
        generator_metrics = self.generator_metrics.get_all_metrics()
        total_metrics = len(retriever_metrics) + len(generator_metrics)
        
        return {
            'retriever_metrics': self.retriever_metrics.get_metric_descriptions(),
            'generator_metrics': self.generator_metrics.get_metric_descriptions(),
            'total_metrics': total_metrics,
            'metric_breakdown': {
                'retriever': len(retriever_metrics),
                'generator': len(generator_metrics),
                'chunking_evaluation': 8,  # 8 chunking evaluation metrics
                'total_with_chunking': total_metrics + 8
            }
        }
    
    def validate_metrics_config(self) -> Dict[str, Any]:
        """Validate metrics configuration"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if required models are available
        required_models = set()
        
        # Check retriever metrics config
        for metric_name, metric_config in self.config.get('retriever_metrics', {}).items():
            if isinstance(metric_config, dict) and 'model' in metric_config:
                required_models.add(metric_config['model'])
        
        # Check generator metrics config
        for metric_name, metric_config in self.config.get('generator_metrics', {}).items():
            if isinstance(metric_config, dict) and 'model' in metric_config:
                required_models.add(metric_config['model'])
        
        # Add warnings for model availability
        for model in required_models:
            if model.startswith('gpt-') and 'OPENAI_API_KEY' not in os.environ:
                validation_results['warnings'].append(
                    f"Model {model} requires OpenAI API key"
                )
        
        return validation_results 