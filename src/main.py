"""
Main RAG Evaluation Application
Orchestrates the complete evaluation pipeline from RAG system to DeepEval metrics
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from rag.rag_system import RAGSystem
from evaluation.deepeval_framework import DeepEvalFramework
from evaluation.test_cases import TestCaseGenerator
from data.dataset_manager import DatasetManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class RAGEvaluationPipeline:
    """
    Complete RAG evaluation pipeline
    
    Orchestrates:
    1. RAG system setup and document indexing
    2. Test dataset preparation
    3. RAG inference on test cases
    4. DeepEval component-level evaluation
    5. Results generation and reporting
    """
    
    def __init__(
        self,
        rag_config_path: str = "config/rag_config.yaml",
        eval_config_path: str = "config/eval_config.yaml"
    ):
        """
        Initialize evaluation pipeline
        
        Args:
            rag_config_path: Path to RAG system configuration
            eval_config_path: Path to evaluation configuration
        """
        logger.info("Initializing RAG evaluation pipeline...")
        
        # Initialize components
        self.rag_system = RAGSystem(rag_config_path)
        self.eval_framework = DeepEvalFramework(eval_config_path)
        self.test_generator = TestCaseGenerator()
        self.dataset_manager = DatasetManager()
        
        logger.info("RAG evaluation pipeline initialized successfully")
    
    def run_complete_evaluation(
        self,
        dataset_name: str = "sample",
        dataset_size: int = 50,
        document_paths: List[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            dataset_name: Name of dataset to use ("ms_marco", "sample", or "custom")
            dataset_size: Number of test cases to evaluate
            document_paths: Paths to documents for RAG indexing
            save_results: Whether to save evaluation results
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting complete RAG evaluation pipeline")
        start_time = time.time()
        
        results = {
            "pipeline_config": {
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "document_paths": document_paths,
                "timestamp": start_time
            },
            "steps": {},
            "final_results": {},
            "execution_time": 0
        }
        
        try:
            # Step 1: Index documents into RAG system with evaluations
            if document_paths:
                logger.info("Step 1: Indexing documents with chunking evaluations...")
                indexing_results = self.rag_system.index_documents(document_paths, enable_evaluations=True)
                results["steps"]["indexing"] = indexing_results
                
                if not indexing_results.get("success", False):
                    raise Exception("Document indexing failed")
                
                # Log chunking evaluation results
                if indexing_results.get("chunking_quality") is not None:
                    logger.info(f"Chunking quality: {indexing_results['chunking_quality']:.3f}")
                    logger.info(f"Processing quality: {indexing_results['processing_quality']:.3f}")
                    logger.info(f"Evaluation time: {indexing_results['evaluation_time']:.2f}s")
            else:
                logger.info("Step 1: Skipping document indexing (no documents provided)")
                # Create sample documents for demonstration
                sample_docs = self._create_sample_documents()
                self._save_sample_documents(sample_docs)
                indexing_results = self.rag_system.index_documents([
                    "data/sample_doc1.txt", "data/sample_doc2.txt", "data/sample_doc3.txt"
                ], enable_evaluations=True)
                results["steps"]["indexing"] = indexing_results
            
            # Step 2: Prepare test dataset
            logger.info("Step 2: Preparing test dataset...")
            test_data = self._prepare_test_dataset(dataset_name, dataset_size)
            results["steps"]["dataset_preparation"] = {
                "dataset_name": dataset_name,
                "test_cases_loaded": len(test_data),
                "dataset_validation": self.dataset_manager.validate_dataset(test_data)
            }
            
            # Step 3: Generate RAG responses
            logger.info("Step 3: Generating RAG responses...")
            rag_responses = self._generate_rag_responses(test_data)
            results["steps"]["rag_inference"] = {
                "total_queries": len(test_data),
                "successful_responses": len([r for r in rag_responses if r.get("success", False)]),
                "avg_response_time": sum(r.get("metadata", {}).get("execution_time", 0) for r in rag_responses) / len(rag_responses) if rag_responses else 0
            }
            
            # Step 4: Create DeepEval test cases
            logger.info("Step 4: Creating DeepEval test cases...")
            test_cases = self._create_deepeval_test_cases(test_data, rag_responses)
            results["steps"]["test_case_creation"] = {
                "test_cases_created": len(test_cases)
            }
            
            # Step 5: Run component-level evaluation
            logger.info("Step 5: Running DeepEval component evaluation...")
            evaluation_results = self.eval_framework.evaluate_complete_rag_system(test_cases)
            results["final_results"] = evaluation_results
            
            # Step 6: Generate summary
            logger.info("Step 6: Generating evaluation summary...")
            evaluation_summary = self._generate_evaluation_summary(evaluation_results)
            results["summary"] = evaluation_summary
            
            results["execution_time"] = time.time() - start_time
            results["success"] = True
            
            # Save results if requested
            if save_results:
                self._save_pipeline_results(results)
            
            logger.info(f"Complete evaluation pipeline finished in {results['execution_time']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            results["error"] = str(e)
            results["success"] = False
            results["execution_time"] = time.time() - start_time
            return results
    
    def _prepare_test_dataset(self, dataset_name: str, dataset_size: int) -> List[Dict[str, Any]]:
        """Prepare test dataset based on specified type"""
        if dataset_name == "ms_marco":
            return self.dataset_manager.load_ms_marco_dataset(subset_size=dataset_size)
        elif dataset_name == "sample":
            return self.dataset_manager.create_sample_dataset(num_samples=dataset_size)
        elif dataset_name == "custom":
            # For demo, create sample data
            return self.dataset_manager.create_sample_dataset(num_samples=dataset_size)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    def _generate_rag_responses(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate RAG responses for test queries"""
        responses = []
        
        for test_item in test_data:
            try:
                query = test_item["query"]
                response = self.rag_system.query(query)
                responses.append(response.__dict__ if hasattr(response, '__dict__') else response)
            except Exception as e:
                logger.error(f"Error generating response for query: {e}")
                responses.append({
                    "query": test_item["query"],
                    "answer": f"Error: {str(e)}",
                    "retrieved_contexts": [],
                    "success": False
                })
        
        return responses
    
    def _create_deepeval_test_cases(
        self,
        test_data: List[Dict[str, Any]],
        rag_responses: List[Dict[str, Any]]
    ):
        """Create DeepEval test cases from test data and RAG responses"""
        # Convert to RAG test cases
        rag_test_cases = self.test_generator.create_custom_test_cases(test_data)
        
        # Add RAG responses
        updated_cases = self.test_generator.add_rag_responses_to_test_cases(
            rag_test_cases, rag_responses
        )
        
        # Convert to DeepEval format
        llm_test_cases = self.test_generator.convert_to_deepeval_format(updated_cases)
        
        return llm_test_cases
    
    def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results including chunking metrics"""
        summary = {
            "components_evaluated": list(evaluation_results.keys()),
            "total_test_cases": 0,
            "total_execution_time": 0,
            "component_scores": {},
            "overall_performance": {},
            "chunking_evaluation": {},
            "processing_evaluation": {}
        }
        
        all_scores = []
        
        for component, results in evaluation_results.items():
            if hasattr(results, 'test_cases_count'):
                summary["total_test_cases"] = max(summary["total_test_cases"], results.test_cases_count)
                summary["total_execution_time"] += results.execution_time
                
                # Extract scores
                component_scores = {}
                for metric, scores in results.aggregate_scores.items():
                    if isinstance(scores, dict) and 'average' in scores:
                        component_scores[metric] = scores['average']
                        all_scores.append(scores['average'])
                
                summary["component_scores"][component] = component_scores
        
        # Calculate overall performance
        if all_scores:
            summary["overall_performance"] = {
                "average_score": sum(all_scores) / len(all_scores),
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "performance_grade": "Excellent" if sum(all_scores) / len(all_scores) >= 0.8 
                                   else "Good" if sum(all_scores) / len(all_scores) >= 0.6
                                   else "Needs Improvement"
            }
        
        return summary
    
    def _create_sample_documents(self) -> List[str]:
        """Create sample documents for demonstration"""
        return [
            """
            Machine Learning Fundamentals
            
            Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves the development of algorithms and statistical models that can identify patterns in data and make predictions or decisions based on that data.
            
            There are three main types of machine learning:
            1. Supervised Learning: Uses labeled training data to learn a function that maps inputs to outputs
            2. Unsupervised Learning: Finds hidden patterns in data without labeled examples
            3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties
            
            Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.
            """,
            """
            Deep Learning and Neural Networks
            
            Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input. These networks are inspired by the structure and function of the human brain.
            
            Key components of neural networks:
            - Neurons (nodes): Basic processing units
            - Layers: Input, hidden, and output layers
            - Weights and Biases: Parameters that determine the network's behavior
            - Activation Functions: Determine the output of neurons
            
            Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition. Popular frameworks include TensorFlow, PyTorch, and Keras.
            """,
            """
            Natural Language Processing (NLP)
            
            Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between computers and human language. It combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to process and analyze large amounts of natural language data.
            
            Key NLP tasks include:
            - Text Classification: Categorizing text into predefined classes
            - Named Entity Recognition: Identifying entities like names, locations, organizations
            - Sentiment Analysis: Determining emotional tone of text
            - Machine Translation: Translating text from one language to another
            - Question Answering: Automatically answering questions posed in natural language
            
            Modern NLP leverages transformer models like BERT, GPT, and T5 to achieve state-of-the-art performance across various tasks.
            """
        ]
    
    def _save_sample_documents(self, documents: List[str]):
        """Save sample documents to files"""
        os.makedirs("data", exist_ok=True)
        
        for i, doc in enumerate(documents, 1):
            file_path = f"data/sample_doc{i}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc)
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results"""
        import json
        
        timestamp = int(time.time())
        results_path = f"reports/complete_evaluation_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Pipeline results saved to {results_path}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--dataset", default="sample", help="Dataset to use (sample, ms_marco)")
    parser.add_argument("--size", type=int, default=20, help="Number of test cases")
    parser.add_argument("--documents", nargs="+", help="Document paths to index")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = RAGEvaluationPipeline()
    
    # Run evaluation
    results = pipeline.run_complete_evaluation(
        dataset_name=args.dataset,
        dataset_size=args.size,
        document_paths=args.documents,
        save_results=not args.no_save
    )
    
    # Print summary
    if results.get("success"):
        print("\n🎉 Evaluation completed successfully!")
        
        # Print chunking evaluation results if available
        if "steps" in results and "indexing" in results["steps"]:
            indexing_results = results["steps"]["indexing"]
            if indexing_results.get("chunking_quality") is not None:
                print(f"\n🔍 Chunking Evaluation Results:")
                print(f"  Chunking Quality: {indexing_results['chunking_quality']:.3f}")
                print(f"  Processing Quality: {indexing_results['processing_quality']:.3f}")
                print(f"  Evaluation Time: {indexing_results['evaluation_time']:.2f}s")
                print(f"  Processing Time: {indexing_results['processing_time']:.2f}s")
                print(f"  Total Chunks: {indexing_results['total_chunks']}")
                print(f"  Avg Chunks per Doc: {indexing_results['average_chunks_per_doc']:.1f}")
        
        if "summary" in results:
            summary = results["summary"]
            print(f"\n📊 Evaluation Summary:")
            print(f"Components Evaluated: {', '.join(summary['components_evaluated'])}")
            print(f"Total Test Cases: {summary['total_test_cases']}")
            print(f"Execution Time: {summary['total_execution_time']:.2f}s")
            
            if "overall_performance" in summary:
                perf = summary["overall_performance"]
                print(f"Overall Score: {perf['average_score']:.3f}")
                print(f"Performance Grade: {perf['performance_grade']}")
            
            print(f"\n📈 Component Scores:")
            for component, scores in summary.get("component_scores", {}).items():
                print(f"  {component.title()}:")
                for metric, score in scores.items():
                    print(f"    {metric}: {score:.3f}")
    else:
        print(f"\n❌ Evaluation failed: {results.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    main() 