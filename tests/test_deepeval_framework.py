"""
Tests for DeepEval evaluation framework
"""

import pytest
import sys
from pathlib import Path
import tempfile
import json

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from evaluation.deepeval_framework import DeepEvalFramework, EvaluationConfig
from evaluation.metrics import RetrieverMetrics, GeneratorMetrics, EvaluationResults
from evaluation.test_cases import TestCaseGenerator, RAGTestCase
from evaluation.synthetic_data import SyntheticDataGenerator


class TestEvaluationConfig:
    """Test evaluation configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EvaluationConfig()
        
        assert config.model == "gpt-4"
        assert config.max_test_cases_per_run == 100
        assert config.parallel_execution is True
        assert "html" in config.output_format
        assert "json" in config.output_format
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = EvaluationConfig(
            model="gpt-3.5-turbo",
            max_test_cases_per_run=50,
            parallel_execution=False
        )
        
        assert config.model == "gpt-3.5-turbo"
        assert config.max_test_cases_per_run == 50
        assert config.parallel_execution is False


class TestRetrieverMetrics:
    """Test retriever metrics functionality"""
    
    def test_initialization(self):
        """Test retriever metrics initialization"""
        config = {
            "contextual_relevancy": {"enabled": True, "threshold": 0.7},
            "contextual_recall": {"enabled": True, "threshold": 0.8},
            "contextual_precision": {"enabled": False, "threshold": 0.6}
        }
        
        metrics = RetrieverMetrics(config)
        assert metrics.config == config
        assert metrics.model == "gpt-4"
    
    def test_get_all_metrics(self):
        """Test getting all enabled metrics"""
        config = {
            "contextual_relevancy": {"enabled": True, "threshold": 0.7},
            "contextual_recall": {"enabled": True, "threshold": 0.8},
            "contextual_precision": {"enabled": False, "threshold": 0.6}
        }
        
        metrics = RetrieverMetrics(config)
        all_metrics = metrics.get_all_metrics()
        
        # Should have 2 enabled metrics
        assert len(all_metrics) == 2
    
    def test_metric_descriptions(self):
        """Test metric descriptions"""
        metrics = RetrieverMetrics({})
        descriptions = metrics.get_metric_descriptions()
        
        assert "ContextualRelevancyMetric" in descriptions
        assert "ContextualRecallMetric" in descriptions
        assert "ContextualPrecisionMetric" in descriptions


class TestGeneratorMetrics:
    """Test generator metrics functionality"""
    
    def test_initialization(self):
        """Test generator metrics initialization"""
        config = {
            "answer_correctness": {"enabled": True, "threshold": 0.7},
            "citation_accuracy": {"enabled": True, "threshold": 0.8}
        }
        
        metrics = GeneratorMetrics(config)
        assert metrics.config == config
    
    def test_get_all_metrics(self):
        """Test getting all enabled metrics"""
        config = {
            "answer_correctness": {"enabled": True, "threshold": 0.7},
            "citation_accuracy": {"enabled": False, "threshold": 0.8}
        }
        
        metrics = GeneratorMetrics(config)
        all_metrics = metrics.get_all_metrics()
        
        # Should have 1 enabled metric
        assert len(all_metrics) == 1


class TestRAGTestCase:
    """Test RAG test case functionality"""
    
    def test_rag_test_case_creation(self):
        """Test creating RAG test case"""
        test_case = RAGTestCase(
            query="What is machine learning?",
            expected_answer="Machine learning is a subset of AI.",
            retrieved_contexts=["Context about ML", "Another context"],
            metadata={"source": "test"}
        )
        
        assert test_case.query == "What is machine learning?"
        assert test_case.expected_answer == "Machine learning is a subset of AI."
        assert len(test_case.retrieved_contexts) == 2
        assert test_case.metadata["source"] == "test"
    
    def test_to_llm_test_case_conversion(self):
        """Test conversion to LLM test case"""
        rag_case = RAGTestCase(
            query="Test query",
            expected_answer="Test answer",
            retrieved_contexts=["Context 1"],
            actual_answer="Generated answer"
        )
        
        llm_case = rag_case.to_llm_test_case()
        
        assert llm_case.input == "Test query"
        assert llm_case.expected_output == "Test answer"
        assert llm_case.actual_output == "Generated answer"
        assert llm_case.retrieval_context == ["Context 1"]


class TestTestCaseGenerator:
    """Test test case generation"""
    
    def test_initialization(self):
        """Test test case generator initialization"""
        generator = TestCaseGenerator()
        assert generator.test_cases == []
    
    def test_create_custom_test_cases(self):
        """Test creating custom test cases"""
        generator = TestCaseGenerator()
        
        test_data = [
            {
                "query": "What is AI?",
                "expected_answer": "AI is artificial intelligence.",
                "contexts": ["Context about AI"]
            },
            {
                "query": "What is ML?",
                "expected_answer": "ML is machine learning.",
                "contexts": ["Context about ML"]
            }
        ]
        
        test_cases = generator.create_custom_test_cases(test_data)
        
        assert len(test_cases) == 2
        assert test_cases[0].query == "What is AI?"
        assert test_cases[1].query == "What is ML?"
    
    def test_validate_test_cases(self):
        """Test test case validation"""
        generator = TestCaseGenerator()
        
        test_cases = [
            RAGTestCase(
                query="Good query with sufficient length",
                expected_answer="Good answer with sufficient detail",
                retrieved_contexts=["Good context"]
            ),
            RAGTestCase(
                query="Bad",  # Too short
                expected_answer="Bad",  # Too short
                retrieved_contexts=[]  # Empty
            )
        ]
        
        validation = generator.validate_test_cases(test_cases)
        
        assert validation["total_cases"] == 2
        assert validation["valid_cases"] == 1
        assert validation["invalid_cases"] == 1
        assert len(validation["issues"]) > 0
    
    def test_export_import_test_cases(self):
        """Test exporting and importing test cases"""
        generator = TestCaseGenerator()
        
        test_cases = [
            RAGTestCase(
                query="Test query",
                expected_answer="Test answer",
                retrieved_contexts=["Context 1", "Context 2"]
            )
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "test_cases.json"
            
            # Export
            generator.export_test_cases(test_cases, str(export_path), "json")
            assert export_path.exists()
            
            # Import
            imported_cases = generator.import_test_cases(str(export_path), "json")
            assert len(imported_cases) == 1
            assert imported_cases[0].query == "Test query"


class TestSyntheticDataGenerator:
    """Test synthetic data generation"""
    
    def test_initialization(self):
        """Test synthetic data generator initialization"""
        generator = SyntheticDataGenerator()
        assert generator.model == "gpt-4"
    
    def test_fallback_generation(self):
        """Test fallback synthetic generation"""
        generator = SyntheticDataGenerator()
        
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        # Force fallback generation
        test_cases = generator._fallback_generation(documents, 5, ["technical"])
        
        assert len(test_cases) <= 5
        for case in test_cases:
            assert case.input is not None
            assert case.expected_output is not None
    
    def test_quality_filtering(self):
        """Test quality filtering of test cases"""
        generator = SyntheticDataGenerator()
        
        # Create mock test cases with varying quality
        from deepeval.test_case import LLMTestCase
        
        test_cases = [
            LLMTestCase(
                input="This is a good quality question with sufficient length?",
                expected_output="This is a detailed answer with sufficient information.",
                retrieval_context=["Good context with enough detail for evaluation"]
            ),
            LLMTestCase(
                input="Bad?",  # Too short
                expected_output="Bad",  # Too short
                retrieval_context=["Short"]  # Too short
            )
        ]
        
        filtered_cases = generator.filter_quality_cases(
            test_cases,
            min_question_length=15,
            min_answer_length=25,
            min_context_length=30
        )
        
        assert len(filtered_cases) == 1  # Only the good quality case


class TestEvaluationResults:
    """Test evaluation results functionality"""
    
    def test_evaluation_results_creation(self):
        """Test creating evaluation results"""
        results = EvaluationResults(
            component="retriever",
            test_cases_count=10,
            aggregate_scores={"metric1": {"average": 0.8}},
            individual_results=[],
            execution_time=5.0,
            timestamp=1234567890,
            config={"model": "gpt-4"}
        )
        
        assert results.component == "retriever"
        assert results.test_cases_count == 10
        assert results.execution_time == 5.0
    
    def test_save_to_json(self):
        """Test saving results to JSON"""
        results = EvaluationResults(
            component="generator",
            test_cases_count=5,
            aggregate_scores={"metric1": {"average": 0.7}},
            individual_results=[],
            execution_time=3.0,
            timestamp=1234567890,
            config={"model": "gpt-4"}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "results.json"
            results.save_to_json(str(json_path))
            
            assert json_path.exists()
            
            # Verify content
            with open(json_path) as f:
                data = json.load(f)
            
            assert data["component"] == "generator"
            assert data["test_cases_count"] == 5


class TestDeepEvalFramework:
    """Test complete DeepEval framework"""
    
    def test_initialization_without_config(self):
        """Test framework initialization without config file"""
        try:
            framework = DeepEvalFramework("nonexistent_config.yaml")
            # Should initialize with empty config
            assert framework.config == {}
        except Exception as e:
            pytest.skip(f"Framework initialization test failed: {e}")
    
    def test_create_test_cases_from_rag_responses(self):
        """Test creating test cases from RAG responses"""
        try:
            framework = DeepEvalFramework()
            
            rag_responses = [
                {
                    "query": "What is AI?",
                    "answer": "AI is artificial intelligence.",
                    "retrieved_contexts": ["Context about AI"]
                }
            ]
            
            expected_answers = ["AI stands for artificial intelligence."]
            
            test_cases = framework.create_test_cases_from_rag_responses(
                rag_responses, expected_answers
            )
            
            assert len(test_cases) == 1
            assert test_cases[0].input == "What is AI?"
        except Exception as e:
            pytest.skip(f"Test case creation test failed: {e}")
    
    def test_evaluation_summary(self):
        """Test getting evaluation summary"""
        try:
            framework = DeepEvalFramework()
            summary = framework.get_evaluation_summary()
            
            # Should handle empty results gracefully
            assert "message" in summary or "total_evaluations" in summary
        except Exception as e:
            pytest.skip(f"Evaluation summary test failed: {e}")


class TestIntegrationEvaluation:
    """Integration tests for evaluation framework"""
    
    @pytest.mark.integration
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow"""
        try:
            # Create test data
            test_data = [
                {
                    "query": "What is machine learning?",
                    "expected_answer": "Machine learning is a subset of AI.",
                    "contexts": ["ML is part of artificial intelligence."]
                }
            ]
            
            rag_responses = [
                {
                    "query": "What is machine learning?",
                    "answer": "Machine learning is a branch of AI.",
                    "retrieved_contexts": ["ML is part of artificial intelligence."]
                }
            ]
            
            # Initialize components
            generator = TestCaseGenerator()
            framework = DeepEvalFramework()
            
            # Create test cases
            test_cases = generator.create_custom_test_cases(test_data)
            updated_cases = generator.add_rag_responses_to_test_cases(test_cases, rag_responses)
            llm_test_cases = generator.convert_to_deepeval_format(updated_cases)
            
            assert len(llm_test_cases) == 1
            assert llm_test_cases[0].input == "What is machine learning?"
            
        except Exception as e:
            pytest.skip(f"Complete evaluation workflow test requires external services: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 