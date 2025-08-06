"""
Test Case Generation and Management for RAG Evaluation
Implements test case creation patterns following DeepEval methodology
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

logger = logging.getLogger(__name__)


@dataclass
class RAGTestCase:
    """Enhanced test case for RAG evaluation"""
    query: str
    expected_answer: str
    retrieved_contexts: List[str]
    actual_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_llm_test_case(self) -> LLMTestCase:
        """Convert to DeepEval LLMTestCase"""
        return LLMTestCase(
            input=self.query,
            actual_output=self.actual_answer or "",
            expected_output=self.expected_answer,
            retrieval_context=self.retrieved_contexts
        )


class TestCaseGenerator:
    """
    Test case generation and management
    
    Features:
    - MS MARCO dataset integration
    - Custom test case creation
    - Test case validation
    - Dataset export/import
    """
    
    def __init__(self):
        """Initialize test case generator"""
        self.test_cases: List[RAGTestCase] = []
        logger.info("Test case generator initialized")
    
    def load_ms_marco_dataset(
        self,
        dataset_path: str,
        subset_size: int = 1000
    ) -> List[RAGTestCase]:
        """
        Load MS MARCO dataset for evaluation
        
        Args:
            dataset_path: Path to MS MARCO dataset file
            subset_size: Number of samples to load
            
        Returns:
            List of RAG test cases
        """
        logger.info(f"Loading MS MARCO dataset from {dataset_path}")
        
        try:
            test_cases = []
            
            # Load MS MARCO data (expecting JSON format)
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process samples
            samples = data.get('data', data) if isinstance(data, dict) else data
            
            for i, sample in enumerate(samples[:subset_size]):
                try:
                    # Extract required fields
                    query = sample.get('query', sample.get('question', ''))
                    passages = sample.get('passages', sample.get('contexts', []))
                    answers = sample.get('answers', sample.get('wellFormedAnswers', []))
                    
                    # Get the first answer as expected
                    expected_answer = answers[0] if answers else ""
                    
                    # Convert passages to context strings
                    contexts = []
                    if isinstance(passages, list):
                        for passage in passages:
                            if isinstance(passage, dict):
                                text = passage.get('passage_text', passage.get('text', ''))
                            else:
                                text = str(passage)
                            if text:
                                contexts.append(text)
                    
                    if query and expected_answer and contexts:
                        test_case = RAGTestCase(
                            query=query,
                            expected_answer=expected_answer,
                            retrieved_contexts=contexts,
                            metadata={
                                'source': 'ms_marco',
                                'index': i,
                                'num_contexts': len(contexts)
                            }
                        )
                        test_cases.append(test_case)
                    
                except Exception as e:
                    logger.warning(f"Error processing MS MARCO sample {i}: {e}")
            
            logger.info(f"Loaded {len(test_cases)} test cases from MS MARCO")
            self.test_cases.extend(test_cases)
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to load MS MARCO dataset: {e}")
            return []
    
    def create_custom_test_cases(
        self,
        test_data: List[Dict[str, Any]]
    ) -> List[RAGTestCase]:
        """
        Create custom test cases from provided data
        
        Args:
            test_data: List of dictionaries with test case data
            
        Returns:
            List of RAG test cases
        """
        logger.info(f"Creating {len(test_data)} custom test cases")
        
        test_cases = []
        
        for i, data in enumerate(test_data):
            try:
                test_case = RAGTestCase(
                    query=data['query'],
                    expected_answer=data['expected_answer'],
                    retrieved_contexts=data.get('contexts', data.get('retrieved_contexts', [])),
                    metadata=data.get('metadata', {'source': 'custom', 'index': i})
                )
                test_cases.append(test_case)
                
            except KeyError as e:
                logger.error(f"Missing required field in test case {i}: {e}")
            except Exception as e:
                logger.error(f"Error creating test case {i}: {e}")
        
        logger.info(f"Created {len(test_cases)} custom test cases")
        self.test_cases.extend(test_cases)
        return test_cases
    
    def add_rag_responses_to_test_cases(
        self,
        test_cases: List[RAGTestCase],
        rag_responses: List[Dict[str, Any]]
    ) -> List[RAGTestCase]:
        """
        Add RAG system responses to existing test cases
        
        Args:
            test_cases: List of test cases
            rag_responses: List of RAG system responses
            
        Returns:
            Updated test cases with actual answers
        """
        logger.info(f"Adding RAG responses to {len(test_cases)} test cases")
        
        updated_cases = []
        
        for i, (test_case, response) in enumerate(zip(test_cases, rag_responses)):
            try:
                # Update test case with actual answer
                updated_case = RAGTestCase(
                    query=test_case.query,
                    expected_answer=test_case.expected_answer,
                    retrieved_contexts=response.get('retrieved_contexts', test_case.retrieved_contexts),
                    actual_answer=response.get('answer', ''),
                    metadata={
                        **(test_case.metadata or {}),
                        'rag_metadata': response.get('metadata', {}),
                        'updated_timestamp': time.time()
                    }
                )
                updated_cases.append(updated_case)
                
            except Exception as e:
                logger.error(f"Error updating test case {i}: {e}")
                updated_cases.append(test_case)  # Keep original if update fails
        
        logger.info(f"Updated {len(updated_cases)} test cases with RAG responses")
        return updated_cases
    
    def convert_to_deepeval_format(
        self,
        test_cases: List[RAGTestCase]
    ) -> List[LLMTestCase]:
        """
        Convert RAG test cases to DeepEval LLMTestCase format
        
        Args:
            test_cases: List of RAG test cases
            
        Returns:
            List of LLMTestCase objects
        """
        logger.info(f"Converting {len(test_cases)} test cases to DeepEval format")
        
        llm_test_cases = []
        
        for test_case in test_cases:
            try:
                llm_case = test_case.to_llm_test_case()
                llm_test_cases.append(llm_case)
            except Exception as e:
                logger.error(f"Error converting test case: {e}")
        
        logger.info(f"Converted {len(llm_test_cases)} test cases")
        return llm_test_cases
    
    def validate_test_cases(
        self,
        test_cases: List[RAGTestCase]
    ) -> Dict[str, Any]:
        """
        Validate test cases for completeness and quality
        
        Args:
            test_cases: List of test cases to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Validating {len(test_cases)} test cases")
        
        validation_results = {
            'total_cases': len(test_cases),
            'valid_cases': 0,
            'invalid_cases': 0,
            'issues': [],
            'statistics': {}
        }
        
        query_lengths = []
        answer_lengths = []
        context_counts = []
        
        for i, test_case in enumerate(test_cases):
            issues = []
            
            # Check required fields
            if not test_case.query.strip():
                issues.append(f"Case {i}: Empty query")
            
            if not test_case.expected_answer.strip():
                issues.append(f"Case {i}: Empty expected answer")
            
            if not test_case.retrieved_contexts:
                issues.append(f"Case {i}: No retrieved contexts")
            
            # Check quality metrics
            if len(test_case.query) < 10:
                issues.append(f"Case {i}: Query too short (<10 chars)")
            
            if len(test_case.expected_answer) < 20:
                issues.append(f"Case {i}: Expected answer too short (<20 chars)")
            
            if len(test_case.retrieved_contexts) < 1:
                issues.append(f"Case {i}: Insufficient contexts (<1)")
            
            # Collect statistics
            query_lengths.append(len(test_case.query))
            answer_lengths.append(len(test_case.expected_answer))
            context_counts.append(len(test_case.retrieved_contexts))
            
            if issues:
                validation_results['invalid_cases'] += 1
                validation_results['issues'].extend(issues)
            else:
                validation_results['valid_cases'] += 1
        
        # Calculate statistics
        if query_lengths:
            validation_results['statistics'] = {
                'avg_query_length': sum(query_lengths) / len(query_lengths),
                'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
                'avg_context_count': sum(context_counts) / len(context_counts),
                'min_query_length': min(query_lengths),
                'max_query_length': max(query_lengths),
                'min_context_count': min(context_counts),
                'max_context_count': max(context_counts)
            }
        
        logger.info(
            f"Validation complete: {validation_results['valid_cases']} valid, "
            f"{validation_results['invalid_cases']} invalid"
        )
        
        return validation_results
    
    def export_test_cases(
        self,
        test_cases: List[RAGTestCase],
        file_path: str,
        format: str = 'json'
    ):
        """
        Export test cases to file
        
        Args:
            test_cases: Test cases to export
            file_path: Output file path
            format: Export format ('json' or 'csv')
        """
        logger.info(f"Exporting {len(test_cases)} test cases to {file_path}")
        
        try:
            if format.lower() == 'json':
                # Export as JSON
                data = [asdict(test_case) for test_case in test_cases]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                # Export as CSV
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'query', 'expected_answer', 'actual_answer',
                        'num_contexts', 'metadata'
                    ])
                    
                    for test_case in test_cases:
                        writer.writerow([
                            test_case.query,
                            test_case.expected_answer,
                            test_case.actual_answer or '',
                            len(test_case.retrieved_contexts),
                            json.dumps(test_case.metadata or {})
                        ])
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Test cases exported successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export test cases: {e}")
    
    def import_test_cases(
        self,
        file_path: str,
        format: str = 'json'
    ) -> List[RAGTestCase]:
        """
        Import test cases from file
        
        Args:
            file_path: Input file path
            format: Import format ('json' or 'csv')
            
        Returns:
            List of imported test cases
        """
        logger.info(f"Importing test cases from {file_path}")
        
        try:
            test_cases = []
            
            if format.lower() == 'json':
                # Import from JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    test_case = RAGTestCase(**item)
                    test_cases.append(test_case)
            
            elif format.lower() == 'csv':
                # Import from CSV
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        metadata = json.loads(row.get('metadata', '{}'))
                        test_case = RAGTestCase(
                            query=row['query'],
                            expected_answer=row['expected_answer'],
                            actual_answer=row.get('actual_answer') or None,
                            retrieved_contexts=[],  # Not stored in CSV
                            metadata=metadata
                        )
                        test_cases.append(test_case)
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Imported {len(test_cases)} test cases")
            self.test_cases.extend(test_cases)
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to import test cases: {e}")
            return []
    
    def get_test_case_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded test cases"""
        if not self.test_cases:
            return {"message": "No test cases loaded"}
        
        sources = {}
        query_lengths = []
        answer_lengths = []
        context_counts = []
        
        for test_case in self.test_cases:
            # Source statistics
            source = test_case.metadata.get('source', 'unknown') if test_case.metadata else 'unknown'
            sources[source] = sources.get(source, 0) + 1
            
            # Length statistics
            query_lengths.append(len(test_case.query))
            answer_lengths.append(len(test_case.expected_answer))
            context_counts.append(len(test_case.retrieved_contexts))
        
        return {
            'total_test_cases': len(self.test_cases),
            'sources': sources,
            'statistics': {
                'avg_query_length': sum(query_lengths) / len(query_lengths),
                'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
                'avg_context_count': sum(context_counts) / len(context_counts)
            }
        } 