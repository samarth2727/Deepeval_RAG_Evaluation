"""
Synthetic Data Generation for RAG Evaluation
Uses DeepEval Synthesizer to generate test cases from documents
"""

import logging
from typing import List, Dict, Any, Optional
import random
import time

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

try:
    from deepeval.synthesizer import Synthesizer
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("DeepEval Synthesizer not available, using fallback implementation")
    Synthesizer = None

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Synthetic test data generation using DeepEval Synthesizer
    
    Features:
    - Document-based test case generation
    - Multi-domain support
    - Quality filtering
    - Batch generation
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize synthetic data generator
        
        Args:
            model: Model to use for synthetic generation
        """
        self.model = model
        self.synthesizer = self._initialize_synthesizer()
        logger.info("Synthetic data generator initialized")
    
    def _initialize_synthesizer(self):
        """Initialize DeepEval synthesizer"""
        if Synthesizer is None:
            logger.warning("DeepEval Synthesizer not available")
            return None
        
        try:
            synthesizer = Synthesizer(model=self.model)
            logger.info(f"DeepEval Synthesizer initialized with model {self.model}")
            return synthesizer
        except Exception as e:
            logger.error(f"Failed to initialize synthesizer: {e}")
            return None
    
    def generate_from_documents(
        self,
        documents: List[str],
        num_samples: int = 100,
        domains: List[str] = None
    ) -> List[LLMTestCase]:
        """
        Generate synthetic test cases from documents
        
        Args:
            documents: List of document contents
            num_samples: Number of test cases to generate
            domains: Target domains for generation
            
        Returns:
            List of synthetic test cases
        """
        logger.info(f"Generating {num_samples} synthetic test cases from {len(documents)} documents")
        
        if self.synthesizer is None:
            return self._fallback_generation(documents, num_samples, domains)
        
        try:
            # Prepare documents for synthesis
            document_paths = self._prepare_documents(documents)
            
            # Generate golden dataset
            goldens = self.synthesizer.generate_goldens_from_docs(
                document_paths=document_paths,
                max_goldens_per_document=num_samples // len(documents) + 1
            )
            
            # Convert to test cases
            test_cases = []
            for golden in goldens[:num_samples]:
                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output="",  # Will be filled by RAG system
                    expected_output=golden.expected_output,
                    retrieval_context=golden.context or []
                )
                test_cases.append(test_case)
            
            logger.info(f"Generated {len(test_cases)} synthetic test cases")
            return test_cases
            
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")
            return self._fallback_generation(documents, num_samples, domains)
    
    def _prepare_documents(self, documents: List[str]) -> List[str]:
        """Prepare documents for synthesis by saving to temporary files"""
        import tempfile
        import os
        
        document_paths = []
        
        for i, doc_content in enumerate(documents):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.txt',
                    delete=False,
                    encoding='utf-8'
                ) as f:
                    f.write(doc_content)
                    document_paths.append(f.name)
            
            except Exception as e:
                logger.error(f"Error preparing document {i}: {e}")
        
        return document_paths
    
    def _fallback_generation(
        self,
        documents: List[str],
        num_samples: int,
        domains: List[str] = None
    ) -> List[LLMTestCase]:
        """
        Fallback synthetic generation when DeepEval Synthesizer is not available
        
        Args:
            documents: List of document contents
            num_samples: Number of test cases to generate
            domains: Target domains for generation
            
        Returns:
            List of synthetic test cases
        """
        logger.info("Using fallback synthetic generation")
        
        test_cases = []
        domains = domains or ["general", "technical", "scientific"]
        
        # Template questions for different domains
        question_templates = {
            "general": [
                "What is {topic}?",
                "How does {topic} work?",
                "Why is {topic} important?",
                "What are the benefits of {topic}?",
                "Explain {topic} in simple terms."
            ],
            "technical": [
                "What are the technical specifications of {topic}?",
                "How do you implement {topic}?",
                "What are the best practices for {topic}?",
                "What are the common issues with {topic}?",
                "How do you troubleshoot {topic}?"
            ],
            "scientific": [
                "What is the scientific basis of {topic}?",
                "How was {topic} discovered?",
                "What research supports {topic}?",
                "What are the scientific applications of {topic}?",
                "How does {topic} relate to other scientific concepts?"
            ]
        }
        
        for i in range(num_samples):
            try:
                # Select random document
                doc = random.choice(documents)
                
                # Extract potential topics (simple word extraction)
                words = doc.split()
                potential_topics = [w for w in words if len(w) > 5 and w.isalpha()]
                
                if not potential_topics:
                    continue
                
                topic = random.choice(potential_topics[:10])  # Use first 10 words as potential topics
                domain = random.choice(domains)
                template = random.choice(question_templates.get(domain, question_templates["general"]))
                
                # Generate question
                question = template.format(topic=topic)
                
                # Generate simple answer (extract relevant sentences)
                sentences = doc.split('.')
                relevant_sentences = [s for s in sentences if topic.lower() in s.lower()]
                answer = '. '.join(relevant_sentences[:2]) if relevant_sentences else f"Information about {topic} can be found in the provided context."
                
                # Create test case
                test_case = LLMTestCase(
                    input=question,
                    actual_output="",
                    expected_output=answer,
                    retrieval_context=[doc[:1000]]  # Use first 1000 chars as context
                )
                
                test_cases.append(test_case)
                
            except Exception as e:
                logger.error(f"Error in fallback generation for sample {i}: {e}")
        
        logger.info(f"Generated {len(test_cases)} fallback synthetic test cases")
        return test_cases
    
    def generate_domain_specific_cases(
        self,
        domain: str,
        documents: List[str],
        num_samples: int = 50
    ) -> List[LLMTestCase]:
        """
        Generate domain-specific test cases
        
        Args:
            domain: Target domain (e.g., "medical", "legal", "technical")
            documents: Domain-specific documents
            num_samples: Number of test cases to generate
            
        Returns:
            List of domain-specific test cases
        """
        logger.info(f"Generating {num_samples} {domain}-specific test cases")
        
        # Domain-specific prompts and templates
        domain_configs = {
            "medical": {
                "question_types": ["symptoms", "treatment", "diagnosis", "causes", "prevention"],
                "complexity": "high"
            },
            "legal": {
                "question_types": ["rights", "procedures", "requirements", "definitions", "consequences"],
                "complexity": "high"
            },
            "technical": {
                "question_types": ["implementation", "configuration", "troubleshooting", "best_practices"],
                "complexity": "medium"
            },
            "general": {
                "question_types": ["definition", "explanation", "examples", "benefits", "process"],
                "complexity": "low"
            }
        }
        
        config = domain_configs.get(domain, domain_configs["general"])
        
        # Generate using domain-specific approach
        if self.synthesizer:
            try:
                # Use synthesizer with domain-specific prompts
                return self.generate_from_documents(documents, num_samples, [domain])
            except Exception as e:
                logger.error(f"Domain-specific generation failed: {e}")
        
        # Fallback to general generation
        return self._fallback_generation(documents, num_samples, [domain])
    
    def filter_quality_cases(
        self,
        test_cases: List[LLMTestCase],
        min_question_length: int = 10,
        min_answer_length: int = 20,
        min_context_length: int = 50
    ) -> List[LLMTestCase]:
        """
        Filter test cases based on quality criteria
        
        Args:
            test_cases: Input test cases
            min_question_length: Minimum question length
            min_answer_length: Minimum answer length
            min_context_length: Minimum context length
            
        Returns:
            Filtered high-quality test cases
        """
        logger.info(f"Filtering {len(test_cases)} test cases for quality")
        
        filtered_cases = []
        
        for test_case in test_cases:
            try:
                # Check quality criteria
                if (len(test_case.input) >= min_question_length and
                    len(test_case.expected_output or "") >= min_answer_length and
                    sum(len(ctx) for ctx in test_case.retrieval_context or []) >= min_context_length):
                    
                    filtered_cases.append(test_case)
            
            except Exception as e:
                logger.error(f"Error filtering test case: {e}")
        
        logger.info(f"Filtered to {len(filtered_cases)} high-quality test cases")
        return filtered_cases
    
    def augment_test_cases(
        self,
        test_cases: List[LLMTestCase],
        augmentation_factor: int = 2
    ) -> List[LLMTestCase]:
        """
        Augment test cases by creating variations
        
        Args:
            test_cases: Original test cases
            augmentation_factor: Number of variations per case
            
        Returns:
            Augmented test cases
        """
        logger.info(f"Augmenting {len(test_cases)} test cases with factor {augmentation_factor}")
        
        augmented_cases = list(test_cases)  # Keep originals
        
        # Simple augmentation strategies
        augmentation_strategies = [
            self._rephrase_question,
            self._add_context_noise,
            self._modify_expected_answer
        ]
        
        for test_case in test_cases:
            for i in range(augmentation_factor - 1):  # -1 because we keep original
                try:
                    strategy = random.choice(augmentation_strategies)
                    augmented_case = strategy(test_case)
                    if augmented_case:
                        augmented_cases.append(augmented_case)
                
                except Exception as e:
                    logger.error(f"Error augmenting test case: {e}")
        
        logger.info(f"Augmented to {len(augmented_cases)} test cases")
        return augmented_cases
    
    def _rephrase_question(self, test_case: LLMTestCase) -> Optional[LLMTestCase]:
        """Rephrase question with simple variations"""
        original_question = test_case.input
        
        # Simple rephrasing patterns
        rephrase_patterns = [
            ("What is", "Can you explain"),
            ("How does", "In what way does"),
            ("Why is", "What makes"),
            ("What are", "Which are")
        ]
        
        rephrased = original_question
        for old, new in rephrase_patterns:
            if old in rephrased:
                rephrased = rephrased.replace(old, new, 1)
                break
        
        if rephrased != original_question:
            return LLMTestCase(
                input=rephrased,
                actual_output=test_case.actual_output,
                expected_output=test_case.expected_output,
                retrieval_context=test_case.retrieval_context
            )
        
        return None
    
    def _add_context_noise(self, test_case: LLMTestCase) -> LLMTestCase:
        """Add some noise to context (e.g., shuffle order)"""
        if test_case.retrieval_context:
            shuffled_context = test_case.retrieval_context.copy()
            random.shuffle(shuffled_context)
            
            return LLMTestCase(
                input=test_case.input,
                actual_output=test_case.actual_output,
                expected_output=test_case.expected_output,
                retrieval_context=shuffled_context
            )
        
        return test_case
    
    def _modify_expected_answer(self, test_case: LLMTestCase) -> LLMTestCase:
        """Slightly modify expected answer while keeping meaning"""
        if test_case.expected_output:
            # Simple modifications: add explanatory phrases
            modified_answer = test_case.expected_output
            
            if not modified_answer.endswith('.'):
                modified_answer += '.'
            
            return LLMTestCase(
                input=test_case.input,
                actual_output=test_case.actual_output,
                expected_output=modified_answer,
                retrieval_context=test_case.retrieval_context
            )
        
        return test_case
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files created during generation"""
        import os
        
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}") 