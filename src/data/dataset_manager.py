"""
Dataset Management System
Centralized management for evaluation datasets
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Centralized dataset management for RAG evaluation
    
    Features:
    - MS MARCO dataset handling
    - Custom dataset management
    - Dataset preprocessing
    - Dataset statistics and validation
    """
    
    def __init__(self, datasets_dir: str = "datasets"):
        """
        Initialize dataset manager
        
        Args:
            datasets_dir: Directory to store datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Available datasets
        self.available_datasets = {
            'ms_marco': {
                'name': 'MS MARCO QA Dataset',
                'description': 'Microsoft Machine Reading Comprehension Dataset',
                'url': 'https://msmarco.blob.core.windows.net/msmarcoranking/train_v2.1.json.gz',
                'file_name': 'ms_marco_train.json.gz',
                'processed_name': 'ms_marco_processed.json'
            }
        }
        
        logger.info(f"Dataset manager initialized with directory: {datasets_dir}")
    
    def download_ms_marco(self, subset_size: int = 1000) -> str:
        """
        Download and process MS MARCO dataset
        
        Args:
            subset_size: Number of samples to extract
            
        Returns:
            Path to processed dataset file
        """
        dataset_info = self.available_datasets['ms_marco']
        file_path = self.datasets_dir / dataset_info['file_name']
        processed_path = self.datasets_dir / dataset_info['processed_name']
        
        # Check if already processed
        if processed_path.exists():
            logger.info(f"MS MARCO dataset already exists at {processed_path}")
            return str(processed_path)
        
        # Download if not exists
        if not file_path.exists():
            logger.info("Downloading MS MARCO dataset...")
            self._download_file(dataset_info['url'], file_path)
        
        # Process dataset
        logger.info("Processing MS MARCO dataset...")
        self._process_ms_marco(file_path, processed_path, subset_size)
        
        return str(processed_path)
    
    def _download_file(self, url: str, file_path: Path):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=file_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            logger.info(f"Downloaded {file_path}")
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            raise
    
    def _process_ms_marco(self, source_path: Path, output_path: Path, subset_size: int):
        """Process raw MS MARCO data into evaluation format"""
        import gzip
        
        try:
            processed_data = []
            
            # Open and process file
            open_func = gzip.open if source_path.suffix == '.gz' else open
            
            with open_func(source_path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= subset_size:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract relevant fields
                        query = data.get('query', '')
                        passages = data.get('passages', [])
                        answers = data.get('answers', [])
                        
                        if query and passages and answers:
                            # Process passages
                            contexts = []
                            for passage in passages:
                                if passage.get('is_selected', 0) == 1:  # Selected passages
                                    contexts.append(passage.get('passage_text', ''))
                            
                            # Get answer
                            answer = answers[0] if answers else ''
                            
                            if contexts and answer:
                                processed_item = {
                                    'query': query,
                                    'expected_answer': answer,
                                    'contexts': contexts,
                                    'metadata': {
                                        'source': 'ms_marco',
                                        'index': i,
                                        'num_passages': len(passages)
                                    }
                                }
                                processed_data.append(processed_item)
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {i}: {e}")
            
            # Save processed data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {len(processed_data)} samples to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing MS MARCO data: {e}")
            raise
    
    def load_dataset(self, dataset_name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load dataset by name
        
        Args:
            dataset_name: Name of dataset to load
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            List of dataset samples
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == 'ms_marco':
            return self.load_ms_marco_dataset(**kwargs)
        elif dataset_name == 'custom':
            return self.load_custom_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def load_ms_marco_dataset(self, subset_size: int = 1000) -> List[Dict[str, Any]]:
        """Load MS MARCO dataset"""
        # Download/process if needed
        dataset_path = self.download_ms_marco(subset_size)
        
        # Load processed data
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} MS MARCO samples")
        return data
    
    def load_custom_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load custom dataset from file
        
        Args:
            file_path: Path to custom dataset file
            
        Returns:
            List of dataset samples
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(data)} samples from custom dataset")
            return data
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            raise
    
    def create_sample_dataset(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Create a small sample dataset for testing
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            List of sample dataset items
        """
        sample_data = []
        
        # Sample questions and contexts about AI/ML
        samples = [
            {
                'query': 'What is machine learning?',
                'expected_answer': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.',
                'contexts': [
                    'Machine learning is a method of data analysis that automates analytical model building.',
                    'It is a branch of artificial intelligence based on the idea that systems can learn from data.'
                ]
            },
            {
                'query': 'How does deep learning work?',
                'expected_answer': 'Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input.',
                'contexts': [
                    'Deep learning is part of a broader family of machine learning methods based on artificial neural networks.',
                    'The adjective "deep" refers to the use of multiple layers in the network.'
                ]
            },
            {
                'query': 'What is natural language processing?',
                'expected_answer': 'Natural language processing (NLP) is a subfield of AI that focuses on the interaction between computers and human language.',
                'contexts': [
                    'NLP combines computational linguistics with statistical, machine learning, and deep learning models.',
                    'These technologies enable computers to process and analyze large amounts of natural language data.'
                ]
            }
        ]
        
        # Repeat and modify samples to reach desired count
        for i in range(num_samples):
            base_sample = samples[i % len(samples)]
            sample = {
                **base_sample,
                'metadata': {
                    'source': 'sample',
                    'index': i,
                    'base_index': i % len(samples)
                }
            }
            sample_data.append(sample)
        
        logger.info(f"Created {len(sample_data)} sample dataset items")
        return sample_data
    
    def validate_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate dataset format and quality
        
        Args:
            data: Dataset to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_samples': len(data),
            'valid_samples': 0,
            'invalid_samples': 0,
            'errors': [],
            'statistics': {}
        }
        
        required_fields = ['query', 'expected_answer', 'contexts']
        
        query_lengths = []
        answer_lengths = []
        context_counts = []
        
        for i, sample in enumerate(data):
            sample_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in sample:
                    sample_errors.append(f"Sample {i}: Missing field '{field}'")
                elif not sample[field]:
                    sample_errors.append(f"Sample {i}: Empty field '{field}'")
            
            # Check data types
            if 'contexts' in sample and not isinstance(sample['contexts'], list):
                sample_errors.append(f"Sample {i}: 'contexts' should be a list")
            
            # Quality checks
            if 'query' in sample and len(sample['query']) < 5:
                sample_errors.append(f"Sample {i}: Query too short")
            
            if 'expected_answer' in sample and len(sample['expected_answer']) < 10:
                sample_errors.append(f"Sample {i}: Answer too short")
            
            # Collect statistics
            if 'query' in sample:
                query_lengths.append(len(sample['query']))
            if 'expected_answer' in sample:
                answer_lengths.append(len(sample['expected_answer']))
            if 'contexts' in sample and isinstance(sample['contexts'], list):
                context_counts.append(len(sample['contexts']))
            
            if sample_errors:
                validation_results['invalid_samples'] += 1
                validation_results['errors'].extend(sample_errors)
            else:
                validation_results['valid_samples'] += 1
        
        # Calculate statistics
        if query_lengths:
            validation_results['statistics'] = {
                'avg_query_length': sum(query_lengths) / len(query_lengths),
                'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
                'avg_context_count': sum(context_counts) / len(context_counts) if context_counts else 0,
                'min_query_length': min(query_lengths),
                'max_query_length': max(query_lengths),
                'total_contexts': sum(context_counts) if context_counts else 0
            }
        
        logger.info(
            f"Dataset validation: {validation_results['valid_samples']} valid, "
            f"{validation_results['invalid_samples']} invalid"
        )
        
        return validation_results
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available datasets"""
        info = {
            'available_datasets': self.available_datasets,
            'datasets_directory': str(self.datasets_dir),
            'downloaded_datasets': []
        }
        
        # Check which datasets are downloaded
        for dataset_name, dataset_info in self.available_datasets.items():
            processed_path = self.datasets_dir / dataset_info['processed_name']
            if processed_path.exists():
                info['downloaded_datasets'].append({
                    'name': dataset_name,
                    'path': str(processed_path),
                    'size': processed_path.stat().st_size
                })
        
        return info 