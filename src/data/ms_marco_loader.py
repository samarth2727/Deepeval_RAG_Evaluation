"""
MS MARCO Dataset Loader
Specialized loader for Microsoft Machine Reading Comprehension dataset
"""

import json
import gzip
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MSMarcoLoader:
    """
    MS MARCO dataset loader and processor
    
    Features:
    - Download MS MARCO dataset
    - Process and filter samples
    - Convert to evaluation format
    """
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize MS MARCO loader
        
        Args:
            data_dir: Directory to store dataset files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # MS MARCO dataset URLs
        self.dataset_urls = {
            'train': 'https://msmarco.blob.core.windows.net/msmarcoranking/train_v2.1.json.gz',
            'dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/dev_v2.1.json.gz'
        }
        
        logger.info(f"MS MARCO loader initialized with data directory: {data_dir}")
    
    def download_dataset(self, split: str = 'train', force_download: bool = False) -> Path:
        """
        Download MS MARCO dataset
        
        Args:
            split: Dataset split ('train' or 'dev')
            force_download: Force download even if file exists
            
        Returns:
            Path to downloaded file
        """
        if split not in self.dataset_urls:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.dataset_urls.keys())}")
        
        url = self.dataset_urls[split]
        filename = f"ms_marco_{split}.json.gz"
        file_path = self.data_dir / filename
        
        if file_path.exists() and not force_download:
            logger.info(f"Dataset file already exists: {file_path}")
            return file_path
        
        logger.info(f"Downloading MS MARCO {split} dataset...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            logger.info(f"Downloaded MS MARCO dataset to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def load_samples(
        self,
        split: str = 'train',
        max_samples: int = 1000,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and process MS MARCO samples
        
        Args:
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of processed samples
        """
        # Download dataset if needed
        file_path = self.download_dataset(split)
        
        logger.info(f"Loading {max_samples} samples from {file_path}")
        
        samples = []
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if len(samples) >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Apply filtering if specified
                        if filter_criteria and not self._matches_criteria(data, filter_criteria):
                            continue
                        
                        # Process sample
                        processed_sample = self._process_sample(data, i)
                        
                        if processed_sample:
                            samples.append(processed_sample)
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {i}: {e}")
            
            logger.info(f"Loaded {len(samples)} valid samples")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading samples: {e}")
            return []
    
    def _process_sample(self, data: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single MS MARCO sample
        
        Args:
            data: Raw sample data
            index: Sample index
            
        Returns:
            Processed sample or None if invalid
        """
        try:
            query = data.get('query', '').strip()
            passages = data.get('passages', [])
            answers = data.get('answers', [])
            
            if not query or not passages or not answers:
                return None
            
            # Extract relevant passages (those marked as selected)
            relevant_passages = []
            for passage in passages:
                if passage.get('is_selected', 0) == 1:
                    text = passage.get('passage_text', '').strip()
                    if text:
                        relevant_passages.append(text)
            
            # If no selected passages, use top-ranked ones
            if not relevant_passages:
                for passage in passages[:3]:  # Take top 3
                    text = passage.get('passage_text', '').strip()
                    if text:
                        relevant_passages.append(text)
            
            # Get the best answer
            best_answer = answers[0] if answers else ""
            
            if not relevant_passages or not best_answer:
                return None
            
            return {
                'query': query,
                'expected_answer': best_answer,
                'contexts': relevant_passages,
                'metadata': {
                    'source': 'ms_marco',
                    'index': index,
                    'num_passages': len(passages),
                    'num_answers': len(answers)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {index}: {e}")
            return None
    
    def _matches_criteria(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """
        Check if sample matches filtering criteria
        
        Args:
            data: Sample data
            criteria: Filtering criteria
            
        Returns:
            True if sample matches criteria
        """
        try:
            # Min query length
            if 'min_query_length' in criteria:
                query = data.get('query', '')
                if len(query) < criteria['min_query_length']:
                    return False
            
            # Min number of passages
            if 'min_passages' in criteria:
                passages = data.get('passages', [])
                if len(passages) < criteria['min_passages']:
                    return False
            
            # Require answers
            if criteria.get('require_answers', True):
                answers = data.get('answers', [])
                if not answers:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_sample_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about loaded samples
        
        Args:
            samples: List of samples
            
        Returns:
            Statistics dictionary
        """
        if not samples:
            return {'total_samples': 0}
        
        query_lengths = [len(s['query']) for s in samples]
        answer_lengths = [len(s['expected_answer']) for s in samples]
        context_counts = [len(s['contexts']) for s in samples]
        
        return {
            'total_samples': len(samples),
            'avg_query_length': sum(query_lengths) / len(query_lengths),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'avg_context_count': sum(context_counts) / len(context_counts),
            'min_query_length': min(query_lengths),
            'max_query_length': max(query_lengths),
            'total_contexts': sum(context_counts)
        } 