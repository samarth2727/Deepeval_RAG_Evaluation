"""
Metrics Tracker Module
Tracks and manages RAG evaluation metrics over time
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks RAG evaluation metrics over time
    
    Features:
    - Store metrics history
    - Calculate trends and aggregations
    - Performance monitoring
    - Alert generation
    """
    
    def __init__(self, storage_path: str = "reports/metrics_history.json"):
        """
        Initialize metrics tracker
        
        Args:
            storage_path: Path to store metrics history
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[Dict[str, Any]] = []
        self._load_history()
        
        logger.info(f"Metrics tracker initialized with {len(self.metrics_history)} historical records")
    
    def record_evaluation(
        self,
        evaluation_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a new evaluation result
        
        Args:
            evaluation_results: Results from DeepEval evaluation
            metadata: Additional metadata about the evaluation
        """
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'results': evaluation_results,
            'metadata': metadata or {},
            'summary': self._calculate_summary(evaluation_results)
        }
        
        self.metrics_history.append(record)
        self._save_history()
        
        logger.info(f"Recorded evaluation at {timestamp}")
    
    def get_recent_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get metrics from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent evaluation records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = []
        for record in self.metrics_history:
            record_time = datetime.fromisoformat(record['timestamp'])
            if record_time >= cutoff_time:
                recent_metrics.append(record)
        
        return recent_metrics
    
    def get_metrics_trend(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Get trend for a specific metric
        
        Args:
            metric_name: Name of the metric to analyze
            days: Number of days to analyze
            
        Returns:
            Trend analysis results
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        values = []
        timestamps = []
        
        for record in self.metrics_history:
            record_time = datetime.fromisoformat(record['timestamp'])
            if record_time >= cutoff_time:
                if metric_name in record['summary']:
                    values.append(record['summary'][metric_name])
                    timestamps.append(record_time)
        
        if not values:
            return {'trend': 'no_data', 'values': [], 'timestamps': []}
        
        # Calculate trend
        if len(values) >= 2:
            recent_avg = sum(values[-3:]) / min(3, len(values))
            older_avg = sum(values[:3]) / min(3, len(values))
            
            if recent_avg > older_avg * 1.05:
                trend = 'improving'
            elif recent_avg < older_avg * 0.95:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'values': values,
            'timestamps': [t.isoformat() for t in timestamps],
            'current_value': values[-1] if values else None,
            'average': sum(values) / len(values) if values else None,
            'min': min(values) if values else None,
            'max': max(values) if values else None
        }
    
    def get_performance_alerts(self, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check for performance alerts based on thresholds
        
        Args:
            thresholds: Dictionary of metric_name -> threshold_value
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        latest_record = self.metrics_history[-1]
        latest_summary = latest_record['summary']
        
        for metric_name, threshold in thresholds.items():
            if metric_name in latest_summary:
                current_value = latest_summary[metric_name]
                
                if current_value < threshold:
                    alerts.append({
                        'type': 'performance_degradation',
                        'metric': metric_name,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'high' if current_value < threshold * 0.8 else 'medium',
                        'timestamp': latest_record['timestamp']
                    })
        
        return alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get formatted data for dashboard display
        
        Returns:
            Dashboard data dictionary
        """
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest_record = self.metrics_history[-1]
        recent_metrics = self.get_recent_metrics(24)
        
        # Calculate key metrics trends
        key_metrics = ['answer_correctness', 'contextual_relevancy', 'contextual_recall']
        trends = {}
        
        for metric in key_metrics:
            trends[metric] = self.get_metrics_trend(metric, days=7)
        
        return {
            'status': 'active',
            'latest_evaluation': latest_record,
            'recent_count': len(recent_metrics),
            'total_evaluations': len(self.metrics_history),
            'trends': trends,
            'alerts': self.get_performance_alerts({
                'answer_correctness': 0.7,
                'contextual_relevancy': 0.75,
                'contextual_recall': 0.8
            })
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export metrics history to pandas DataFrame
        
        Returns:
            DataFrame with metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        rows = []
        for record in self.metrics_history:
            row = {
                'timestamp': record['timestamp'],
                **record['summary'],
                **{f"meta_{k}": v for k, v in record['metadata'].items()}
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _calculate_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate summary statistics from evaluation results
        
        Args:
            evaluation_results: Raw evaluation results
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        # Extract key metrics
        if 'retriever_metrics' in evaluation_results:
            retriever = evaluation_results['retriever_metrics']
            if 'contextual_relevancy' in retriever:
                summary['contextual_relevancy'] = retriever['contextual_relevancy'].get('score', 0.0)
            if 'contextual_recall' in retriever:
                summary['contextual_recall'] = retriever['contextual_recall'].get('score', 0.0)
            if 'contextual_precision' in retriever:
                summary['contextual_precision'] = retriever['contextual_precision'].get('score', 0.0)
        
        if 'generator_metrics' in evaluation_results:
            generator = evaluation_results['generator_metrics']
            if 'answer_correctness' in generator:
                summary['answer_correctness'] = generator['answer_correctness'].get('score', 0.0)
            if 'citation_accuracy' in generator:
                summary['citation_accuracy'] = generator['citation_accuracy'].get('score', 0.0)
        
        # Calculate overall score
        if summary:
            summary['overall_score'] = sum(summary.values()) / len(summary)
        
        # Add evaluation metadata
        summary['num_test_cases'] = evaluation_results.get('num_test_cases', 0)
        summary['evaluation_time'] = evaluation_results.get('evaluation_time', 0.0)
        
        return summary
    
    def _load_history(self):
        """Load metrics history from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self.metrics_history = json.load(f)
                logger.info(f"Loaded {len(self.metrics_history)} historical records")
            else:
                self.metrics_history = []
                logger.info("No existing metrics history found")
        except Exception as e:
            logger.error(f"Error loading metrics history: {e}")
            self.metrics_history = []
    
    def _save_history(self):
        """Save metrics history to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics history: {e}")
    
    def clear_history(self):
        """Clear all metrics history"""
        self.metrics_history = []
        self._save_history()
        logger.info("Cleared all metrics history")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about the metrics
        
        Returns:
            Statistics dictionary
        """
        if not self.metrics_history:
            return {'total_evaluations': 0}
        
        df = self.export_to_dataframe()
        
        stats = {
            'total_evaluations': len(self.metrics_history),
            'date_range': {
                'start': self.metrics_history[0]['timestamp'],
                'end': self.metrics_history[-1]['timestamp']
            }
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            if col.startswith('meta_'):
                continue
            
            stats[f"{col}_stats"] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        return stats 