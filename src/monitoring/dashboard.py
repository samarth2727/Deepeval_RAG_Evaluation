"""
Streamlit Dashboard for RAG Evaluation Monitoring
Real-time visualization of evaluation results and system performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import time
from datetime import datetime, timedelta

# Setup page config
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class RAGDashboard:
    """
    Comprehensive dashboard for RAG evaluation monitoring
    
    Features:
    - Real-time metrics visualization
    - Component-level performance analysis
    - Historical trends
    - System health monitoring
    - Interactive evaluation reports
    """
    
    def __init__(self):
        """Initialize dashboard"""
        self.reports_dir = Path("reports")
        self.logs_dir = Path("logs")
        
        # Create directories if they don't exist
        self.reports_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def run(self):
        """Run the Streamlit dashboard"""
        # Title and header
        st.title("RAG Evaluation Dashboard")
        st.markdown("### Production-Ready RAG Evaluation with DeepEval")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_sidebar(self):
        """Render sidebar with navigation and controls"""
        st.sidebar.title("Navigation")
        
        # Page selection
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Detailed Metrics", "Component Analysis", "Historical Trends", "System Health", "Run Evaluation"]
        )
        
        # Refresh controls
        st.sidebar.markdown("### Controls")
        if st.sidebar.button("Refresh Data"):
            st.rerun()
        
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Filter controls
        st.sidebar.markdown("### Filters")
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now()
        )
        
        # Component filter
        components = st.sidebar.multiselect(
            "Components",
            ["retriever", "generator"],
            default=["retriever", "generator"]
        )
        
        # Store selections in session state
        st.session_state.current_page = page
        st.session_state.date_range = date_range
        st.session_state.selected_components = components
    
    def _render_main_content(self):
        """Render main dashboard content based on selected page"""
        page = st.session_state.get('current_page', 'Overview')
        
        if page == "Overview":
            self._render_overview()
        elif page == "Detailed Metrics":
            self._render_detailed_metrics()
        elif page == "Component Analysis":
            self._render_component_analysis()
        elif page == "Historical Trends":
            self._render_historical_trends()
        elif page == "System Health":
            self._render_system_health()
        elif page == "Run Evaluation":
            self._render_run_evaluation()
    
    def _render_overview(self):
        """Render overview page with key metrics"""
        st.header("Evaluation Overview")
        
        # Load latest evaluation results
        latest_results = self._load_latest_results()
        
        if not latest_results:
            st.warning("No evaluation results found. Run an evaluation to see metrics.")
            return
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric_card(
                "Overall Score",
                latest_results.get("summary", {}).get("overall_performance", {}).get("average_score", 0),
                ""
            )
        
        with col2:
            self._render_metric_card(
                "Test Cases",
                latest_results.get("summary", {}).get("total_test_cases", 0),
                ""
            )
        
        with col3:
            self._render_metric_card(
                "Execution Time",
                f"{latest_results.get('execution_time', 0):.1f}s",
                ""
            )
        
        with col4:
            performance_grade = latest_results.get("summary", {}).get("overall_performance", {}).get("performance_grade", "Unknown")
            self._render_metric_card(
                "Performance",
                performance_grade,
                ""
            )
        
        # Chunking evaluation metrics if available
        if "steps" in latest_results and "indexing" in latest_results["steps"]:
            indexing_results = latest_results["steps"]["indexing"]
            if indexing_results.get("chunking_quality") is not None:
                st.subheader("Chunking Evaluation Metrics")
                
                chunk_col1, chunk_col2, chunk_col3, chunk_col4 = st.columns(4)
                
                with chunk_col1:
                    self._render_metric_card(
                        "Chunking Quality",
                        f"{indexing_results.get('chunking_quality', 0):.3f}",
                        ""
                    )
                
                with chunk_col2:
                    self._render_metric_card(
                        "Processing Quality",
                        f"{indexing_results.get('processing_quality', 0):.3f}",
                        ""
                    )
                
                with chunk_col3:
                    self._render_metric_card(
                        "Total Chunks",
                        indexing_results.get('total_chunks', 0),
                        ""
                    )
                
                with chunk_col4:
                    self._render_metric_card(
                        "Evaluation Time",
                        f"{indexing_results.get('evaluation_time', 0):.2f}s",
                        ""
                    )
                
                # Detailed chunking metrics - check multiple possible locations
                quality_metrics = None
                if "quality_metrics" in indexing_results and indexing_results["quality_metrics"]:
                    quality_metrics = indexing_results["quality_metrics"]
                elif "results" in indexing_results and indexing_results["results"]:
                    # Check if quality metrics are in the results
                    for result in indexing_results["results"]:
                        if isinstance(result, dict) and "quality_metrics" in result:
                            quality_metrics = result["quality_metrics"]
                            break
                
                if quality_metrics and "chunking" in quality_metrics:
                    st.subheader("Detailed Chunking Metrics")
                    chunk_details = quality_metrics["chunking"]
                    
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    with detail_col1:
                        st.metric("Coherence", f"{chunk_details.get('coherence', 0):.3f}")
                        st.metric("Completeness", f"{chunk_details.get('completeness', 0):.3f}")
                    
                    with detail_col2:
                        st.metric("Overlap Quality", f"{chunk_details.get('overlap_quality', 0):.3f}")
                        st.metric("Size Consistency", f"{chunk_details.get('size_consistency', 0):.3f}")
                    
                    with detail_col3:
                        st.metric("Semantic Boundaries", f"{chunk_details.get('semantic_boundaries', 0):.3f}")
                        st.metric("Overall Quality", f"{chunk_details.get('overall', 0):.3f}")
        
        # Component scores chart
        st.subheader("Component Performance")
        self._render_component_scores_chart(latest_results)
        
        # Recent evaluations table
        st.subheader("Recent Evaluations")
        self._render_recent_evaluations_table()
    
    def _render_detailed_metrics(self):
        """Render detailed metrics analysis"""
        st.header("Detailed Metrics Analysis")
        
        latest_results = self._load_latest_results()
        
        if not latest_results:
            st.warning("No evaluation results found.")
            return
        
        # Metrics breakdown by component
        final_results = latest_results.get("final_results", {})
        
        for component in st.session_state.get('selected_components', []):
            if component in final_results:
                self._render_component_detailed_metrics(component, final_results[component])
    
    def _render_component_analysis(self):
        """Render component-specific analysis"""
        st.header("Component Analysis")
        
        # Component selection
        component = st.selectbox("Select Component", ["retriever", "generator"])
        
        # Load and analyze component data
        component_data = self._load_component_historical_data(component)
        
        if component_data:
            # Performance over time
            self._render_component_performance_over_time(component, component_data)
            
            # Metric distribution
            self._render_metric_distribution(component, component_data)
            
            # Top performing test cases
            self._render_top_performing_cases(component, component_data)
        else:
            st.info(f"No historical data found for {component} component.")
    
    def _render_historical_trends(self):
        """Render historical trends analysis"""
        st.header("Historical Trends")
        
        # Load historical data
        historical_data = self._load_historical_evaluation_data()
        
        if not historical_data:
            st.info("No historical data available.")
            return
        
        # Overall performance trend
        st.subheader("Overall Performance Trend")
        self._render_performance_trend_chart(historical_data)
        
        # Component comparison over time
        st.subheader("Component Performance Comparison")
        self._render_component_comparison_chart(historical_data)
        
        # Statistics table
        st.subheader("Evaluation Statistics")
        self._render_evaluation_statistics_table(historical_data)
    
    def _render_system_health(self):
        """Render system health monitoring"""
        st.header("System Health")
        
        # System status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Component Status")
            self._render_component_status()
        
        with col2:
            st.subheader("Recent Errors")
            self._render_recent_errors()
        
        # Performance metrics
        st.subheader("Performance Metrics")
        self._render_system_performance_metrics()
    
    def _render_run_evaluation(self):
        """Render evaluation execution interface"""
        st.header("Run New Evaluation")
        
        with st.form("evaluation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                dataset = st.selectbox("Dataset", ["sample", "ms_marco", "custom"])
                dataset_size = st.number_input("Number of Test Cases", min_value=1, max_value=1000, value=20)
            
            with col2:
                document_paths = st.text_area("Document Paths (one per line)", help="Leave empty to use sample documents")
                save_results = st.checkbox("Save Results", value=True)
            
            submitted = st.form_submit_button("Run Evaluation")
            
            if submitted:
                self._run_evaluation(dataset, dataset_size, document_paths, save_results)
    
    def _render_metric_card(self, title: str, value: Any, icon: str):
        """Render a metric card"""
        st.metric(
            label=f"{icon} {title}",
            value=value
        )
    
    def _render_component_scores_chart(self, results: Dict[str, Any]):
        """Render component scores chart"""
        component_scores = results.get("summary", {}).get("component_scores", {})
        
        if not component_scores:
            st.info("No component scores available.")
            return
        
        # Prepare data for plotting
        components = []
        metrics = []
        scores = []
        
        for component, metric_scores in component_scores.items():
            for metric, score in metric_scores.items():
                components.append(component.title())
                metrics.append(metric)
                scores.append(score)
        
        if components:
            fig = px.bar(
                x=components,
                y=scores,
                color=metrics,
                title="Component Scores by Metric",
                labels={"x": "Component", "y": "Score"}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_recent_evaluations_table(self):
        """Render table of recent evaluations"""
        evaluations = self._load_recent_evaluations()
        
        if evaluations:
            df = pd.DataFrame(evaluations)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent evaluations found.")
    
    def _render_component_detailed_metrics(self, component: str, component_results):
        """Render detailed metrics for a component"""
        st.subheader(f"{component.title()} Component Metrics")
        
        if hasattr(component_results, 'aggregate_scores'):
            metrics_data = []
            
            for metric, scores in component_results.aggregate_scores.items():
                if isinstance(scores, dict):
                    metrics_data.append({
                        "Metric": metric,
                        "Average": f"{scores.get('average', 0):.3f}",
                        "Min": f"{scores.get('min', 0):.3f}",
                        "Max": f"{scores.get('max', 0):.3f}",
                        "Count": scores.get('count', 0)
                    })
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                st.dataframe(df, use_container_width=True)
            
            # Create radar chart for metrics
            self._render_metrics_radar_chart(component, component_results.aggregate_scores)
    
    def _render_metrics_radar_chart(self, component: str, aggregate_scores: Dict[str, Any]):
        """Render radar chart for component metrics"""
        metrics = []
        values = []
        
        for metric, scores in aggregate_scores.items():
            if isinstance(scores, dict) and 'average' in scores:
                metrics.append(metric)
                values.append(scores['average'])
        
        if metrics and values:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=component.title()
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"{component.title()} Metrics Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _load_latest_results(self) -> Dict[str, Any]:
        """Load latest evaluation results"""
        try:
            result_files = list(self.reports_dir.glob("complete_evaluation_*.json"))
            
            if not result_files:
                return {}
            
            # Get the most recent file
            latest_file = max(result_files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return {}
    
    def _load_recent_evaluations(self) -> List[Dict[str, Any]]:
        """Load recent evaluation summaries"""
        try:
            result_files = list(self.reports_dir.glob("complete_evaluation_*.json"))
            evaluations = []
            
            for file_path in sorted(result_files, key=os.path.getctime, reverse=True)[:10]:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                evaluation = {
                    "Timestamp": datetime.fromtimestamp(data.get("pipeline_config", {}).get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                    "Dataset": data.get("pipeline_config", {}).get("dataset_name", "Unknown"),
                    "Test Cases": data.get("summary", {}).get("total_test_cases", 0),
                    "Overall Score": f"{data.get('summary', {}).get('overall_performance', {}).get('average_score', 0):.3f}",
                    "Grade": data.get("summary", {}).get("overall_performance", {}).get("performance_grade", "Unknown"),
                    "Execution Time": f"{data.get('execution_time', 0):.1f}s"
                }
                evaluations.append(evaluation)
            
            return evaluations
        
        except Exception as e:
            st.error(f"Error loading evaluations: {e}")
            return []
    
    def _load_component_historical_data(self, component: str) -> List[Dict[str, Any]]:
        """Load historical data for a specific component"""
        # This would typically come from a database
        # For now, return mock data for demonstration
        return []
    
    def _load_historical_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load historical evaluation data"""
        try:
            result_files = list(self.reports_dir.glob("complete_evaluation_*.json"))
            historical_data = []
            
            for file_path in result_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                historical_data.append(data)
            
            return sorted(historical_data, key=lambda x: x.get("pipeline_config", {}).get("timestamp", 0))
        
        except Exception as e:
            st.error(f"Error loading historical data: {e}")
            return []
    
    def _render_component_status(self):
        """Render component status indicators"""
        # Mock status for demonstration
        components = [
            {"name": "RAG System", "status": "✅ Healthy", "last_check": "2 min ago"},
            {"name": "DeepEval Framework", "status": "✅ Healthy", "last_check": "1 min ago"},
            {"name": "Vector Database", "status": "⚠️ Warning", "last_check": "5 min ago"},
            {"name": "LLM Service", "status": "✅ Healthy", "last_check": "30 sec ago"}
        ]
        
        for comp in components:
            st.text(f"{comp['status']} {comp['name']} ({comp['last_check']})")
    
    def _render_recent_errors(self):
        """Render recent error messages"""
        # This would typically come from log files
        st.text("No recent errors")
    
    def _render_system_performance_metrics(self):
        """Render system performance metrics"""
        # Mock performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Response Time", "1.2s", delta="-0.3s")
        
        with col2:
            st.metric("Success Rate", "98.5%", delta="2.1%")
        
        with col3:
            st.metric("Throughput", "45 req/min", delta="5 req/min")
    
    def _run_evaluation(self, dataset: str, dataset_size: int, document_paths: str, save_results: bool):
        """Run evaluation with specified parameters"""
        with st.spinner("Running evaluation..."):
            try:
                # Import and run the main pipeline
                import sys
                from pathlib import Path
                
                # Add src to path
                src_path = Path(__file__).parent.parent
                if str(src_path) not in sys.path:
                    sys.path.append(str(src_path))
                
                from main import RAGEvaluationPipeline
                
                # Parse document paths
                doc_paths = None
                if document_paths.strip():
                    doc_paths = [path.strip() for path in document_paths.split('\n') if path.strip()]
                
                # Create and run pipeline
                pipeline = RAGEvaluationPipeline()
                results = pipeline.run_complete_evaluation(
                    dataset_name=dataset,
                    dataset_size=dataset_size,
                    document_paths=doc_paths,
                    save_results=save_results
                )
                
                if results.get("success"):
                    st.success("✅ Evaluation completed successfully!")
                    
                    # Display results summary
                    summary = results.get("summary", {})
                    if summary:
                        st.json(summary)
                    
                    # Refresh the page to show new results
                    time.sleep(2)
                    st.experimental_rerun()
                else:
                    st.error(f"❌ Evaluation failed: {results.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Error running evaluation: {str(e)}")
    
    def _render_performance_trend_chart(self, historical_data: List[Dict[str, Any]]):
        """Render performance trend chart"""
        if not historical_data:
            return
        
        timestamps = []
        scores = []
        
        for data in historical_data:
            timestamp = data.get("pipeline_config", {}).get("timestamp", 0)
            score = data.get("summary", {}).get("overall_performance", {}).get("average_score", 0)
            
            if timestamp and score:
                timestamps.append(datetime.fromtimestamp(timestamp))
                scores.append(score)
        
        if timestamps and scores:
            fig = px.line(
                x=timestamps,
                y=scores,
                title="Overall Performance Trend",
                labels={"x": "Time", "y": "Score"}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_component_comparison_chart(self, historical_data: List[Dict[str, Any]]):
        """Render component comparison chart"""
        # Implementation would analyze component performance over time
        st.info("Component comparison chart - Coming soon!")
    
    def _render_evaluation_statistics_table(self, historical_data: List[Dict[str, Any]]):
        """Render evaluation statistics table"""
        if not historical_data:
            return
        
        stats = []
        for data in historical_data:
            timestamp = data.get("pipeline_config", {}).get("timestamp", 0)
            summary = data.get("summary", {})
            
            stat = {
                "Date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M"),
                "Dataset": data.get("pipeline_config", {}).get("dataset_name", "Unknown"),
                "Test Cases": summary.get("total_test_cases", 0),
                "Overall Score": f"{summary.get('overall_performance', {}).get('average_score', 0):.3f}",
                "Execution Time": f"{data.get('execution_time', 0):.1f}s"
            }
            stats.append(stat)
        
        if stats:
            df = pd.DataFrame(stats)
            st.dataframe(df, use_container_width=True)


def main():
    """Main function to run the dashboard"""
    dashboard = RAGDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 