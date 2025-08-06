# RAG Evaluation POC - Usage Guide

## 🎯 Quick Start

This guide will help you set up and run the complete RAG evaluation system with DeepEval.

### Prerequisites

1. **Python 3.9+** installed
2. **OpenAI API key** (for GPT-4o-mini and evaluation metrics)
3. **Git** for version control

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd RAG-Deepeval
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

5. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

## 🚀 Running Evaluations

### Basic Usage

Run a complete RAG evaluation with default settings:

```bash
python src/main.py
```

### Custom Evaluation

```bash
python src/main.py \
  --dataset ms_marco \
  --size 50 \
  --documents data/your_docs.txt
```

### Parameters

- `--dataset`: Dataset type (`sample`, `ms_marco`, `custom`)
- `--size`: Number of test cases to evaluate
- `--documents`: Paths to documents for RAG indexing
- `--no-save`: Don't save evaluation results

## 📊 Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run src/dashboard.py
```

Features:
- Real-time evaluation monitoring
- Component performance analysis
- Historical trends
- Run new evaluations
- System health monitoring

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# Unit tests only
pytest -m "not integration"

# Integration tests (requires OpenAI API key)
pytest -m integration

# Specific test file
pytest tests/test_rag_system.py -v
```

## 📁 Project Structure

```
RAG-Deepeval/
├── src/                     # Source code
│   ├── rag/                # RAG system implementation
│   │   ├── rag_system.py   # Main RAG system
│   │   ├── components.py   # Custom components
│   │   └── pipeline.py     # Pipeline orchestration
│   ├── evaluation/         # DeepEval framework
│   │   ├── deepeval_framework.py  # Main evaluation framework
│   │   ├── metrics.py      # Metrics management
│   │   ├── test_cases.py   # Test case generation
│   │   └── synthetic_data.py  # Synthetic data generation
│   ├── data/               # Dataset management
│   ├── monitoring/         # Dashboard and monitoring
│   ├── main.py            # Main application
│   └── dashboard.py       # Dashboard entry point
├── config/                 # Configuration files
├── tests/                  # Test suites
├── datasets/              # Evaluation datasets
├── reports/               # Evaluation reports
├── logs/                  # Application logs
└── docs/                  # Documentation
```

## ⚙️ Configuration

### RAG System (`config/rag_config.yaml`)

```yaml
llm:
  provider: "openai"
  model_name: "gpt-4o-mini"
  api_key: "your-api-key"
  temperature: 0.2
  max_tokens: 512

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  
retrieval:
  top_k: 5
  score_threshold: 0.7
```

### Evaluation (`config/eval_config.yaml`)

```yaml
evaluation:
  model: "gpt-4"
  max_test_cases_per_run: 100

retriever_metrics:
  contextual_relevancy:
    enabled: true
    threshold: 0.7
    
generator_metrics:
  answer_correctness:
    enabled: true
    threshold: 0.7
```

## 📈 Evaluation Metrics

### Retriever Metrics (Official DeepEval)

1. **Contextual Relevancy**: Measures relevance of retrieved context to query
2. **Contextual Recall**: Evaluates if context contains enough information
3. **Contextual Precision**: Assesses precision without unnecessary information

### Generator Metrics (Custom GEval)

1. **Answer Correctness**: Evaluates accuracy and completeness of answers
2. **Citation Accuracy**: Checks correctness of citations and references

## 🔄 CI/CD Integration

The project includes GitHub Actions workflows for:

- **Automated evaluation** on every PR
- **Performance regression detection**
- **Deployment readiness checks**
- **Nightly evaluation runs**

### Setting up CI/CD

1. **Add repository secrets:**
   - `OPENAI_API_KEY`
   - `DEEPEVAL_API_KEY`

2. **Push to repository** - workflows will run automatically

3. **View results** in GitHub Actions tab

## 📊 Understanding Results

### Evaluation Output

```json
{
  "summary": {
    "overall_performance": {
      "average_score": 0.785,
      "performance_grade": "Good"
    },
    "component_scores": {
      "retriever": {
        "ContextualRelevancyMetric": 0.82,
        "ContextualRecallMetric": 0.75
      },
      "generator": {
        "Answer Correctness": 0.78,
        "Citation Accuracy": 0.80
      }
    }
  }
}
```

### Performance Grades

- **Excellent**: ≥ 0.8 (✅)
- **Good**: 0.6 - 0.8 (⚠️)
- **Needs Improvement**: < 0.6 (❌)

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API errors:**
   ```bash
   # Check API key
   echo $OPENAI_API_KEY
   
   # Test API access
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

2. **Import errors:**
   ```bash
   # Ensure correct Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. **Memory issues:**
   ```bash
   # Reduce batch size in config
   # evaluation.max_test_cases_per_run: 10
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python src/main.py
```

## 📚 Advanced Usage

### Custom Datasets

Create custom evaluation datasets:

```python
from src.data.dataset_manager import DatasetManager

manager = DatasetManager()
custom_data = [
    {
        "query": "Your question?",
        "expected_answer": "Expected answer",
        "contexts": ["Relevant context"]
    }
]

# Validate dataset
validation = manager.validate_dataset(custom_data)
print(validation)
```

### Custom Metrics

Add custom evaluation metrics:

```python
from deepeval.metrics import GEval

custom_metric = GEval(
    name="Custom Metric",
    criteria="Your evaluation criteria",
    evaluation_steps=[
        "Step 1: Check something",
        "Step 2: Validate something else"
    ]
)
```

### Synthetic Data Generation

Generate synthetic test cases:

```python
from src.evaluation.synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator()
test_cases = generator.generate_from_documents(
    documents=["Your document content"],
    num_samples=100,
    domains=["technical"]
)
```

## 🚀 Production Deployment

### Docker Deployment

Create a Dockerfile:

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "src/main.py"]
```

### Monitoring Setup

1. **Set up log aggregation** (ELK Stack, Splunk)
2. **Configure alerts** for evaluation failures
3. **Monitor performance metrics** over time
4. **Set up automated reporting**

### Scaling Considerations

- Use **distributed computing** for large datasets
- Implement **result caching** for repeated evaluations
- Consider **GPU acceleration** for embeddings
- Use **async processing** for better throughput

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open Pull Request

## 📞 Support

- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check `/docs` folder for detailed guides

## 🔗 Useful Links

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Haystack Documentation](https://haystack.deepset.ai/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)

---

*Happy Evaluating! 🎉* 