# RAG Evaluation Metrics Documentation

## Overview
This RAG evaluation system implements **13 specialized metrics** across three categories: Retriever Metrics, Generator Metrics, and Chunking Evaluation Metrics.

## Complete Metrics List

### **Retriever Metrics (3)**
Component-level evaluation of document retrieval performance.

1. **Contextual Relevancy**
   - **Purpose**: Measures if retrieved contexts are relevant to the query
   - **Implementation**: `ContextualRelevancyMetric` from DeepEval
   - **Threshold**: 0.7 (configurable)
   - **Description**: Evaluates whether the retrieved documents contain information that is directly relevant to answering the user's question

2. **Contextual Recall**
   - **Purpose**: Evaluates if all relevant information is retrieved
   - **Implementation**: `ContextualRecallMetric` from DeepEval
   - **Threshold**: 0.7 (configurable)
   - **Description**: Measures whether the retrieval system finds all the information needed to answer the query completely

3. **Contextual Precision**
   - **Purpose**: Assesses precision of the retrieval system without noise
   - **Implementation**: `ContextualPrecisionMetric` from DeepEval
   - **Threshold**: 0.7 (configurable)
   - **Description**: Evaluates whether the retrieved documents contain only relevant information without unnecessary content

### **Generator Metrics (2)**
Component-level evaluation of response generation quality.

4. **Answer Correctness**
   - **Purpose**: Custom GEval metric for factual accuracy
   - **Implementation**: Custom `GEval` metric
   - **Threshold**: 0.7 (configurable)
   - **Evaluation Criteria**: "Evaluate if the actual output's answer is correct and complete from the input and retrieved context"
   - **Steps**:
     - Check if the answer directly addresses the question
     - Verify factual accuracy against the retrieved context
     - Assess completeness of the response

5. **Citation Accuracy**
   - **Purpose**: Custom GEval metric for proper source attribution
   - **Implementation**: Custom `GEval` metric
   - **Threshold**: 0.8 (configurable)
   - **Evaluation Criteria**: "Check if citations are correct and relevant based on input and retrieved context"
   - **Steps**:
     - Verify that cited sources exist in the retrieved context
     - Check if citations are properly formatted
     - Assess relevance of cited content to the answer

### **Chunking Evaluation Metrics (8)**
Processing-level evaluation of document chunking and processing quality.

6. **Chunk Coherence**
   - **Purpose**: Evaluates semantic coherence and logical flow within chunks
   - **Implementation**: Custom evaluation logic
   - **Description**: Assesses whether text chunks maintain logical flow and semantic coherence

7. **Chunk Completeness**
   - **Purpose**: Assesses if chunks contain complete information units
   - **Implementation**: Custom evaluation logic
   - **Description**: Evaluates whether chunks contain complete sentences, paragraphs, or thoughts

8. **Overlap Quality**
   - **Purpose**: Evaluates quality of chunk overlaps for context preservation
   - **Implementation**: Custom evaluation logic
   - **Description**: Measures the effectiveness of chunk overlaps in maintaining context continuity

9. **Size Consistency**
   - **Purpose**: Monitors chunk size uniformity and consistency
   - **Implementation**: Custom evaluation logic
   - **Description**: Evaluates whether chunks maintain consistent sizes within acceptable ranges

10. **Semantic Boundaries**
    - **Purpose**: Checks if chunks respect sentence and paragraph boundaries
    - **Implementation**: Custom evaluation logic
    - **Description**: Assesses whether chunks break at natural semantic boundaries

11. **Text Cleanliness**
    - **Purpose**: Evaluates text formatting and processing quality
    - **Implementation**: Custom evaluation logic
    - **Description**: Measures the cleanliness and formatting quality of processed text

12. **Content Preservation**
    - **Purpose**: Ensures original meaning is maintained during processing
    - **Implementation**: Custom evaluation logic
    - **Description**: Evaluates whether the original content meaning is preserved during processing

13. **Metadata Quality**
    - **Purpose**: Assesses completeness and accuracy of chunk metadata
    - **Implementation**: Custom evaluation logic
    - **Description**: Evaluates the quality and completeness of metadata associated with chunks

## Metric Categories Summary

| Category | Count | Metrics |
|----------|-------|---------|
| **Retriever Metrics** | 3 | Contextual Relevancy, Contextual Recall, Contextual Precision |
| **Generator Metrics** | 2 | Answer Correctness, Citation Accuracy |
| **Chunking Evaluation Metrics** | 8 | Coherence, Completeness, Overlap Quality, Size Consistency, Semantic Boundaries, Text Cleanliness, Content Preservation, Metadata Quality |
| **TOTAL** | **13** | All specialized metrics |

## Usage

### Running Evaluations with All Metrics
```bash
# Run evaluation with all 13 metrics
python src/main.py --dataset sample --size 20

# Expected output includes:
# - 3 Retriever metrics
# - 2 Generator metrics  
# - 8 Chunking evaluation metrics
```

### Metric Configuration
Metrics can be configured in `config/eval_config.yaml`:
```yaml
retriever_metrics:
  contextual_relevancy:
    enabled: true
    threshold: 0.7
    
generator_metrics:
  answer_correctness:
    enabled: true
    threshold: 0.7
```

### Dashboard Integration
All 13 metrics are displayed in the Streamlit dashboard:
- **Overview**: Summary of all metric scores
- **Detailed Metrics**: Individual metric breakdowns
- **Historical Trends**: Performance tracking over time
- **Chunking Metrics**: Real-time chunking quality assessment

## Metric Validation

Each metric is validated through:
- **Unit Tests**: Individual metric testing
- **Integration Tests**: End-to-end evaluation testing
- **Performance Tests**: Latency and throughput validation
- **Quality Gates**: Automated performance monitoring

## Future Enhancements

Potential additional metrics to consider:
- **Response Relevance**: How relevant the generated answer is to the query
- **Response Completeness**: How complete the answer is
- **Response Fluency**: How natural and fluent the response is
- **Response Consistency**: How consistent the response is with the context
- **Response Accuracy**: How accurate the response is compared to ground truth

## Conclusion

The system provides **13 comprehensive metrics** that evaluate RAG systems at multiple levels:
- **Component Level**: Retriever and Generator performance
- **Processing Level**: Document chunking and processing quality
- **Quality Level**: Real-time optimization and monitoring

This comprehensive evaluation framework ensures thorough assessment of RAG system performance across all critical dimensions. 